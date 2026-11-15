from __future__ import annotations

import threading

import pytest
from pydantic import BaseModel

from paper_orchestra.config import MemoryConfig, PipelineConfig
from paper_orchestra.llm.base import (
    Capabilities,
    LLMConfigError,
    LLMParseError,
    ToolError,
    UsageTracker,
    WebSearch,
)
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.llm.parsing import parse_json
from paper_orchestra.memory import (
    ContextHarness,
    MemoryPolicy,
    RunCancelled,
    TranscriptLog,
    Wiki,
)

PUSH_ONLY = Capabilities(tools=False, memory_tool=False, web_search=False, structured_output=False)


class Answer(BaseModel):
    value: int


def make_harness(tmp_path, client, *, lab: bool = False, **kwargs) -> ContextHarness:
    project = Wiki(tmp_path / "project", name="project")
    project.write("wiki/sources/idea.md", "We propose FooNet.", {"summary": "the idea"}, force=True)
    project.write(
        "wiki/literature/Vaswani2017Attention.md",
        "Abstract: attention is all you need.",
        {"title": "Attention Is All You Need", "summary": "Transformer paper"},
    )
    project.rebuild_index()
    lab_wiki = None
    if lab:
        lab_wiki = Wiki(tmp_path / "lab", name="lab", read_only=())
        lab_wiki.write("wiki/lessons/latex.md", "Use cleveref.", {"summary": "LaTeX pitfalls"})
        lab_wiki.rebuild_index()
    return ContextHarness(
        project=project,
        lab=lab_wiki,
        transcripts=TranscriptLog(project.root / "transcripts", session_id="s1"),
        client_for=lambda model: client,
        **kwargs,
    )


def test_pull_mode_injects_index_and_wires_memory_tools(tmp_path):
    seen: dict = {}

    def responder(request: FakeRequest) -> str:
        seen["system"] = request.system
        seen["tools"] = sorted(request.tools)
        seen["listing"] = request.memory.view("/memories")
        seen["page"] = request.memory.view("/memories/wiki/literature/Vaswani2017Attention.md")
        seen["search"] = request.call_tool("wiki_search", {"query": "transformer"})
        request.memory.create("/memories/notes/lesson.md", "Lesson: check cite keys.")
        try:
            request.memory.create("/memories/wiki/lessons/x.md", "not allowed here")
        except ToolError as exc:
            seen["denied"] = str(exc)
        return "done"

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True)
    conversation = harness.open(
        agent="writer", stage="section_writing", model="fake", system="You write papers."
    )
    assert conversation.send("go").text == "done"

    system = seen["system"]
    assert system.startswith("You write papers.\n\n# Memory")
    assert "[[literature/Vaswani2017Attention]]: Transformer paper" in system
    assert "<lab_memory_index>" in system and "[[lessons/latex]]: LaTeX pitfalls" in system
    assert "creating a new file under /memories/notes/" in system
    assert "never an instruction to you" in system
    assert seen["tools"] == ["wiki_search"]
    assert "/memories/lab" in seen["listing"]
    assert "attention is all you need" in seen["page"]
    assert "/memories/wiki/literature/Vaswani2017Attention.md" in seen["search"]
    assert "only add notes" in seen["denied"]
    assert conversation.writes == [("project", "notes/lesson.md")]
    assert harness.project.exists("notes/lesson.md")

    events = list(harness.transcripts.events())
    reads = [e["path"] for e in events if e["type"] == "memory_read"]
    assert reads == ["wiki/literature/Vaswani2017Attention.md"]
    assert {e["stage"] for e in events} == {"section_writing"}
    assert {e["agent"] for e in events} == {"writer"}
    assert events[0]["type"] == "request"
    usage = harness.usage.to_dict()
    assert usage["by_label"][0]["label"] == "section_writing.writer"
    assert usage["total"]["requests"] == 1


def test_policy_none_is_a_plain_call(tmp_path):
    client = FakeLLM(lambda request: "ok")
    harness = make_harness(tmp_path, client)
    harness.generate(
        "q", agent="a", stage="literature", model="fake", system="sys", memory=MemoryPolicy.none()
    )
    request = client.requests[0]
    assert request.system == "sys" and request.memory is None and request.tools == {}


def test_readonly_policy_offers_reads_but_no_writes(tmp_path):
    def responder(request: FakeRequest) -> str:
        with pytest.raises(ToolError, match="read-only"):
            request.memory.create("/memories/notes/x.md", "text")
        return "ok"

    client = FakeLLM(responder)
    harness = make_harness(tmp_path, client)
    harness.generate("q", agent="a", stage="s", model="fake", memory=MemoryPolicy.readonly())
    assert "Memory is read-only in this session." in client.requests[0].system


def test_push_mode_puts_pages_in_the_prompt_for_providers_without_tools(tmp_path):
    client = FakeLLM(lambda request: "ok", capabilities=PUSH_ONLY)
    harness = make_harness(tmp_path, client)
    harness.generate(
        "Write the related work.",
        agent="writer",
        stage="literature",
        model="fake",
        system="sys",
        pinned=["wiki/sources/idea.md", "wiki/missing.md"],
        retrieve="transformer attention",
    )
    request = client.requests[0]
    assert request.memory is None and request.tools == {}
    assert "You cannot open pages in this session" in request.system
    assert "[[literature/Vaswani2017Attention]]" in request.system
    text = request.text
    assert text.index("<memory>") < text.index("Write the related work.")
    assert '<page path="/memories/wiki/sources/idea.md">' in text and "FooNet" in text
    assert '<page path="/memories/wiki/literature/Vaswani2017Attention.md">' in text
    assert "wiki/missing.md" not in text


def test_context_block_respects_budget_and_only_first_turn_gets_it(tmp_path):
    client = FakeLLM(lambda request: "ok")
    harness = make_harness(tmp_path, client)
    harness.write_page("wiki/big.md", "x" * 5_000, {"summary": "big"})
    block = harness.context_block(
        pinned=["wiki/big.md", "wiki/sources/idea.md"], budget_chars=1_000
    )
    assert 'path="/memories/wiki/big.md" omitted=' in block and "FooNet" in block
    conversation = harness.open(agent="a", stage="s", model="fake")
    conversation.send("first", pinned=["wiki/sources/idea.md"])
    conversation.send("second", pinned=["wiki/sources/idea.md"])
    assert "<memory>" in client.requests[0].text and "<memory>" not in client.requests[1].text


def test_tools_on_a_provider_without_tool_support_is_a_config_error(tmp_path):
    from paper_orchestra.llm.base import ToolSpec

    harness = make_harness(tmp_path, FakeLLM(lambda r: "ok", capabilities=PUSH_ONLY))
    with pytest.raises(LLMConfigError, match="cannot use tools"):
        harness.open(
            agent="a",
            stage="s",
            model="fake",
            tools=[ToolSpec("t", "d", {"type": "object"}, lambda i: "")],
        )


def test_schema_native_then_text_fallback(tmp_path):
    native = FakeLLM(lambda request: '{"value": 7}')
    harness = make_harness(tmp_path, native)
    response = harness.generate("q", agent="a", stage="s", model="fake", schema=Answer)
    assert response.parsed == Answer(value=7)
    assert native.requests[0].schema is Answer and "JSON Schema" not in native.requests[0].text

    text_only = FakeLLM(lambda request: 'Sure:\n```json\n{"value": 9}\n```', capabilities=PUSH_ONLY)
    harness = make_harness(tmp_path / "b", text_only)
    response = harness.generate("q", agent="a", stage="s", model="fake", schema=Answer)
    assert response.parsed == Answer(value=9)
    assert text_only.requests[0].schema is None
    assert "conform to this JSON Schema" in text_only.requests[0].text


def test_schema_with_web_search_goes_straight_to_text(tmp_path):
    client = FakeLLM(lambda request: '```json\n{"value": 1}\n```')
    harness = make_harness(tmp_path, client)
    response = harness.generate(
        "q",
        agent="a",
        stage="literature",
        model="fake",
        schema=Answer,
        web_search=WebSearch(max_uses=3),
        memory=MemoryPolicy.none(),
    )
    assert response.parsed == Answer(value=1)
    request = client.requests[0]
    assert request.schema is None and request.web_search == WebSearch(max_uses=3)
    assert "conform to this JSON Schema" in request.text


def test_schema_fallback_when_provider_rejects_native_output(tmp_path):
    class Rejecting(FakeLLM):
        def _respond(self, request):
            if request.schema is not None:
                raise LLMConfigError("structured outputs disabled")
            return super()._respond(request)

    client = Rejecting(lambda request: '{"value": 4}')
    harness = make_harness(tmp_path, client)
    conversation = harness.open(agent="a", stage="s", model="fake", memory=MemoryPolicy.none())
    assert conversation.send("q", schema=Answer).parsed == Answer(value=4)
    assert conversation.send("again", schema=Answer).parsed == Answer(value=4)
    # After the first rejection the conversation stays in text mode.
    assert [r.schema for r in client.requests] == [None, None]


def test_generate_retries_parse_failures_then_raises(tmp_path):
    client = FakeLLM(["nope", "still nope", '{"value": 2}'])
    harness = make_harness(tmp_path, client)
    assert harness.generate(
        "q", agent="a", stage="s", model="fake", parser=parse_json, memory=MemoryPolicy.none()
    ).parsed == {"value": 2}
    client = FakeLLM(["nope", "nope"])
    harness = make_harness(tmp_path / "b", client)
    with pytest.raises(LLMParseError):
        harness.generate(
            "q",
            agent="a",
            stage="s",
            model="fake",
            schema=Answer,
            parse_retries=1,
            memory=MemoryPolicy.none(),
        )
    # Failed attempts are still paid for, so they are still counted.
    assert harness.usage.total().requests == 2


def test_pipeline_writes_are_visible_to_the_next_session(tmp_path):
    client = FakeLLM(lambda request: "ok")
    harness = make_harness(tmp_path, client)
    harness.write_page("wiki/sources/experimental_log.md", "acc 91.2", {"summary": "raw results"})
    harness.write_page("wiki/reviews/v1.md", "Overall 6", {"summary": "first review"})
    note = harness.note("Reviewer punished overclaiming.", stage="refinement", kind="lesson")
    harness.record(
        "compile", stage="section_writing", ok=False, errors=["Undefined control sequence"]
    )
    harness.generate("q", agent="a", stage="s", model="fake")
    system = client.requests[0].system
    assert "[[reviews/v1]]: first review" in system
    assert "[[sources/experimental_log]]: raw results" in system
    assert harness.project.is_read_only("wiki/sources/experimental_log.md")
    assert harness.read_page("wiki/reviews/v1.md").strip() == "Overall 6"
    assert harness.read_page("wiki/absent.md") is None
    assert note in [n.path for n in harness.project.pending_notes()]
    types = [e["type"] for e in harness.transcripts.events()]
    assert "note" in types and "compile" in types


def test_disabled_memory_makes_every_memory_call_a_no_op(tmp_path):
    client = FakeLLM(lambda request: "ok")
    harness = ContextHarness(client_for=lambda model: client, usage=UsageTracker())
    assert not harness.enabled
    harness.write_page("wiki/x.md", "body")
    assert harness.note("n", stage="s") is None
    assert harness.read_page("wiki/x.md") is None
    assert harness.context_block(pinned=["wiki/x.md"]) == ""
    harness.record("compile", stage="s")
    harness.refresh()
    assert harness.dream(stage="s", focus="f") is None and harness.finish() == []
    harness.generate("q", agent="a", stage="s", model="fake", system="sys", pinned=["wiki/x.md"])
    assert client.requests[0].system == "sys" and client.requests[0].memory is None
    assert not list(tmp_path.iterdir())


def test_create_from_config(tmp_path):
    cfg = PipelineConfig(
        base_dir=tmp_path / "run",
        template_dir=tmp_path,
        memory=MemoryConfig(lab_dir=tmp_path / "lab", dream="end"),
    )
    harness = ContextHarness.create(cfg, client_for=lambda model: FakeLLM(["x"]))
    assert harness.project.root == tmp_path / "run" / "memory"
    assert harness.lab.root == tmp_path / "lab"
    # Records written by the pipeline are read-only to every model session.
    for section in ("sources", "literature", "reviews", "figures", "paper"):
        assert harness.project.is_read_only(f"wiki/{section}/x.md")
    assert not harness.project.is_read_only("wiki/lessons/x.md")
    assert "[[sources/experimental_log]]" in harness.project.standing_notes()
    assert harness.lab.index_max_lines == 80 and harness.lab.standing_max_lines == 15
    # Data kept for pipeline code only sits beside the lab wiki, where no session can read it.
    assert harness.cache.root == tmp_path / "lab" / "cache"
    harness.cache.put("semantic-scholar", "a title", {"abstract": "Secret abstract"})
    assert harness.lab.list("wiki") == [] and harness.lab.grep("Secret") == []
    # The run's inputs and drafts are scanned before anything is shared with the lab.
    draft = tmp_path / "run" / "latex_writeup" / "raw_draft_paper.tex"
    draft.parent.mkdir(parents=True)
    draft.write_text("We introduce GlimmerNet.", encoding="utf-8")
    assert "GlimmerNet" in harness.leak_scanner().reason("GlimmerNet tables overflow.")
    # Re-creating the harness (a resumed run) keeps standing notes a dream has rewritten.
    harness.project.set_standing_notes("- curated rule")
    again = ContextHarness.create(cfg, client_for=lambda model: FakeLLM(["x"]))
    assert again.project.standing_notes() == "- curated rule"
    assert harness.transcripts.directory == tmp_path / "run" / "memory" / "transcripts"
    assert harness.dream_mode == "end" and harness.default_effort == "high"

    cfg.memory = MemoryConfig(lab=False)
    assert ContextHarness.create(cfg, client_for=lambda m: None).lab is None
    cfg.memory = MemoryConfig(enabled=False)
    off = ContextHarness.create(cfg, client_for=lambda m: None)
    assert off.project is None and off.transcripts is None and not off.enabled


def test_concurrent_sessions_share_the_harness_safely(tmp_path):
    def responder(request: FakeRequest) -> str:
        request.memory.create(f"/memories/notes/{request.label.replace('.', '-')}.md", "note")
        return "ok"

    harness = make_harness(tmp_path, FakeLLM(responder))

    def work(i: int) -> None:
        harness.write_page(f"wiki/figures/fig{i}.md", "caption", {"summary": f"figure {i}"})
        harness.generate("q", agent=f"w{i}", stage="plotting", model="fake")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(harness.project.pending_notes()) == 12
    assert len(harness.project.list("wiki/figures")) == 12
    assert harness.usage.total().requests == 12
    harness.refresh()
    assert harness.project.lint().ok


def test_oversize_pages_are_stored_in_linked_parts(tmp_path):
    client = FakeLLM(lambda request: "ok")
    project = Wiki(tmp_path / "project", max_page_bytes=6_000)
    harness = ContextHarness(project=project, client_for=lambda model: client)
    log_text = "".join(f"| run {i} | {i * 1.5:.1f} |\n" for i in range(900))
    harness.write_page("wiki/sources/experimental_log.md", log_text, {"summary": "raw results"})
    parts = [i.path for i in project.list("wiki/sources") if "-part-" in i.path]
    assert len(parts) >= 3
    assert (
        "".join(
            project.read(p).body for p in sorted(parts, key=lambda p: int(p[:-3].rsplit("-", 1)[1]))
        )
        == log_text
    )
    parent = project.read("wiki/sources/experimental_log.md")
    assert len(parent.links()) == len(parts)
    assert project.lint().broken_links == []


def test_lab_only_session_never_sees_the_project(tmp_path):
    seen: dict = {}

    def responder(request: FakeRequest) -> str:
        seen["system"] = request.system
        seen["tools"] = sorted(request.tools)
        seen["listing"] = request.memory.view("/memories")
        for hidden in ("/memories/wiki/sources/idea.md", "/memories/lab/wiki/lessons/latex.md"):
            with pytest.raises(ToolError):
                request.memory.view(hidden)
        request.memory.create("/memories/wiki/lessons/new.md", "Prefer booktabs rules in tables.")
        with pytest.raises(ToolError, match="number from this paper"):
            request.memory.create("/memories/wiki/lessons/leak.md", "The method reaches 91.27.")
        return "ok"

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True)
    harness.write_page("wiki/sources/experimental_log.md", "Accuracy 91.27 on the benchmark.")
    conversation = harness.open(
        agent="dreamer",
        stage="finalize",
        model="fake",
        system="lab",
        memory=MemoryPolicy(write="all", transcripts=False),
        memory_root="lab",
    )
    conversation.send("go")
    assert "FooNet" not in seen["system"] and "[[lessons/latex]]" in seen["system"]
    assert "sources" not in seen["listing"] and "/memories/wiki/lessons" in seen["listing"]
    assert seen["tools"] == ["wiki_search"]
    assert conversation.writes == [("lab", "wiki/lessons/new.md")]
    assert harness.lab.exists("wiki/lessons/new.md") and not harness.lab.exists(
        "wiki/lessons/leak.md"
    )
    assert not harness.project.exists("wiki/lessons/new.md")


def test_delete_and_list_pages_from_pipeline_code(tmp_path):
    client = FakeLLM(lambda request: "ok")
    harness = make_harness(tmp_path, client)
    assert "wiki/literature/Vaswani2017Attention.md" in harness.list_pages("wiki/literature")
    assert harness.delete_page("wiki/literature/Vaswani2017Attention.md")
    assert not harness.delete_page("wiki/literature/Vaswani2017Attention.md")
    assert harness.list_pages("wiki/literature") == []
    harness.generate("q", agent="a", stage="s", model="fake")
    assert "Vaswani2017Attention" not in client.requests[0].system
    assert ContextHarness(client_for=lambda m: client).list_pages() == []


def test_lab_consolidation_can_be_left_to_a_separate_pass(tmp_path):
    labels: list[str] = []

    def responder(request: FakeRequest) -> str:
        labels.append(request.label)
        if request.label == "dream.project":
            request.memory.create(
                "/memories/wiki/lessons/latex.md",
                "---\nsummary: LaTeX\nportable: true\n---\nLoad cleveref before using it.",
            )
        return "ok"

    harness = make_harness(
        tmp_path,
        FakeLLM(responder),
        lab=True,
        dream_mode="end",
        dream_model="fake",
        consolidate_lab=False,
    )
    harness.note("cleveref was missing", stage="section_writing", kind="lesson")
    reports = harness.finish()
    # The run only adds its lesson to the lab as a note ...
    assert [r.scope for r in reports] == ["project"] and labels == ["dream.project"]
    assert len(harness.lab.pending_notes()) == 1
    # ... and a single later pass folds pending notes into pages.
    report = harness.consolidate_lab_memory()
    assert report.scope == "lab" and report.llm_used and report.notes_archived == 1
    assert labels == ["dream.project", "dream.lab"]


def test_runs_read_the_lab_index_but_never_rewrite_it(tmp_path):
    client = FakeLLM(lambda request: "ok")
    harness = make_harness(tmp_path, client, lab=True)
    index = harness.lab.root / "MEMORY.md"
    # Another process (the lab consolidation) changes the shared wiki while this run works.
    index.write_text(index.read_text().replace("_None yet._", "- Rule from the consolidator."))
    (harness.lab.root / "wiki" / "lessons" / "new.md").write_text("A [[Latex]] lesson.")
    consolidated = index.read_text()

    harness.note("a lesson", stage="outline", kind="lesson")
    harness.write_page("wiki/paper/outline.md", "The outline.", {"summary": "outline"})
    harness.refresh()
    harness.generate("task", agent="writer", stage="outline", model="fake")
    harness.dream(stage="outline", focus="f")
    harness.refresh()

    assert index.read_text() == consolidated
    assert (harness.lab.root / "wiki" / "lessons" / "new.md").read_text() == "A [[Latex]] lesson."
    # The run still sees the consolidator's standing notes, and its own index is maintained.
    assert "Rule from the consolidator." in client.requests[0].system
    assert "[[paper/outline]]" in harness.project.injected_context()


def test_task_sessions_do_not_see_unchecked_lab_notes(tmp_path):
    def responder(request: FakeRequest) -> str:
        listing = request.memory.view("/memories")
        assert "/memories/lab/wiki" in listing and "/memories/lab/notes" not in listing
        assert "/memories/notes" in listing  # this run's own notes are visible
        with pytest.raises(ToolError, match="does not exist"):
            request.memory.view("/memories/lab/notes")
        with pytest.raises(ToolError, match="does not exist"):
            request.memory.view("/memories/lab/" + pending)
        for exact in (False, True):
            found = request.call_tool("wiki_search", {"query": "quokka", "exact": exact})
            assert "/memories/notes/" in found and "/memories/lab" not in found
        assert "/memories/lab/wiki/lessons/latex.md" in request.call_tool(
            "wiki_search", {"query": "cleveref", "scope": "lab"}
        )
        return "ok"

    harness = make_harness(tmp_path, FakeLLM(responder), lab=True)
    pending = harness.lab.add_note("An unchecked quokka lesson from another run.", kind="lesson")
    harness.note("Our own quokka observation.", stage="outline")
    harness.generate("task", agent="writer", stage="outline", model="fake")


def test_push_mode_supplies_lessons_when_the_caller_names_nothing(tmp_path):
    client = FakeLLM(lambda request: "ok", capabilities=PUSH_ONLY)
    harness = make_harness(tmp_path, client, lab=True)
    harness.generate("first task", agent="writer", stage="outline", model="fake")
    # Nothing has been learned yet: only the lab's lessons are there to give.
    first = client.requests[0]
    assert "You cannot open pages in this session" in first.system
    assert '<page path="/memories/lab/wiki/lessons/latex.md">' in first.text
    assert "sources/idea" not in first.text  # inputs are in the agents' own prompts

    harness.project.write(
        "wiki/lessons/revisions.md", "Late limitations paragraphs lowered scores.", {"summary": "s"}
    )
    harness.project.write("wiki/synthesis/themes.md", "Two clusters of work.", {"summary": "s"})
    harness.note("v2 REJECTED: Overall 6 -> 5", stage="refinement", kind="lesson")
    harness.note("stage finished", stage="refinement", kind="observation")
    harness.generate("second task", agent="reviser", stage="refinement", model="fake")
    text = client.requests[1].text
    for fragment in (
        "Late limitations paragraphs lowered scores.",
        "Two clusters of work.",
        "v2 REJECTED: Overall 6 -> 5",
        "Use cleveref.",
    ):
        assert fragment in text
    assert "stage finished" not in text and text.rstrip().endswith("second task")
    # A session with memory off, or one that names its pages, gets exactly what it asked for.
    harness.generate("plain", agent="a", stage="s", model="fake", memory=MemoryPolicy.none())
    assert "<memory>" not in client.requests[2].text
    harness.generate("pinned", agent="a", stage="s", model="fake", pinned=["wiki/sources/idea.md"])
    assert "We propose FooNet." in client.requests[3].text
    assert "Late limitations" not in client.requests[3].text
    # A session that must not see the lab is not handed its pages either.
    harness.generate(
        "no lab", agent="a", stage="s", model="fake", memory=MemoryPolicy(write="none", lab=False)
    )
    assert "Use cleveref." not in client.requests[4].text
    assert "lab_memory_index" not in client.requests[4].system


def test_a_cancelled_run_stops_at_the_next_model_call(tmp_path):
    client = FakeLLM(lambda request: "ok")
    harness = make_harness(tmp_path, client)
    conversation = harness.open(agent="writer", stage="outline", model="fake")
    conversation.send("first")
    assert not harness.cancelled
    harness.cancel()
    with pytest.raises(RunCancelled):
        conversation.send("second")
    with pytest.raises(RunCancelled):
        harness.generate("third", agent="writer", stage="outline", model="fake")
    assert len(client.requests) == 1 and harness.cancelled
    # Deterministic bookkeeping still works, so the run can be closed cleanly.
    harness.note("stopped by the user", stage="outline")
    harness.refresh()


# -- the session that maintains the shared memory -----------------------------


def lab_harness(tmp_path, responder):
    project = Wiki(tmp_path / "project", name="project")
    project.write(
        "wiki/sources/idea.md", "We propose FooNet for long video.", {"summary": "idea"}, force=True
    )
    project.write(
        "wiki/sources/experimental_log.md", "FooNet reaches 91.27.", {"summary": "log"}, force=True
    )
    project.rebuild_index()
    lab = Wiki(tmp_path / "lab", name="lab", read_only=())
    lab.write("wiki/lessons/latex.md", "Use cleveref.", {"summary": "LaTeX pitfalls"})
    lab.rebuild_index()
    return ContextHarness(
        project=project,
        lab=lab,
        transcripts=TranscriptLog(project.root / "transcripts", session_id="s1"),
        client_for=lambda model: FakeLLM(responder),
    )


def test_lab_only_session_cannot_search_the_project(tmp_path):
    seen: dict = {}

    def responder(request: FakeRequest) -> str:
        seen["schema"] = request.tools["wiki_search"].input_schema["properties"]
        seen["keyword"] = request.call_tool("wiki_search", {"query": "FooNet long video"})
        seen["exact"] = request.call_tool("wiki_search", {"query": "91.27", "exact": True})
        seen["own"] = request.call_tool("wiki_search", {"query": "cleveref"})
        return "ok"

    harness = lab_harness(tmp_path, responder)
    harness.open(
        agent="dreamer",
        stage="finalize",
        model="fake",
        memory_root="lab",
        memory=MemoryPolicy(write="all"),
    ).send("go")
    assert seen["keyword"] == "No matching pages."
    assert seen["exact"] == "No line contains that text verbatim."
    assert "/memories/wiki/lessons/latex.md" in seen["own"]
    assert "scope" not in seen["schema"]  # one wiki only: nothing to choose between


def test_lab_guard_also_vets_edits(tmp_path):

    def responder(request: FakeRequest) -> str:
        request.memory.create("/memories/wiki/lessons/tables.md", "Prefer booktabs.\nSecond line.")
        with pytest.raises(ToolError, match="number from this paper"):
            request.memory.str_replace(
                "/memories/wiki/lessons/tables.md", "Prefer booktabs.", "It reached 91.27."
            )
        with pytest.raises(ToolError, match="number from this paper"):
            request.memory.insert("/memories/wiki/lessons/tables.md", 4, "It reached 91.27.")
        return "ok"

    harness = lab_harness(tmp_path, responder)
    harness.open(
        agent="dreamer",
        stage="finalize",
        model="fake",
        memory_root="lab",
        memory=MemoryPolicy(write="all"),
    ).send("go")
    assert "91.27" not in harness.lab.read_text("wiki/lessons/tables.md")


def test_a_run_that_leaves_lab_consolidation_to_others_writes_no_lab_index(tmp_path):
    cfg = PipelineConfig(
        base_dir=tmp_path / "run",
        template_dir=tmp_path,
        memory=MemoryConfig(lab_dir=tmp_path / "lab", lab_consolidate=False),
    )
    client = FakeLLM(lambda request: "ok")
    harness = ContextHarness.create(cfg, client_for=lambda model: client)
    harness.generate("task", agent="writer", stage="outline", model="fake")
    harness.lab.add_note("A lesson.", kind="lesson")
    harness.refresh()
    # The run saw an (empty) lab index and added a note, but created no shared index file.
    assert "Nothing has been consolidated yet" in client.requests[0].system
    assert not (tmp_path / "lab" / "MEMORY.md").exists()
    assert len(harness.lab.pending_notes()) == 1
    # The pass that maintains the lab writes it.
    maintained = ContextHarness.create(
        PipelineConfig(
            base_dir=tmp_path / "other",
            template_dir=tmp_path,
            memory=MemoryConfig(lab_dir=tmp_path / "lab", dream="off"),
        ),
        client_for=lambda model: client,
    )
    maintained.consolidate_lab_memory()
    assert (tmp_path / "lab" / "MEMORY.md").is_file()
