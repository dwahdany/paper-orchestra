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
from paper_orchestra.memory import ContextHarness, MemoryPolicy, TranscriptLog, Wiki

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
    assert "creating a file under /memories/notes/" in system
    assert seen["tools"] == ["wiki_search"]
    assert "/memories/lab" in seen["listing"]
    assert "attention is all you need" in seen["page"]
    assert "/memories/wiki/literature/Vaswani2017Attention.md" in seen["search"]
    assert "only add notes" in seen["denied"]
    assert conversation.writes == [("project", "notes/lesson.md")]
    assert harness.project.exists("notes/lesson.md")

    events = list(harness.transcripts.events())
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
    assert "you cannot open other pages" in request.system
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
    assert harness.lab.root == tmp_path / "lab" and harness.lab.read_only == ()
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
