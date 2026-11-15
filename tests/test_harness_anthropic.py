"""The context harness on top of the real Anthropic client (mock transport)."""

# ruff: noqa: F811 - pytest fixtures are imported and then named as parameters

from __future__ import annotations

from paper_orchestra.memory import ContextHarness, MemoryPolicy, TranscriptLog, Wiki
from test_llm_anthropic import (  # noqa: F401 - fixtures
    EPHEMERAL,
    MODEL,
    _hermetic_environment,
    api,
    client,
    msg,
    text,
    tool_use,
)


def make_harness(tmp_path, client) -> ContextHarness:
    project = Wiki(tmp_path / "project", name="project")
    project.write("wiki/sources/idea.md", "We propose FooNet.", {"summary": "idea"}, force=True)
    project.rebuild_index()
    return ContextHarness(
        project=project,
        transcripts=TranscriptLog(project.root / "transcripts", session_id="s1"),
        client_for=lambda model: client,
    )


def test_memory_free_one_shot_calls_do_not_pay_for_a_cache_nothing_reads(tmp_path, api, client):
    harness = make_harness(tmp_path, client)
    api.script.append(msg([text("a score")]))
    harness.generate(
        "Review this.",
        agent="reviewer",
        stage="refinement",
        model=MODEL,
        system="You review.",
        memory=MemoryPolicy.none(),
    )
    (body,) = api.requests
    assert "cache_control" not in body and "tools" not in body
    assert body["system"][0]["cache_control"] == EPHEMERAL  # shared by every such call


def test_one_shot_calls_that_read_memory_keep_the_cache_across_their_tool_loop(
    tmp_path,
    api,
    client,
):
    harness = make_harness(tmp_path, client)
    api.script += [
        msg([tool_use("tu_1", "memory", {"command": "view", "path": "/memories"})], "tool_use"),
        msg([text("an outline")]),
    ]
    response = harness.generate("Outline it.", agent="outliner", stage="outline", model=MODEL)
    assert response.text == "an outline"
    # One turn, two requests: the second reads back what the first wrote.
    assert [body["cache_control"] for body in api.requests] == [EPHEMERAL] * 2


def test_conversations_keep_the_cache_on_every_turn(tmp_path, api, client):
    harness = make_harness(tmp_path, client)
    api.script += [msg([text("a draft")]), msg([text("a repaired draft")])]
    conversation = harness.open(
        agent="writer", stage="section_writing", model=MODEL, memory=MemoryPolicy.none()
    )
    conversation.send("Write the paper.")
    conversation.send("Fix the compile error.")
    assert [body["cache_control"] for body in api.requests] == [EPHEMERAL] * 2
