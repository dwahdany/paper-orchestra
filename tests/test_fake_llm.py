from __future__ import annotations

import pytest
from pydantic import BaseModel

from paper_orchestra.llm.base import (
    Image,
    LLMParseError,
    Pdf,
    Text,
    ToolError,
    ToolSpec,
    Usage,
    UsageTracker,
    normalize_parts,
    sniff_image_media_type,
)
from paper_orchestra.llm.fake import FakeLLM
from paper_orchestra.llm.parsing import parse_json

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


class Answer(BaseModel):
    value: int


def test_image_media_type_is_sniffed_not_assumed():
    assert Image.from_bytes(PNG).media_type == "image/png"
    assert Image.from_bytes(JPEG).media_type == "image/jpeg"
    assert sniff_image_media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    with pytest.raises(ValueError):
        Image.from_bytes(b"not an image")


def test_normalize_parts():
    parts = normalize_parts(["a", Text("b"), Image.from_bytes(PNG), Pdf(b"%PDF-1.4")])
    assert [type(p).__name__ for p in parts] == ["Text", "Text", "Image", "Pdf"]
    assert normalize_parts("solo") == [Text("solo")]
    with pytest.raises(TypeError):
        normalize_parts([object()])


def test_mapping_responder_matches_label_then_prefix_then_default():
    client = FakeLLM({"outline": "exact", "literature.": "prefix", "*": "default"})
    assert client.generate("x", model="m", label="outline").text == "exact"
    assert client.generate("x", model="m", label="literature.discover").text == "prefix"
    assert client.generate("x", model="m", label="other").text == "default"
    assert client.labels() == ["outline", "literature.discover", "other"]


def test_schema_and_parser_paths():
    client = FakeLLM({"good": '{"value": 3}', "bad": '{"value": "x"}', "text": "plain"})
    assert client.generate("q", model="m", label="good", schema=Answer).parsed == Answer(value=3)
    assert client.generate("q", model="m", label="good", parser=parse_json).parsed == {"value": 3}
    with pytest.raises(LLMParseError):
        client.generate("q", model="m", label="bad", schema=Answer, parse_retries=1)
    assert len([r for r in client.requests if r.label == "bad"]) == 2
    response = client.generate(
        "q", model="m", label="text", parser=parse_json, require_parsed=False
    )
    assert response.parsed is None and response.text == "plain"


def test_tools_memory_events_and_usage():
    events: list[dict] = []
    calls: list[dict] = []

    def lookup(tool_input: dict) -> str:
        calls.append(tool_input)
        if tool_input.get("fail"):
            raise ToolError("no such page")
        return "page text"

    def responder(request):
        first = request.call_tool("lookup", {"q": "a"})
        second = request.call_tool("lookup", {"fail": True})
        return f"{first}|{second}|turn{request.turn}|{len(request.history)}"

    client = FakeLLM(responder)
    conversation = client.conversation(
        model="m",
        label="writer",
        tools=[ToolSpec("lookup", "look something up", {"type": "object"}, lookup)],
        on_event=events.append,
    )
    assert conversation.send("one").text == "page text|no such page|turn0|0"
    assert conversation.send(["two", Image.from_bytes(PNG)]).text.endswith("turn1|1")
    assert conversation.usage.requests == 2
    assert len(calls) == 4
    types = [e["type"] for e in events]
    assert types.count("tool_call") == 4 and types.count("tool_result") == 4
    assert [e["is_error"] for e in events if e["type"] == "tool_result"][:2] == [False, True]
    assert "[image image/png" in [e for e in events if e["type"] == "request"][1]["user"]


def test_queue_responder_runs_out_loudly():
    client = FakeLLM(["only one"])
    assert client.generate("q", model="m").text == "only one"
    with pytest.raises(AssertionError, match="ran out"):
        client.generate("q", model="m", label="next")


def test_usage_tracker_totals_by_label():
    tracker = UsageTracker()
    tracker.record("outline", "m1", Usage(input_tokens=10, output_tokens=5, requests=1))
    tracker.record("outline", "m1", Usage(input_tokens=1, output_tokens=1, requests=1))
    tracker.record("writing", "m2", Usage(cache_read_input_tokens=7, requests=1, web_searches=2))
    summary = tracker.to_dict()
    assert summary["total"]["input_tokens"] == 11 and summary["total"]["requests"] == 3
    assert summary["total"]["web_searches"] == 2
    assert [row["label"] for row in summary["by_label"]] == ["outline", "writing"]
