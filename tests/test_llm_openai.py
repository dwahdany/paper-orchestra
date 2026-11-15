"""Offline tests for the OpenAI client.

The real openai SDK runs end to end: only its HTTP transport is replaced by a
scripted one, so request bodies and response parsing are the SDK's own.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pymupdf
import pytest

openai = pytest.importorskip("openai")

from openai import _base_client  # noqa: E402

from paper_orchestra.llm.base import (  # noqa: E402
    Capabilities,
    Image,
    LLMConfigError,
    LLMError,
    LLMParseError,
    LLMRefusal,
    LLMTruncated,
    Pdf,
    ToolSpec,
    Usage,
    WebSearch,
)
from paper_orchestra.llm.openai_client import OpenAIClient, RetryPolicy  # noqa: E402
from paper_orchestra.llm.parsing import parse_json  # noqa: E402

# The SDK is built on httpx or on its httpx2 fork, depending on the release.
HTTPX = getattr(_base_client, "httpx2", None) or _base_client.httpx

MODEL = "gpt-4o"


class Wire:
    """Scripted transport: records every request body and replays queued responses."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self.script: list[Any] = []
        self.client_kwargs: dict[str, Any] = {}

    def queue(self, *items: Any) -> None:
        self.script.extend(items)

    def __call__(self, request: Any) -> Any:
        self.requests.append(json.loads(request.content))
        self.urls.append(str(request.url))
        assert self.script, "the client sent a request the test did not script"
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(
    text: str | None = "hello",
    *,
    finish: str = "stop",
    refusal: str | None = None,
    usage: dict[str, Any] | None = None,
) -> Any:
    message = {"role": "assistant", "content": text, "refusal": refusal}
    return HTTPX.Response(
        200,
        headers={"x-request-id": "req_ok"},
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test-2026-01-01",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


def api_error(
    status: int,
    message: str = "boom",
    *,
    code: str | None = None,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    error = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    return HTTPX.Response(
        status, headers={"x-request-id": "req_err", **(headers or {})}, json={"error": error}
    )


def roles(body: dict[str, Any]) -> list[str]:
    return [message["role"] for message in body["messages"]]


def user_text(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def make_pdf(text: str) -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    return document.tobytes()


@pytest.fixture
def wire(monkeypatch):
    """Route every ``openai.OpenAI`` the code under test builds through a scripted transport."""
    wire = Wire()
    real_client = openai.OpenAI

    def offline_client(**kwargs: Any) -> Any:
        wire.client_kwargs = dict(kwargs)
        return real_client(http_client=HTTPX.Client(transport=HTTPX.MockTransport(wire)), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", offline_client)
    return wire


def build(monkeypatch, wire: Wire, **options: Any) -> SimpleNamespace:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    # rng=0 pins the jitter to the lower bound: half of each exponential step.
    client = OpenAIClient(sleep=sleeps.append, rng=lambda: 0.0, **options)
    return SimpleNamespace(client=client, wire=wire, sleeps=sleeps)


@pytest.fixture
def api(monkeypatch, wire):
    return build(monkeypatch, wire)


# -- construction ---------------------------------------------------------------------


def test_missing_credentials_error_names_the_environment_variable(wire):
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        OpenAIClient()
    assert wire.client_kwargs == {}, "no SDK client may be built without credentials"


def test_missing_sdk_error_gives_the_install_command(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(LLMConfigError, match="uv sync --extra openai"):
        OpenAIClient()


def test_importing_the_module_does_not_import_the_sdk():
    code = (
        "import sys, paper_orchestra.llm.openai_client\n"
        "assert 'openai' not in sys.modules, 'openai was imported eagerly'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_sdk_client_gets_the_key_a_long_timeout_and_no_sdk_retries(api):
    kwargs = api.wire.client_kwargs
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["max_retries"] == 0, "retries belong to the client's own policy"
    assert kwargs["timeout"].read == 30 * 60

    description = api.client.describe()
    assert description == {
        "provider": "openai",
        "credential_source": "api_key",
        "sdk_version": openai.__version__,
    }
    assert "sk-test" not in repr(description)


def test_capabilities_match_the_spec():
    assert OpenAIClient.capabilities == Capabilities(pdf_input=False)
    assert not OpenAIClient.capabilities.web_search
    assert not OpenAIClient.capabilities.structured_output


# -- requests and responses -----------------------------------------------------------


def test_text_turn_places_system_and_options_and_reports_usage_and_events(api):
    api.wire.queue(
        ok(
            "Hello there",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 7,
                "total_tokens": 107,
                "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 10},
            },
        )
    )
    events: list[dict[str, Any]] = []
    conversation = api.client.conversation(
        model=MODEL,
        system="You are terse.",
        temperature=0.2,
        max_tokens=512,
        effort="high",
        label="outline.writer",
        on_event=events.append,
    )
    response = conversation.send("Say hello")

    assert api.wire.urls[0].endswith("/chat/completions")
    assert api.wire.requests[0] == {
        "model": MODEL,
        "messages": [{"role": "system", "content": "You are terse."}, user_text("Say hello")],
        "temperature": 0.2,
        "max_completion_tokens": 512,
    }

    assert response.text == "Hello there"
    assert response.parsed is None
    assert response.stop_reason == "end_turn"
    assert response.model == "gpt-test-2026-01-01"
    assert response.request_id == "req_ok"
    # Cache reads and writes are reported inside prompt_tokens; Usage keeps them apart.
    expected = Usage(
        input_tokens=50,
        output_tokens=7,
        cache_read_input_tokens=40,
        cache_creation_input_tokens=10,
        requests=1,
    )
    assert response.usage == expected
    assert conversation.usage == expected

    assert events == [
        {"type": "request", "label": "outline.writer", "model": MODEL, "user": "Say hello"},
        {
            "type": "assistant",
            "text": "Hello there",
            "stop_reason": "end_turn",
            "model": "gpt-test-2026-01-01",
            "request_id": "req_ok",
        },
        {"type": "usage", **expected.to_dict()},
    ]
    json.dumps(events)


@pytest.mark.parametrize("model", ["o1", "o3-mini", "o4-mini"])
def test_o_series_models_use_the_developer_role_and_no_temperature(api, model):
    api.wire.queue(ok("done"))
    api.client.generate("Think.", model=model, system="Be careful.", temperature=0.3)

    body = api.wire.requests[0]
    assert body["messages"][0] == {"role": "developer", "content": "Be careful."}
    assert "temperature" not in body


def test_a_rejected_temperature_is_dropped_and_remembered_for_the_model(api, caplog):
    rejection = api_error(
        400,
        "Unsupported value: 'temperature' does not support 0.3 with this model.",
        code="unsupported_value",
        param="temperature",
    )
    api.wire.queue(rejection, ok("first"), ok("second"), ok("other model"))

    with caplog.at_level(logging.WARNING, logger="paper_orchestra.llm.openai_client"):
        first = api.client.generate("q", model="gpt-5", temperature=0.3)
    assert first.text == "first"
    assert "temperature" in api.wire.requests[0]
    assert "temperature" not in api.wire.requests[1]
    assert api.sleeps == [], "a parameter fix is not a backoff"
    assert "rejected temperature" in caplog.text

    # Later conversations on that model skip the doomed request; other models are unaffected.
    api.client.generate("q", model="gpt-5", temperature=0.3)
    api.client.generate("q", model=MODEL, temperature=0.3)
    assert len(api.wire.requests) == 4
    assert "temperature" not in api.wire.requests[2]
    assert api.wire.requests[3]["temperature"] == 0.3


def test_an_out_of_range_temperature_is_dropped_once_but_not_remembered(api):
    rejection = api_error(
        400,
        "Invalid 'temperature': decimal above maximum value.",
        code="decimal_above_max_value",
        param="temperature",
    )
    api.wire.queue(rejection, ok("first"), ok("second"))

    api.client.generate("q", model=MODEL, temperature=3.0)
    api.client.generate("q", model=MODEL, temperature=0.5)

    sent = [body.get("temperature") for body in api.wire.requests]
    assert sent == [3.0, None, 0.5], "one bad value must not disable temperature for the model"


def test_a_rejected_token_limit_is_dropped_for_that_request(api):
    rejection = api_error(
        400,
        "max_tokens is too large: 64000. This model supports at most 16384 completion tokens.",
        code="invalid_value",
        param="max_tokens",
    )
    api.wire.queue(rejection, ok("fits now"))

    response = api.client.generate("q", model=MODEL, max_tokens=64000)

    assert response.text == "fits now"
    assert api.wire.requests[0]["max_completion_tokens"] == 64000
    assert "max_completion_tokens" not in api.wire.requests[1]


def test_images_are_data_urls_with_their_own_media_type(api):
    webp = Image(data=b"RIFF\x00\x00\x00\x00WEBPVP8 ", media_type="image/webp")
    png = Image.from_bytes(b"\x89PNG\r\n\x1a\n not really a png")
    api.wire.queue(ok("seen"))
    events: list[dict[str, Any]] = []

    api.client.generate(["Compare.", webp, png], model=MODEL, on_event=events.append)

    content = api.wire.requests[0]["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "Compare."},
        {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{webp.to_base64()}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png.to_base64()}"}},
    ]
    assert events[0]["user"] == (
        f"Compare.\n[image image/webp, {len(webp.data)} bytes]\n"
        f"[image image/png, {len(png.data)} bytes]"
    )


def test_pdf_parts_are_sent_as_extracted_text(api):
    body_text = "Attention is all you need. " * 10
    pdf = Pdf(data=make_pdf(body_text), title="paper.pdf")
    api.wire.queue(ok("reviewed"))
    events: list[dict[str, Any]] = []

    api.client.generate([pdf, "Review this paper."], model=MODEL, on_event=events.append)

    document, task = api.wire.requests[0]["messages"][0]["content"]
    assert document["type"] == "text"
    assert document["text"].startswith('<document title="paper.pdf">\n')
    assert document["text"].endswith("\n</document>")
    assert "Attention is all you need." in document["text"]
    assert task == {"type": "text", "text": "Review this paper."}
    assert events[0]["user"] == f"[pdf paper.pdf, {len(pdf.data)} bytes]\nReview this paper."


def test_a_pdf_without_extractable_text_fails_before_any_request(api):
    with pytest.raises(LLMError, match=r"scan\.pdf"):
        api.client.generate([Pdf(data=b"not a pdf", title="scan.pdf"), "Review."], model=MODEL)
    assert api.wire.requests == []


def test_multi_turn_resends_the_history_unchanged(api):
    api.wire.queue(ok("4"), ok("5"))
    conversation = api.client.conversation(model=MODEL, system="Count.")

    conversation.send("2+2?")
    assert conversation.send("plus one?").text == "5"

    first, second = (body["messages"] for body in api.wire.requests)
    assert second[:2] == first
    assert second == [
        {"role": "system", "content": "Count."},
        user_text("2+2?"),
        {"role": "assistant", "content": "4"},
        user_text("plus one?"),
    ]
    assert conversation.usage == Usage(input_tokens=20, output_tokens=10, requests=2)


def test_send_can_override_max_tokens_for_one_turn(api):
    api.wire.queue(ok("a"), ok("b"))
    conversation = api.client.conversation(model=MODEL, max_tokens=100)
    conversation.send("one", max_tokens=7)
    conversation.send("two")
    assert [body["max_completion_tokens"] for body in api.wire.requests] == [7, 100]


def test_empty_message_is_rejected_without_a_request(api):
    with pytest.raises(ValueError, match="empty message"):
        api.client.conversation(model=MODEL).send("")
    assert api.wire.requests == []


# -- parsing --------------------------------------------------------------------------


def test_parser_result_parse_failure_and_relaxed_parsing(api):
    api.wire.queue(ok('```json\n{"a": 1}\n```'), ok("not json at all"), ok("still not json"))
    events: list[dict[str, Any]] = []
    conversation = api.client.conversation(model=MODEL, on_event=events.append)

    assert conversation.send("q", parser=parse_json).parsed == {"a": 1}

    with pytest.raises(LLMParseError) as err:
        conversation.send("again", parser=parse_json)
    assert err.value.raw_text == "not json at all"
    assert events[-1]["type"] == "error"
    assert events[-1]["error"].startswith("LLMParseError: ")

    relaxed = conversation.send("once more", parser=parse_json, require_parsed=False)
    assert (relaxed.parsed, relaxed.text) == (None, "still not json")
    # The reply that failed to parse is a real turn: a follow-up can refer to it.
    assert roles(api.wire.requests[2]) == ["user", "assistant", "user", "assistant", "user"]
    assert api.wire.requests[2]["messages"][3] == {
        "role": "assistant",
        "content": "not json at all",
    }


def test_generate_asks_again_from_scratch_when_parsing_fails(api):
    api.wire.queue(ok("nope"), ok('{"ok": true}'))
    response = api.client.generate("q", model=MODEL, parser=parse_json, parse_retries=1)
    assert response.parsed == {"ok": True}
    assert [roles(body) for body in api.wire.requests] == [["user"], ["user"]]


def test_unsupported_options_raise_config_errors_before_any_request(api):
    tool = ToolSpec(name="t", description="d", input_schema={"type": "object"}, handler=str)
    with pytest.raises(LLMConfigError, match="tools"):
        api.client.conversation(model=MODEL, tools=[tool])
    with pytest.raises(LLMConfigError, match="memory"):
        api.client.conversation(model=MODEL, memory=object())
    with pytest.raises(LLMConfigError, match="web search"):
        api.client.conversation(model=MODEL, web_search=WebSearch())
    with pytest.raises(LLMConfigError, match="structured output"):
        api.client.conversation(model=MODEL).send("q", schema={"type": "object"})
    with pytest.raises(LLMConfigError, match="cannot generate images"):
        api.client.generate_image("A diagram", model="gpt-image-1")
    assert api.wire.requests == []


# -- retries and failures -------------------------------------------------------------


def test_retry_policy_backs_off_exponentially_with_jitter_and_a_cap():
    policy = RetryPolicy(max_attempts=5, base_delay=4.0, max_delay=60.0)
    # The first retry already waits (the old code retried immediately).
    assert policy.delay(1, 0.0) == 2.0
    assert policy.delay(1, 1.0) == 4.0
    assert [policy.delay(n, 1.0) for n in range(1, 7)] == [4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert all(8.0 <= policy.delay(3, j / 10) <= 16.0 for j in range(10))
    # A server hint lengthens the wait up to the cap and never shortens it.
    assert policy.delay(1, 0.0, hint=30.0) == 30.0
    assert policy.delay(1, 0.0, hint=600.0) == 60.0
    assert policy.delay(4, 1.0, hint=1.0) == 32.0
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_transient_failures_are_retried_with_backoff(api):
    api.wire.queue(
        api_error(429, "slow down", code="rate_limit_exceeded", headers={"retry-after": "7"}),
        HTTPX.Response(500, text="oops"),
        HTTPX.ConnectError("connection refused"),
        HTTPX.ReadTimeout("no answer"),
        ok("finally"),
    )
    response = api.client.generate("q", model=MODEL)

    assert response.text == "finally"
    assert len(api.wire.requests) == 5
    # The 429 asks for 7 s; the others wait half of 8, 16 and 32 s (rng pinned to 0).
    assert api.sleeps == [7.0, 4.0, 8.0, 16.0]
    assert response.usage.requests == 1


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "30000", "retry-after": "1"}, 30.0),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, 2.0),
        ({"retry-after": "0.5"}, 2.0),
        ({}, 2.0),
    ],
)
def test_retry_after_headers_only_lengthen_the_wait(api, headers, expected):
    api.wire.queue(api_error(429, "slow down", headers=headers), ok("fine"))
    api.client.generate("q", model=MODEL)
    assert api.sleeps == [expected]


@pytest.mark.parametrize(
    ("status", "code", "error_type"),
    [
        (400, "context_length_exceeded", LLMError),
        (404, "model_not_found", LLMError),
        (429, "insufficient_quota", LLMError),
        (401, "invalid_api_key", LLMConfigError),
        (403, None, LLMConfigError),
    ],
)
def test_other_client_errors_fail_fast(api, status, code, error_type):
    api.wire.queue(api_error(status, "the provider says no", code=code))
    events: list[dict[str, Any]] = []
    with pytest.raises(LLMError) as err:
        api.client.generate("q", model=MODEL, on_event=events.append)

    assert type(err.value) is error_type
    assert str(status) in str(err.value)
    assert "the provider says no" in str(err.value)
    assert "req_err" in str(err.value), "the request id belongs in the message"
    assert len(api.wire.requests) == 1
    assert api.sleeps == []
    assert [event["type"] for event in events] == ["request", "error"]


def test_gives_up_after_the_attempt_budget_and_keeps_the_conversation_usable(monkeypatch, wire):
    api = build(monkeypatch, wire, retry=RetryPolicy(max_attempts=3))
    wire.queue(api_error(503), api_error(503), api_error(503), ok("back"))
    events: list[dict[str, Any]] = []
    conversation = api.client.conversation(model=MODEL, on_event=events.append)

    with pytest.raises(LLMError, match="after 3 attempts") as err:
        conversation.send("first")
    assert "503" in str(err.value)
    assert api.sleeps == [2.0, 4.0]
    assert events[-1] == {"type": "error", "error": f"LLMError: {err.value}"}

    # The failed turn was never added, so the next request starts clean.
    assert conversation.send("second").text == "back"
    assert wire.requests[-1]["messages"] == [user_text("second")]


def test_an_empty_answer_is_asked_again_and_its_tokens_are_counted(api):
    api.wire.queue(ok(None), ok("real answer"))
    conversation = api.client.conversation(model=MODEL)

    response = conversation.send("q")

    assert response.text == "real answer"
    assert api.sleeps == [2.0]
    assert response.usage == Usage(input_tokens=20, output_tokens=10, requests=2)
    assert conversation.usage == response.usage


def test_refusals_raise_without_retrying(api):
    api.wire.queue(ok(None, refusal="I can't help with that."), ok("...", finish="content_filter"))
    conversation = api.client.conversation(model=MODEL)

    with pytest.raises(LLMRefusal, match="I can't help with that"):
        conversation.send("q")
    with pytest.raises(LLMRefusal) as filtered:
        conversation.send("q")
    assert filtered.value.category == "content_filter"

    assert len(api.wire.requests) == 2
    assert api.sleeps == []
    assert roles(api.wire.requests[1]) == ["user"], "a refused turn must not enter the history"
    assert conversation.usage.requests == 2


def test_truncation_raises_with_the_partial_text(api):
    api.wire.queue(ok("The first half", finish="length"), ok("complete"))
    events: list[dict[str, Any]] = []
    conversation = api.client.conversation(model=MODEL, max_tokens=4, on_event=events.append)

    with pytest.raises(LLMTruncated) as err:
        conversation.send("write a lot")
    assert err.value.partial_text == "The first half"
    assert [event["type"] for event in events] == ["request", "usage", "error"]

    conversation.send("write a lot", max_tokens=4096)
    assert roles(api.wire.requests[1]) == ["user"]
    assert api.wire.requests[1]["max_completion_tokens"] == 4096
