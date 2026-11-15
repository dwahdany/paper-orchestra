"""Offline tests for the Gemini client.

The real google-genai SDK runs end to end: only its HTTP transport is replaced
by a scripted one, so request bodies and response parsing are the SDK's own.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

genai = pytest.importorskip("google.genai")

from google.auth import exceptions as auth_exceptions  # noqa: E402
from google.genai import _api_client  # noqa: E402
from google.oauth2.credentials import Credentials  # noqa: E402

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
from paper_orchestra.llm.gemini_client import GeminiClient, RetryPolicy  # noqa: E402
from paper_orchestra.llm.parsing import parse_json  # noqa: E402

# The SDK is built on httpx or on its httpx2 fork, depending on the release.
HTTPX = getattr(_api_client, "httpx2", None) or _api_client.httpx

MODEL = "gemini-3-flash-preview"
PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"


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
    text: str = "hello",
    *,
    parts: list[dict[str, Any]] | None = None,
    finish: str = "STOP",
    usage: dict[str, int] | None = None,
    grounding: dict[str, Any] | None = None,
) -> Any:
    candidate: dict[str, Any] = {
        "content": {"role": "model", "parts": [{"text": text}] if parts is None else parts},
        "finishReason": finish,
    }
    if grounding is not None:
        candidate["groundingMetadata"] = grounding
    return HTTPX.Response(
        200,
        json={
            "candidates": [candidate],
            "usageMetadata": usage or {"promptTokenCount": 10, "candidatesTokenCount": 5},
            "modelVersion": "gemini-test-001",
            "responseId": "resp-1",
        },
    )


def image_ok(data: bytes = PNG) -> Any:
    blob = {"mimeType": "image/png", "data": base64.b64encode(data).decode()}
    return ok(parts=[{"inlineData": blob}])


def blocked_prompt(reason: str = "SAFETY") -> Any:
    return HTTPX.Response(
        200,
        json={"promptFeedback": {"blockReason": reason}, "usageMetadata": {"promptTokenCount": 4}},
    )


def api_error(
    code: int, message: str = "boom", status: str = "UNAVAILABLE", details: Any = None
) -> Any:
    error: dict[str, Any] = {"code": code, "message": message, "status": status}
    if details is not None:
        error["details"] = details
    return HTTPX.Response(code, json={"error": error})


def inline(part: dict[str, Any]) -> tuple[str, bytes]:
    blob = part["inlineData"]
    # The SDK writes bytes as URL-safe base64, which the API accepts like the standard form.
    return blob.get("mimeType") or blob["mime_type"], base64.urlsafe_b64decode(blob["data"])


def roles(body: dict[str, Any]) -> list[str]:
    return [content["role"] for content in body["contents"]]


@pytest.fixture
def wire(monkeypatch):
    """Route every ``genai.Client`` the code under test builds through a scripted transport."""
    wire = Wire()
    real_client = genai.Client

    def offline_client(**kwargs: Any) -> Any:
        wire.client_kwargs = dict(kwargs)
        transport = HTTPX.MockTransport(wire)
        options = kwargs.pop("http_options").model_copy(
            update={"httpx_client": HTTPX.Client(transport=transport)}
        )
        if kwargs.get("vertexai"):
            # A static token keeps google-auth from looking for real credentials.
            kwargs["credentials"] = Credentials(token="offline-token")
        return real_client(http_options=options, **kwargs)

    monkeypatch.setattr(genai, "Client", offline_client)
    return wire


def build(monkeypatch, wire: Wire, *, vertex: bool = False, **options: Any) -> SimpleNamespace:
    if vertex:
        monkeypatch.setenv("VERTEX_AI_PROJECT", "proj")
        monkeypatch.setenv("VERTEX_AI_LOCATION", "us-central1")
    else:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sleeps: list[float] = []
    # rng=0 pins the jitter to the lower bound: half of each exponential step.
    client = GeminiClient(sleep=sleeps.append, rng=lambda: 0.0, **options)
    return SimpleNamespace(client=client, wire=wire, sleeps=sleeps)


@pytest.fixture
def api(monkeypatch, wire):
    return build(monkeypatch, wire)


# -- construction ---------------------------------------------------------------------


def test_missing_credentials_error_names_the_environment_variables(monkeypatch, wire):
    with pytest.raises(LLMConfigError) as err:
        GeminiClient()
    for name in ("VERTEX_AI_PROJECT", "VERTEX_AI_LOCATION", "GEMINI_API_KEY"):
        assert name in str(err.value)

    monkeypatch.setenv("VERTEX_AI_PROJECT", "proj")  # half a Vertex AI configuration
    with pytest.raises(LLMConfigError):
        GeminiClient()
    assert wire.client_kwargs == {}, "no SDK client may be built without credentials"


def test_missing_sdk_error_gives_the_install_command(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(LLMConfigError, match="uv sync --extra gemini") as err:
        GeminiClient()
    assert "google-genai" in str(err.value)


def test_importing_the_module_does_not_import_the_sdk():
    code = (
        "import sys, paper_orchestra.llm.gemini_client\n"
        "assert 'google.genai' not in sys.modules, 'google.genai was imported eagerly'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_api_key_selects_the_gemini_api(api):
    kwargs = api.wire.client_kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["vertexai"] is False
    assert "project" not in kwargs
    assert kwargs["http_options"].timeout == 30 * 60 * 1000

    description = api.client.describe()
    assert description == {
        "provider": "gemini",
        "credential_source": "api_key",
        "sdk_version": genai.__version__,
    }
    assert "test-key" not in repr(description)


def test_vertex_ai_takes_precedence_over_the_api_key(monkeypatch, wire):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    vertex = build(monkeypatch, wire, vertex=True)
    kwargs = wire.client_kwargs
    assert (kwargs["vertexai"], kwargs["project"], kwargs["location"]) == (
        True,
        "proj",
        "us-central1",
    )
    assert "api_key" not in kwargs
    assert vertex.client.describe() == {
        "provider": "gemini",
        "credential_source": "vertex_ai",
        "project": "proj",
        "location": "us-central1",
        "sdk_version": genai.__version__,
    }

    wire.queue(ok("from vertex"))
    assert vertex.client.generate("hi", model=MODEL).text == "from vertex"
    assert "/projects/proj/locations/us-central1/" in wire.urls[0]


def test_half_configured_vertex_ai_falls_back_to_the_api_key_with_a_warning(
    monkeypatch, wire, caplog
):
    monkeypatch.setenv("VERTEX_AI_LOCATION", "us-central1")
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.llm.gemini_client"):
        client = build(monkeypatch, wire).client
    assert client.describe()["credential_source"] == "api_key"
    assert "VERTEX_AI_PROJECT" in caplog.text


def test_capabilities_match_the_spec():
    assert GeminiClient.capabilities == Capabilities(
        web_search=True, pdf_input=True, image_generation=True, structured_output=False
    )
    assert not GeminiClient.capabilities.tools
    assert not GeminiClient.capabilities.memory_tool


# -- requests and responses -----------------------------------------------------------


def test_text_turn_places_system_and_options_and_reports_usage_and_events(api):
    api.wire.queue(
        ok(
            parts=[{"text": "pondering", "thought": True}, {"text": "Hello "}, {"text": "there"}],
            usage={
                "promptTokenCount": 100,
                "cachedContentTokenCount": 40,
                "toolUsePromptTokenCount": 5,
                "candidatesTokenCount": 7,
                "thoughtsTokenCount": 3,
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

    body = api.wire.requests[0]
    assert api.wire.urls[0].endswith(f"/models/{MODEL}:generateContent")
    assert body["systemInstruction"]["parts"] == [{"text": "You are terse."}]
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Say hello"}]}]
    assert body["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": 512}
    assert "tools" not in body

    assert response.text == "Hello there"
    assert response.parsed is None
    assert response.stop_reason == "end_turn"
    assert response.model == "gemini-test-001"
    assert response.request_id == "resp-1"
    # Cached tokens are not billed as input; search results and thoughts are billed.
    expected = Usage(input_tokens=65, output_tokens=10, cache_read_input_tokens=40, requests=1)
    assert response.usage == expected
    assert conversation.usage == expected

    assert events == [
        {"type": "request", "label": "outline.writer", "model": MODEL, "user": "Say hello"},
        {
            "type": "assistant",
            "text": "Hello there",
            "stop_reason": "end_turn",
            "model": "gemini-test-001",
            "request_id": "resp-1",
        },
        {"type": "usage", **expected.to_dict()},
    ]
    json.dumps(events)


def test_images_and_pdfs_travel_inline_with_their_own_media_type(api):
    webp = Image(data=b"RIFF\x00\x00\x00\x00WEBPVP8 ", media_type="image/webp")
    jpeg = Image.from_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    pdf = Pdf(data=b"%PDF-1.7 fake", title="paper.pdf")
    api.wire.queue(ok("seen"))
    events: list[dict[str, Any]] = []

    api.client.generate([pdf, "Describe these.", webp, jpeg], model=MODEL, on_event=events.append)

    parts = api.wire.requests[0]["contents"][0]["parts"]
    assert inline(parts[0]) == ("application/pdf", pdf.data)
    assert parts[1] == {"text": "Describe these."}
    assert inline(parts[2]) == ("image/webp", webp.data)
    assert inline(parts[3]) == ("image/jpeg", jpeg.data)
    assert events[0]["user"] == (
        "[pdf paper.pdf, 13 bytes]\nDescribe these.\n"
        f"[image image/webp, {len(webp.data)} bytes]\n[image image/jpeg, {len(jpeg.data)} bytes]"
    )


def test_multi_turn_resends_the_history_unchanged(api):
    signature = base64.b64encode(b"sig-1").decode()
    api.wire.queue(ok(parts=[{"text": "4", "thoughtSignature": signature}]), ok("5"))
    conversation = api.client.conversation(model=MODEL, system="Count.")

    conversation.send("2+2?")
    assert conversation.send("plus one?").text == "5"

    first, second = api.wire.requests
    assert roles(second) == ["user", "model", "user"]
    assert second["contents"][0] == first["contents"][0]
    # The model turn goes back as received, thought signature included.
    assert second["contents"][1]["parts"] == [{"text": "4", "thoughtSignature": signature}]
    assert second["contents"][2]["parts"] == [{"text": "plus one?"}]
    assert second["systemInstruction"] == first["systemInstruction"]
    assert conversation.usage == Usage(input_tokens=20, output_tokens=10, requests=2)


def test_send_can_override_max_tokens_for_one_turn(api):
    api.wire.queue(ok("a"), ok("b"))
    conversation = api.client.conversation(model=MODEL, max_tokens=100)
    conversation.send("one", max_tokens=7)
    conversation.send("two")
    limits = [body["generationConfig"]["maxOutputTokens"] for body in api.wire.requests]
    assert limits == [7, 100]


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
    assert roles(api.wire.requests[2]) == ["user", "model", "user", "model", "user"]
    assert api.wire.requests[2]["contents"][3]["parts"] == [{"text": "not json at all"}]


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
    with pytest.raises(LLMConfigError, match="structured output"):
        api.client.conversation(model=MODEL).send("q", schema={"type": "object"})
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
    retry_info = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]
    api.wire.queue(
        api_error(429, "quota", "RESOURCE_EXHAUSTED", details=retry_info),
        api_error(503, "overloaded"),
        HTTPX.ConnectError("connection refused"),
        HTTPX.Response(500, text="<html>bad gateway</html>"),
        ok("finally"),
    )
    response = api.client.generate("q", model=MODEL)

    assert response.text == "finally"
    assert len(api.wire.requests) == 5
    # The 429 asks for 17 s; the others wait half of 8, 16 and 32 s (rng pinned to 0).
    assert api.sleeps == [17.0, 4.0, 8.0, 16.0]
    assert response.usage.requests == 1


@pytest.mark.parametrize(
    ("code", "status", "error_type"),
    [
        (400, "INVALID_ARGUMENT", LLMError),
        (404, "NOT_FOUND", LLMError),
        (401, "UNAUTHENTICATED", LLMConfigError),
        (403, "PERMISSION_DENIED", LLMConfigError),
    ],
)
def test_other_client_errors_fail_fast(api, code, status, error_type):
    api.wire.queue(api_error(code, "the provider says no", status))
    events: list[dict[str, Any]] = []
    with pytest.raises(LLMError) as err:
        api.client.generate("q", model=MODEL, on_event=events.append)

    assert type(err.value) is error_type
    assert str(code) in str(err.value)
    assert status in str(err.value)
    assert "the provider says no" in str(err.value)
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
    assert wire.requests[-1]["contents"] == [{"role": "user", "parts": [{"text": "second"}]}]


def test_an_empty_answer_is_asked_again_and_its_tokens_are_counted(api):
    empty = HTTPX.Response(
        200,
        json={
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10},
        },
    )
    api.wire.queue(empty, ok("real answer"))
    conversation = api.client.conversation(model=MODEL)

    response = conversation.send("q")

    assert response.text == "real answer"
    assert api.sleeps == [2.0]
    assert response.usage == Usage(input_tokens=20, output_tokens=5, requests=2)
    assert conversation.usage == response.usage


def test_blocked_prompt_and_safety_stop_raise_refusals_without_retrying(api):
    api.wire.queue(blocked_prompt("PROHIBITED_CONTENT"), ok("partial", finish="SAFETY"))
    conversation = api.client.conversation(model=MODEL)

    with pytest.raises(LLMRefusal) as blocked:
        conversation.send("q")
    assert blocked.value.category == "prohibited_content"

    with pytest.raises(LLMRefusal) as stopped:
        conversation.send("q")
    assert stopped.value.category == "safety"

    assert len(api.wire.requests) == 2
    assert api.sleeps == []
    assert roles(api.wire.requests[1]) == ["user"], "a refused turn must not enter the history"
    assert conversation.usage.requests == 2
    assert conversation.usage.input_tokens == 14


def test_truncation_raises_with_the_partial_text(api):
    api.wire.queue(ok("The first half", finish="MAX_TOKENS"), ok("complete"))
    events: list[dict[str, Any]] = []
    conversation = api.client.conversation(model=MODEL, max_tokens=4, on_event=events.append)

    with pytest.raises(LLMTruncated) as err:
        conversation.send("write a lot")
    assert err.value.partial_text == "The first half"
    assert [event["type"] for event in events] == ["request", "usage", "error"]

    conversation.send("write a lot", max_tokens=4096)
    assert roles(api.wire.requests[1]) == ["user"]
    assert api.wire.requests[1]["generationConfig"]["maxOutputTokens"] == 4096


def test_credential_failures_are_config_errors_unless_retryable(monkeypatch, wire):
    vertex = build(monkeypatch, wire, vertex=True)

    wire.queue(auth_exceptions.DefaultCredentialsError("no default credentials"))
    with pytest.raises(LLMConfigError, match="application-default login"):
        vertex.client.generate("q", model=MODEL)
    assert vertex.sleeps == []

    wire.queue(auth_exceptions.RefreshError("token endpoint down", retryable=True), ok("fine"))
    assert vertex.client.generate("q", model=MODEL).text == "fine"
    assert vertex.sleeps == [2.0]


# -- web search -----------------------------------------------------------------------


def test_web_search_enables_grounding_and_counts_the_searches(api, caplog):
    grounding = {
        "webSearchQueries": ["attention is all you need", "transformer 2017"],
        "groundingChunks": [{"web": {"uri": "https://arxiv.org/abs/1706.03762", "title": "x"}}],
    }
    chunks_only = {"groundingChunks": [{"web": {"uri": "https://example.org", "title": "y"}}]}
    api.wire.queue(ok("grounded", grounding=grounding), ok("also grounded", grounding=chunks_only))
    search = WebSearch(max_uses=3, allowed_domains=("arxiv.org",), blocked_domains=("spam.test",))

    with caplog.at_level(logging.WARNING, logger="paper_orchestra.llm.gemini_client"):
        first = api.client.generate("find", model=MODEL, web_search=search, temperature=0.1)
        second = api.client.generate("find", model=MODEL, web_search=search)

    # The Gemini API has no domain filters, which is said once rather than hidden.
    assert api.wire.requests[0]["tools"] == [{"googleSearch": {}}]
    assert caplog.text.count("allowed_domains") == 1
    assert caplog.text.count("blocked_domains") == 1
    assert first.usage.web_searches == 2
    assert second.usage.web_searches == 1


def test_no_grounding_reported_means_no_searches_counted(api):
    api.wire.queue(ok("from memory"))
    response = api.client.generate("find", model=MODEL, web_search=WebSearch())
    assert response.usage.web_searches == 0


def test_vertex_ai_search_excludes_blocked_domains(monkeypatch, wire):
    vertex = build(monkeypatch, wire, vertex=True)
    wire.queue(ok("grounded"))
    vertex.client.generate(
        "find", model=MODEL, web_search=WebSearch(blocked_domains=("spam.test",))
    )
    search = wire.requests[0]["tools"][0]["googleSearch"]
    assert (search.get("excludeDomains") or search["exclude_domains"]) == ["spam.test"]


# -- image generation -----------------------------------------------------------------


def test_generate_image_returns_the_raw_image_bytes(api):
    api.wire.queue(image_ok())
    image = api.client.generate_image(
        "A flow diagram",
        model="gemini-3-pro-image-preview",
        aspect_ratio="4:3",
        system="Draw clean diagrams.",
    )

    assert image == PNG
    body = api.wire.requests[0]
    assert api.wire.urls[0].endswith("/models/gemini-3-pro-image-preview:generateContent")
    assert body["contents"] == [{"role": "user", "parts": [{"text": "A flow diagram"}]}]
    assert body["systemInstruction"]["parts"] == [{"text": "Draw clean diagrams."}]
    assert body["generationConfig"] == {
        "temperature": 0.7,
        "candidateCount": 1,
        "responseModalities": ["IMAGE"],
        "imageConfig": {"aspectRatio": "4:3"},
    }


def test_generate_image_asks_again_when_the_model_answers_with_text(api):
    api.wire.queue(ok("I would rather describe it."), api_error(503), image_ok())
    assert api.client.generate_image("A diagram", model="gemini-3-pro-image-preview") == PNG
    assert len(api.wire.requests) == 3
    assert api.sleeps == [2.0], "only the 503 waits; a missing image is simply resampled"


def test_generate_image_returns_none_when_blocked_or_never_produced(monkeypatch, wire, caplog):
    api = build(monkeypatch, wire, retry=RetryPolicy(max_attempts=2))
    model = "gemini-3-pro-image-preview"

    with caplog.at_level(logging.WARNING, logger="paper_orchestra.llm.gemini_client"):
        wire.queue(blocked_prompt("IMAGE_SAFETY"))
        assert api.client.generate_image("Something disallowed", model=model) is None
        assert len(wire.requests) == 1
        assert "IMAGE_SAFETY" in caplog.text

        wire.queue(ok("no image"), ok("still no image"))
        assert api.client.generate_image("A diagram", model=model) is None
        assert len(wire.requests) == 3
        assert "attempt 2 of 2" in caplog.text


def test_generate_image_raises_on_api_errors(api):
    api.wire.queue(api_error(400, "unknown model", "INVALID_ARGUMENT"))
    with pytest.raises(LLMError, match="unknown model"):
        api.client.generate_image("A diagram", model="gemini-typo")
    assert api.sleeps == []
