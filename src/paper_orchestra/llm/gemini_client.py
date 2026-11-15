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

"""Gemini (Google Gen AI SDK) behind :class:`~paper_orchestra.llm.base.LLMClient`.

An optional provider, ported from ``utils/gemini_utils.py``.  The SDK is
imported when the client is constructed, so a missing ``google-genai`` install
or missing Gemini credentials never affect another provider.

The client has no client-side tools and no memory tool, so the context harness
drives it in push mode.  Three options have no Gemini equivalent and are
ignored because dropping them cannot make a result wrong: ``effort``,
``max_turns`` and ``WebSearch.max_uses`` (Gemini decides how often to search).

A ``send`` that raises anything other than :class:`LLMParseError` leaves the
conversation exactly as it was, so the caller may send the same turn again.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar

from paper_orchestra.llm.base import (
    Capabilities,
    Conversation,
    Effort,
    EventSink,
    Image,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMParseError,
    LLMRefusal,
    LLMResponse,
    LLMTruncated,
    MemoryBackend,
    Parser,
    Part,
    Pdf,
    Schema,
    Text,
    ToolSpec,
    Usage,
    WebSearch,
    normalize_parts,
)

if TYPE_CHECKING:
    from google.genai import types as genai_types

__all__ = ["GeminiClient", "RetryPolicy"]

log = logging.getLogger(__name__)

T = TypeVar("T")

# A single non-streaming request can run for many minutes on a long document.
_REQUEST_TIMEOUT_MS = 30 * 60 * 1000

_TRANSIENT_STATUS = frozenset({408, 429})

# Finish reasons that mean Gemini withheld or cut the answer on policy grounds.
_REFUSAL_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "MODEL_ARMOR",
    }
)

_STOP_REASONS = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter for transient provider failures."""

    max_attempts: int = 5
    base_delay: float = 4.0
    max_delay: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def delay(self, retry: int, jitter: float, hint: float | None = None) -> float:
        """Seconds to wait before retry number ``retry`` (1 is the first retry).

        ``jitter`` is a sample from [0, 1); the wait is spread over the upper
        half of the exponential step so parallel workers do not retry in
        lockstep.  A server-provided ``hint`` can lengthen the wait, but never
        beyond ``max_delay``.
        """
        step = min(self.max_delay, self.base_delay * 2 ** (retry - 1))
        delay = step * (0.5 + 0.5 * jitter)
        if hint is not None:
            delay = max(delay, min(hint, self.max_delay))
        return delay


class _TransientError(Exception):
    """A failure worth retrying: rate limit, server error, dropped connection."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class _Reply:
    text: str
    content: genai_types.Content
    stop_reason: str | None
    model: str
    request_id: str | None
    raw: genai_types.GenerateContentResponse


def _import_sdk() -> SimpleNamespace:
    try:
        import google.auth.exceptions as auth_exceptions
        import google.genai as genai
        from google.genai import errors, types
    except ImportError as exc:
        raise LLMConfigError(
            "Gemini models need the google-genai package, which is not installed. "
            "Install it with `uv sync --extra gemini`."
        ) from exc
    return SimpleNamespace(genai=genai, types=types, errors=errors, auth=auth_exceptions)


def _require_sendable(name: str, secret: str) -> None:
    """Refuse a credential that cannot travel in an HTTP header.

    The HTTP library rejects such a value with an error that quotes it in
    full, which would carry the secret into logs and transcripts.
    """
    if not (secret.isascii() and secret.isprintable()):
        raise LLMConfigError(
            f"{name} contains a line break, a control character or a non-ASCII character, so "
            "it cannot be sent as a credential. Copy the key again from the provider's console."
        )


def _transport_errors() -> tuple[tuple[type[Exception], ...], tuple[type[Exception], ...]]:
    """Exceptions of the HTTP library under the SDK: ``(worth retrying, all of them)``.

    google-genai lets them through unwrapped and, depending on its version, is
    built on ``httpx`` or on the ``httpx2`` fork, whose classes are unrelated.
    A fault of the local setup (an unsupported URL scheme, a header that
    cannot be sent) is not worth retrying: asking again cannot fix it.
    """
    transient: list[type[Exception]] = []
    every: list[type[Exception]] = []
    for name in ("httpx2", "httpx"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        transient += [
            module.TimeoutException,
            module.NetworkError,
            module.ProxyError,
            module.RemoteProtocolError,
            module.DecodingError,
        ]
        every.append(module.HTTPError)
    return tuple(transient), tuple(every)


def _retry_delay_hint(details: Any) -> float | None:
    """Seconds asked for by a ``google.rpc.RetryInfo`` entry of an error body."""
    error = details.get("error", details) if isinstance(details, dict) else None
    entries = error.get("details") if isinstance(error, dict) else None
    for entry in entries if isinstance(entries, list) else ():
        delay = entry.get("retryDelay") if isinstance(entry, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def _enum_name(value: Any) -> str:
    """Name of an SDK enum member; values the SDK does not know arrive as plain strings."""
    return str(getattr(value, "name", value))


def _finish_reason(response: genai_types.GenerateContentResponse) -> str | None:
    candidates = response.candidates or []
    if not candidates or candidates[0].finish_reason is None:
        return None
    return _enum_name(candidates[0].finish_reason)


def _refusal_category(response: genai_types.GenerateContentResponse) -> str | None:
    """Why Gemini declined to answer, or ``None`` when it did answer."""
    feedback = response.prompt_feedback
    if not response.candidates and feedback is not None and feedback.block_reason is not None:
        return _enum_name(feedback.block_reason)
    finish = _finish_reason(response)
    return finish if finish in _REFUSAL_FINISH_REASONS else None


def _web_searches(response: genai_types.GenerateContentResponse) -> int:
    candidates = response.candidates or []
    metadata = candidates[0].grounding_metadata if candidates else None
    if metadata is None:
        return 0
    if metadata.web_search_queries:
        return len(metadata.web_search_queries)
    return 1 if metadata.grounding_chunks or metadata.grounding_supports else 0


def _usage_of(response: genai_types.GenerateContentResponse) -> Usage:
    """Map Gemini token counts onto :class:`Usage` (cached tokens are not input tokens).

    Gemini reports the cached prefix inside ``prompt_token_count``, bills
    thinking tokens as output and bills search results fed back to the model
    as input.
    """
    meta = response.usage_metadata
    usage = Usage(requests=1, web_searches=_web_searches(response))
    if meta is None:
        return usage
    cached = meta.cached_content_token_count or 0
    usage.input_tokens = max(0, (meta.prompt_token_count or 0) - cached) + (
        meta.tool_use_prompt_token_count or 0
    )
    usage.cache_read_input_tokens = cached
    usage.output_tokens = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
    return usage


def _read_reply(response: genai_types.GenerateContentResponse, model: str) -> _Reply:
    """Turn a response into a reply, raising for refusals, truncation and empty answers."""
    candidates = response.candidates or []
    content = candidates[0].content if candidates else None
    parts = (content.parts if content is not None else None) or []
    text = "".join(part.text for part in parts if isinstance(part.text, str) and not part.thought)
    finish = _finish_reason(response)

    category = _refusal_category(response)
    if category is not None:
        raise LLMRefusal(
            f"Gemini model {model} declined the request ({category}).", category=category.lower()
        )
    if finish == "MAX_TOKENS":
        raise LLMTruncated(
            f"Gemini model {model} hit the output token limit after {len(text)} characters.",
            partial_text=text,
        )
    if content is None or not text.strip():
        # Gemini occasionally returns a candidate without any text; asking again helps.
        raise _TransientError(f"empty response (finish reason {finish})")
    if content.role is None:
        content = content.model_copy(update={"role": "model"})
    return _Reply(
        text=text,
        content=content,
        stop_reason=_STOP_REASONS.get(finish, finish.lower()) if finish else None,
        model=response.model_version or model,
        request_id=response.response_id,
        raw=response,
    )


def _first_image(response: genai_types.GenerateContentResponse) -> bytes | None:
    for candidate in response.candidates or []:
        content = candidate.content
        for part in (content.parts if content is not None else None) or []:
            if part.thought:
                # A thinking image model shows interim drafts as thought parts.
                continue
            blob = part.inline_data
            if blob is not None and blob.data and (blob.mime_type or "image/").startswith("image/"):
                return blob.data
    return None


def _describe_parts(parts: Sequence[Text | Image | Pdf]) -> str:
    """The user turn as transcript text, with placeholders for binary parts."""
    out = []
    for part in parts:
        if isinstance(part, Text):
            out.append(part.text)
        elif isinstance(part, Image):
            out.append(f"[image {part.media_type}, {len(part.data)} bytes]")
        else:
            out.append(f"[pdf {part.title or ''}, {len(part.data)} bytes]")
    return "\n".join(out)


class _GeminiConversation(Conversation):
    def __init__(
        self,
        client: GeminiClient,
        *,
        model: str,
        system: str | None,
        search_tool: genai_types.Tool | None,
        temperature: float | None,
        max_tokens: int | None,
        label: str,
        on_event: EventSink | None,
    ) -> None:
        self._client = client
        self.model = model
        self._system = system
        self._search_tool = search_tool
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._label = label
        self._on_event = on_event
        self._contents: list[genai_types.Content] = []
        self._usage = Usage()

    @property
    def usage(self) -> Usage:
        return self._usage

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(event)

    def send(
        self,
        parts: Part | Sequence[Part],
        *,
        parser: Parser | None = None,
        schema: Schema | None = None,
        require_parsed: bool = True,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if schema is not None:
            raise LLMConfigError(
                "The Gemini client does not support native structured output. Put the JSON "
                "schema in the prompt and pass parser=parse_json (the context harness does "
                "this when capabilities.structured_output is False)."
            )
        client = self._client
        typed = normalize_parts(parts)
        user = client._user_content(typed)
        config = client._request_config(
            system=self._system,
            temperature=self._temperature,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            search_tool=self._search_tool,
        )
        self._emit(
            {
                "type": "request",
                "label": self._label,
                "model": self.model,
                "user": _describe_parts(typed),
            }
        )

        turn = Usage()
        try:
            reply = client._complete(
                model=self.model,
                contents=[*self._contents, user],
                config=config,
                usage=turn,
                label=self._label,
            )
        except LLMError as exc:
            self._usage.add(turn)
            if turn.requests:
                self._emit({"type": "usage", **turn.to_dict()})
            self._emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            raise

        # The model turn goes back verbatim: it carries the thought signatures
        # Gemini expects to see again on the next request.
        self._contents += [user, reply.content]
        self._usage.add(turn)
        self._emit(
            {
                "type": "assistant",
                "text": reply.text,
                "stop_reason": reply.stop_reason,
                "model": reply.model,
                "request_id": reply.request_id,
            }
        )
        self._emit({"type": "usage", **turn.to_dict()})

        parsed = parser(reply.text) if parser is not None else None
        if parser is not None and require_parsed and parsed is None:
            error = LLMParseError(
                f"The response of Gemini model {self.model} could not be parsed.",
                raw_text=reply.text,
            )
            self._emit({"type": "error", "error": f"LLMParseError: {error}"})
            raise error
        return LLMResponse(
            text=reply.text,
            parsed=parsed,
            model=reply.model,
            usage=turn,
            stop_reason=reply.stop_reason,
            request_id=reply.request_id,
            raw=reply.raw,
        )


class GeminiClient(LLMClient):
    """Gemini through the Gemini API (``GEMINI_API_KEY``) or Vertex AI.

    Vertex AI wins when both ``VERTEX_AI_PROJECT`` and ``VERTEX_AI_LOCATION``
    are set; it authenticates with Application Default Credentials on the
    first request.
    """

    provider = "gemini"
    capabilities = Capabilities(
        web_search=True,
        pdf_input=True,
        image_generation=True,
        structured_output=False,
    )

    def __init__(
        self,
        *,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        sdk = _import_sdk()
        project = os.environ.get("VERTEX_AI_PROJECT", "").strip()
        location = os.environ.get("VERTEX_AI_LOCATION", "").strip()
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        http_options = sdk.types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS)

        self._vertex: dict[str, str] | None = None
        if project and location:
            self._vertex = {"project": project, "location": location}
            self._sdk = sdk.genai.Client(
                vertexai=True, project=project, location=location, http_options=http_options
            )
            log.info("Gemini uses Vertex AI (project %s, location %s).", project, location)
        elif api_key:
            if project or location:
                log.warning(
                    "Only one of VERTEX_AI_PROJECT and VERTEX_AI_LOCATION is set, so Vertex AI "
                    "is not used; Gemini falls back to GEMINI_API_KEY."
                )
            _require_sendable("GEMINI_API_KEY", api_key)
            # vertexai=False pins the Gemini API even when GOOGLE_GENAI_USE_VERTEXAI is set.
            self._sdk = sdk.genai.Client(vertexai=False, api_key=api_key, http_options=http_options)
            log.info("Gemini uses the Gemini API with GEMINI_API_KEY.")
        else:
            raise LLMConfigError(
                "Gemini models need credentials: set both VERTEX_AI_PROJECT and "
                "VERTEX_AI_LOCATION (Vertex AI), or set GEMINI_API_KEY (Gemini API)."
            )

        self._sdk_version: str = sdk.genai.__version__
        self._types = sdk.types
        self._errors = sdk.errors
        self._auth_error: type[Exception] = sdk.auth.GoogleAuthError
        self._auth_transport_error: type[Exception] = sdk.auth.TransportError
        self._transient_transport_errors, self._transport_errors = _transport_errors()
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self._lock = threading.Lock()
        self._warned: set[str] = set()

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"provider": self.provider, "credential_source": "api_key"}
        if self._vertex is not None:
            info |= {"credential_source": "vertex_ai", **self._vertex}
        info["sdk_version"] = self._sdk_version
        return info

    def conversation(
        self,
        *,
        model: str,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        memory: MemoryBackend | None = None,
        web_search: WebSearch | None = None,
        effort: Effort | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_turns: int = 60,
        label: str = "",
        on_event: EventSink | None = None,
        single_turn: bool = False,
    ) -> Conversation:
        if tools:
            raise LLMConfigError(
                "The Gemini client does not run client-side tools; use the context harness "
                "in push mode (no tools) with this provider."
            )
        if memory is not None:
            raise LLMConfigError(
                "The Gemini client has no memory tool; use the context harness in push mode "
                "(memory pages injected into the prompt) with this provider."
            )
        return _GeminiConversation(
            self,
            model=model,
            system=system,
            search_tool=self._search_tool(web_search) if web_search is not None else None,
            temperature=temperature,
            max_tokens=max_tokens,
            label=label,
            on_event=on_event,
        )

    def generate_image(
        self,
        prompt: str,
        *,
        model: str,
        aspect_ratio: str = "16:9",
        system: str | None = None,
    ) -> bytes | None:
        """Return the encoded image, or ``None`` when the model produced none.

        A model that answers without an image is asked again (a fresh sample
        often works); a policy block returns ``None`` at once.  API failures
        raise :class:`LLMError` like any other request.
        """
        types = self._types
        # image_size is left to the API default (1K): not every image model accepts the field.
        options: dict[str, Any] = {
            "temperature": 0.7,
            "candidate_count": 1,
            "response_modalities": ["IMAGE"],
            "image_config": types.ImageConfig(aspect_ratio=aspect_ratio),
        }
        if system:
            options["system_instruction"] = system
        config = types.GenerateContentConfig(**options)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        what = f"Gemini image request ({model})"

        attempts = self._retry.max_attempts
        for attempt in range(1, attempts + 1):
            response = self._with_retry(
                lambda: self._request(model=model, contents=contents, config=config), what=what
            )
            image = _first_image(response)
            if image is not None:
                return image
            category = _refusal_category(response)
            if category is not None:
                log.warning("%s was declined (%s); no image was generated.", what, category)
                return None
            log.warning(
                "%s returned no image (finish reason %s), attempt %d of %d.",
                what,
                _finish_reason(response),
                attempt,
                attempts,
            )
        return None

    def _user_content(self, parts: Sequence[Text | Image | Pdf]) -> genai_types.Content:
        """Build the user turn; images and PDFs travel inline with their own media type."""
        types = self._types
        out = []
        for part in parts:
            if isinstance(part, Text):
                # The API rejects empty text parts.
                if part.text:
                    out.append(types.Part.from_text(text=part.text))
            elif isinstance(part, Image):
                out.append(types.Part.from_bytes(data=part.data, mime_type=part.media_type))
            else:
                out.append(types.Part.from_bytes(data=part.data, mime_type="application/pdf"))
        if not out:
            raise ValueError("cannot send an empty message to Gemini")
        return types.Content(role="user", parts=out)

    def _request_config(
        self,
        *,
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        search_tool: genai_types.Tool | None,
    ) -> genai_types.GenerateContentConfig:
        options: dict[str, Any] = {}
        if system:
            options["system_instruction"] = system
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_output_tokens"] = max_tokens
        if search_tool is not None:
            options["tools"] = [search_tool]
        return self._types.GenerateContentConfig(**options)

    def _complete(
        self,
        *,
        model: str,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
        usage: Usage,
        label: str,
    ) -> _Reply:
        """Run one turn, retrying transient failures; ``usage`` collects every response."""

        def attempt() -> _Reply:
            response = self._request(model=model, contents=contents, config=config)
            usage.add(_usage_of(response))
            return _read_reply(response, model)

        what = f"Gemini request ({model}, {label})" if label else f"Gemini request ({model})"
        return self._with_retry(attempt, what=what)

    def _search_tool(self, web_search: WebSearch) -> genai_types.Tool:
        """Google Search grounding; only Vertex AI can exclude domains."""
        options: dict[str, Any] = {}
        if web_search.blocked_domains:
            if self._vertex is not None:
                options["exclude_domains"] = list(web_search.blocked_domains)
            else:
                self._warn_once(
                    "blocked_domains",
                    "Google Search grounding on the Gemini API cannot exclude domains "
                    "(only Vertex AI can); WebSearch.blocked_domains is ignored.",
                )
        if web_search.allowed_domains:
            self._warn_once(
                "allowed_domains",
                "Google Search grounding cannot be restricted to a list of domains; "
                "WebSearch.allowed_domains is ignored.",
            )
        return self._types.Tool(google_search=self._types.GoogleSearch(**options))

    def _warn_once(self, key: str, message: str) -> None:
        with self._lock:
            if key in self._warned:
                return
            self._warned.add(key)
        log.warning(message)

    def _request(
        self,
        *,
        model: str,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
    ) -> genai_types.GenerateContentResponse:
        """One ``generate_content`` call with SDK failures translated."""
        try:
            return self._sdk.models.generate_content(model=model, contents=contents, config=config)
        except self._errors.APIError as exc:
            raise self._translate(exc) from exc
        except self._transient_transport_errors as exc:
            raise _TransientError(f"connection error ({type(exc).__name__}: {exc})") from exc
        except self._transport_errors as exc:
            raise LLMError(
                f"The Gemini request could not be sent ({type(exc).__name__}: {exc})."
            ) from exc
        except (json.JSONDecodeError, self._errors.UnknownApiResponseError) as exc:
            # The SDK decodes a 200 body without a guard; a gateway's HTML error page lands here.
            raise _TransientError("the API returned a body that is not JSON") from exc
        except self._auth_error as exc:
            if self._auth_failure_is_transient(exc):
                raise _TransientError(f"could not refresh Google credentials: {exc}") from exc
            raise LLMConfigError(
                f"Google credentials for Vertex AI are not usable ({exc}). Run `gcloud auth "
                "application-default login` or set GOOGLE_APPLICATION_CREDENTIALS."
            ) from exc

    def _auth_failure_is_transient(self, exc: Exception) -> bool:
        """Whether a google-auth failure is a network problem rather than bad credentials.

        google-auth marks only some token-endpoint answers as retryable; a
        token endpoint or metadata server it could not reach at all surfaces
        as a ``TransportError``, sometimes wrapped in a ``RefreshError``.
        """
        return (
            bool(getattr(exc, "retryable", False))
            or isinstance(exc, self._auth_transport_error)
            or isinstance(exc.__cause__, self._auth_transport_error)
        )

    @staticmethod
    def _translate(exc: Any) -> Exception:
        code = exc.code if isinstance(exc.code, int) else 0
        detail = f"Gemini API error {code} {exc.status or ''}".rstrip()
        detail = f"{detail}: {exc.message or exc.details}"
        if code in _TRANSIENT_STATUS or code >= 500:
            return _TransientError(detail, retry_after=_retry_delay_hint(exc.details))
        if code in (401, 403):
            return LLMConfigError(
                f"{detail} (check GEMINI_API_KEY, or the Vertex AI project, location and "
                "credentials)"
            )
        return LLMError(detail)

    def _with_retry(self, attempt: Callable[[], T], *, what: str) -> T:
        policy = self._retry
        number = 0
        while True:
            number += 1
            try:
                return attempt()
            except _TransientError as exc:
                if number >= policy.max_attempts:
                    raise LLMError(f"{what} failed after {number} attempts: {exc}") from exc
                delay = policy.delay(number, self._rng(), exc.retry_after)
                log.warning(
                    "%s failed: %s. Retrying in %.1f s (attempt %d of %d).",
                    what,
                    exc,
                    delay,
                    number,
                    policy.max_attempts,
                )
                self._sleep(delay)
