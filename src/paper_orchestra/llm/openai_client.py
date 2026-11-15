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

"""OpenAI (Chat Completions) behind :class:`~paper_orchestra.llm.base.LLMClient`.

An optional provider, ported from ``utils/openai_utils.py`` and the OpenAI
branches of ``utils/llm_backend_utils.py``.  The SDK is imported when the
client is constructed, so a missing ``openai`` install or a missing key never
affects another provider.

The client has no client-side tools, no memory tool and no web search, so the
context harness drives it in push mode.  PDFs are sent as their extracted
text.  ``effort`` and ``max_turns`` have no equivalent here and are ignored
because dropping them cannot make a result wrong.  ``temperature`` is left out
for o-series reasoning models and for any other model that turns out not to
support it; a ``temperature`` or token limit the API rejects by name is
dropped with a warning and the request is sent again.

A ``send`` that raises anything other than :class:`LLMParseError` leaves the
conversation exactly as it was, so the caller may send the same turn again.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from paper_orchestra.utils.pdf import PdfError, load_paper_bytes

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

__all__ = ["OpenAIClient", "RetryPolicy"]

log = logging.getLogger(__name__)

T = TypeVar("T")

# A single non-streaming request can run for many minutes on a long document.
_REQUEST_TIMEOUT_S = 30 * 60.0
_CONNECT_TIMEOUT_S = 10.0

_TRANSIENT_STATUS = frozenset({408, 429})

# o-series reasoning models take instructions in the developer role and reject
# sampling parameters.
_O_SERIES = re.compile(r"o\d")

# Optional request parameters a model may reject with a 400 that names them
# (API parameter name -> our keyword; the token limit also goes by its legacy name).
_REJECTABLE = {
    "temperature": "temperature",
    "max_completion_tokens": "max_completion_tokens",
    "max_tokens": "max_completion_tokens",
}

# Error codes that say the model does not take the parameter at all, as
# opposed to this one value being out of range.
_UNSUPPORTED_CODES = frozenset({"unsupported_parameter", "unsupported_value"})

_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}


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
    stop_reason: str | None
    model: str
    request_id: str | None
    raw: ChatCompletion


def _import_sdk() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise LLMConfigError(
            "OpenAI models need the openai package, which is not installed. "
            "Install it with `uv sync --extra openai`."
        ) from exc
    return openai


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


def _is_o_series(model: str) -> bool:
    return _O_SERIES.match(model) is not None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds the server asked us to wait (``retry-after-ms`` or ``retry-after``)."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return max(0.0, float(value) * scale)
        except ValueError:
            # ``retry-after`` may also be an HTTP date; fall back to our own backoff.
            continue
    return None


def _usage_of(completion: ChatCompletion) -> Usage:
    """Map OpenAI token counts onto :class:`Usage` (cached tokens are not input tokens).

    OpenAI reports cache reads and cache writes inside ``prompt_tokens``.
    """
    usage = Usage(requests=1)
    reported = completion.usage
    if reported is None:
        return usage
    # The detail fields are newer than the oldest SDK release this package accepts.
    details = getattr(reported, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or 0
    written = getattr(details, "cache_write_tokens", None) or 0
    # The SDK does not validate responses, so a count the server left out arrives as None.
    usage.input_tokens = max(0, (reported.prompt_tokens or 0) - cached - written)
    usage.cache_read_input_tokens = cached
    usage.cache_creation_input_tokens = written
    usage.output_tokens = reported.completion_tokens or 0
    return usage


def _read_reply(completion: ChatCompletion, model: str) -> _Reply:
    """Turn a completion into a reply, raising for refusals, truncation and empty answers."""
    choice = completion.choices[0] if completion.choices else None
    message = choice.message if choice is not None else None
    content = message.content if message is not None else None
    if isinstance(content, list):
        # OpenAI-compatible servers may answer with typed parts instead of a string.
        content = "".join(part.get("text") or "" for part in content if isinstance(part, dict))
    text = content if isinstance(content, str) else ""
    finish = choice.finish_reason if choice is not None else None

    if message is not None and message.refusal:
        raise LLMRefusal(f"OpenAI model {model} declined the request: {message.refusal}")
    if finish == "content_filter":
        raise LLMRefusal(
            f"OpenAI model {model} stopped on its content filter.", category="content_filter"
        )
    if finish == "length":
        raise LLMTruncated(
            f"OpenAI model {model} hit the output token limit after {len(text)} characters.",
            partial_text=text,
        )
    if not text.strip():
        raise _TransientError(f"empty response (finish reason {finish})")
    return _Reply(
        text=text,
        stop_reason=_STOP_REASONS.get(finish, finish) if finish else None,
        model=completion.model or model,
        request_id=getattr(completion, "_request_id", None),
        raw=completion,
    )


def _pdf_as_text(pdf: Pdf) -> str:
    title = pdf.title or "document.pdf"
    try:
        text = load_paper_bytes(pdf.data)
    except PdfError as exc:
        raise LLMError(
            f"OpenAI models receive PDFs as extracted text, and none could be extracted from "
            f"{title}: {exc}"
        ) from exc
    return f'<document title="{title}">\n{text}\n</document>'


def _user_content(parts: Sequence[Text | Image | Pdf]) -> list[dict[str, Any]]:
    """Chat Completions content parts; images go as data URLs with their own media type."""
    out: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, Text):
            if part.text:
                out.append({"type": "text", "text": part.text})
        elif isinstance(part, Image):
            url = f"data:{part.media_type};base64,{part.to_base64()}"
            out.append({"type": "image_url", "image_url": {"url": url}})
        else:
            out.append({"type": "text", "text": _pdf_as_text(part)})
    if not out:
        raise ValueError("cannot send an empty message to OpenAI")
    return out


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


class _OpenAIConversation(Conversation):
    def __init__(
        self,
        client: OpenAIClient,
        *,
        model: str,
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        label: str,
        on_event: EventSink | None,
    ) -> None:
        self._client = client
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._label = label
        self._on_event = on_event
        self._messages: list[dict[str, Any]] = []
        if system:
            role = "developer" if _is_o_series(model) else "system"
            self._messages.append({"role": role, "content": system})
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
                "The OpenAI client does not support native structured output. Put the JSON "
                "schema in the prompt and pass parser=parse_json (the context harness does "
                "this when capabilities.structured_output is False)."
            )
        typed = normalize_parts(parts)
        user = {"role": "user", "content": _user_content(typed)}
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
            reply = self._client._complete(
                model=self.model,
                messages=[*self._messages, user],
                temperature=self._temperature,
                max_tokens=self._max_tokens if max_tokens is None else max_tokens,
                usage=turn,
                label=self._label,
            )
        except LLMError as exc:
            self._usage.add(turn)
            if turn.requests:
                self._emit({"type": "usage", **turn.to_dict()})
            self._emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            raise

        self._messages += [user, {"role": "assistant", "content": reply.text}]
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
                f"The response of OpenAI model {self.model} could not be parsed.",
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


class OpenAIClient(LLMClient):
    """OpenAI chat models, authenticated with ``OPENAI_API_KEY``."""

    provider = "openai"
    capabilities = Capabilities(pdf_input=False)

    def __init__(
        self,
        *,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        openai = _import_sdk()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigError("OpenAI models need credentials: set OPENAI_API_KEY.")
        _require_sendable("OPENAI_API_KEY", api_key)
        self._openai = openai
        # Retries are done here (max_retries=0) so that one policy, with an
        # injectable clock, covers every failure mode.
        self._sdk = openai.OpenAI(
            api_key=api_key,
            max_retries=0,
            timeout=openai.Timeout(_REQUEST_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S),
        )
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self._lock = threading.Lock()
        self._no_temperature: set[str] = set()

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "credential_source": "api_key",
            "sdk_version": self._openai.__version__,
        }

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
                "The OpenAI client does not run client-side tools; use the context harness "
                "in push mode (no tools) with this provider."
            )
        if memory is not None:
            raise LLMConfigError(
                "The OpenAI client has no memory tool; use the context harness in push mode "
                "(memory pages injected into the prompt) with this provider."
            )
        if web_search is not None:
            raise LLMConfigError(
                "The OpenAI client does not support web search; use a provider whose "
                "capabilities.web_search is True."
            )
        return _OpenAIConversation(
            self,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            label=label,
            on_event=on_event,
        )

    def _complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        usage: Usage,
        label: str,
    ) -> _Reply:
        """Run one turn, retrying transient failures; ``usage`` collects every response."""
        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["max_completion_tokens"] = max_tokens
        if temperature is not None and self._accepts_temperature(model):
            options["temperature"] = temperature

        def attempt() -> _Reply:
            completion = self._request(model, messages, options)
            usage.add(_usage_of(completion))
            return _read_reply(completion, model)

        what = f"OpenAI request ({model}, {label})" if label else f"OpenAI request ({model})"
        return self._with_retry(attempt, what=what)

    def _accepts_temperature(self, model: str) -> bool:
        if _is_o_series(model):
            return False
        with self._lock:
            return model not in self._no_temperature

    def _request(
        self, model: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> ChatCompletion:
        """One ``chat.completions.create`` call with SDK failures translated.

        An optional parameter the model rejects is removed from ``options`` and
        the request is sent again; this ends because each one can go only once.
        """
        openai = self._openai
        while True:
            try:
                completion = self._sdk.chat.completions.create(
                    model=model, messages=messages, **options
                )
                # The SDK hands back a 200 body it cannot read as a completion unchanged
                # (the HTML error page of a gateway arrives as a plain string).
                if not isinstance(completion, openai.types.chat.ChatCompletion):
                    raise _TransientError("the API returned a body that is not a chat completion")
                return completion
            except json.JSONDecodeError as exc:
                raise _TransientError("the API returned a body that is not JSON") from exc
            except openai.BadRequestError as exc:
                rejected = _REJECTABLE.get(exc.param or "")
                if rejected is None or rejected not in options:
                    raise self._translate(exc) from exc
                value = options.pop(rejected)
                log.warning(
                    "OpenAI model %s rejected %s=%r; sending the request without it.",
                    model,
                    rejected,
                    value,
                )
                if rejected == "temperature" and exc.code in _UNSUPPORTED_CODES:
                    with self._lock:
                        self._no_temperature.add(model)
            except openai.APIConnectionError as exc:
                cause = exc.__cause__
                reason = f"{type(cause).__name__}: {cause}" if cause is not None else str(exc)
                raise _TransientError(f"connection error ({reason})") from exc
            except openai.APIStatusError as exc:
                raise self._translate(exc) from exc
            except openai.OpenAIError as exc:
                raise LLMError(f"OpenAI request failed: {exc}") from exc

    @staticmethod
    def _translate(exc: Any) -> Exception:
        status = exc.status_code
        message = exc.body.get("message") if isinstance(exc.body, dict) else None
        detail = f"OpenAI API error {status}" + (f" ({exc.code})" if exc.code else "")
        detail = f"{detail}: {message or exc.message}"
        if exc.request_id:
            detail = f"{detail} (request id {exc.request_id})"
        # A 429 for exhausted quota is a billing state, not a rate limit: waiting does not help.
        if exc.code != "insufficient_quota" and (status in _TRANSIENT_STATUS or status >= 500):
            return _TransientError(detail, retry_after=_retry_after(exc.response.headers))
        if status in (401, 403):
            return LLMConfigError(f"{detail} (check OPENAI_API_KEY and the model's access)")
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
