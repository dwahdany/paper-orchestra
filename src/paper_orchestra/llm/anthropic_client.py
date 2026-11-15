"""Claude through the Anthropic SDK: the pipeline's primary provider.

Three things shape this module:

* **Credentials are the SDK's business.**  The SDK client is built without an
  explicit key, so its own resolution applies: ``ANTHROPIC_API_KEY``, then
  ``ANTHROPIC_AUTH_TOKEN``, then ``ANTHROPIC_PROFILE``, then the Workload
  Identity Federation variables, then a profile on disk.  This module only
  bridges Modal's OIDC token into that chain and reports which source won.
* **History is append-only.**  Claude Fable 5.2 checks every replayed thinking
  block against the system prompt, tool definitions and messages that preceded
  it.  A conversation therefore fixes ``system`` and ``tools`` before its first
  answered request and keeps a journal of wire-JSON messages that is only
  appended to.
* **Nothing provider-specific leaks.**  Callers see the types of
  :mod:`paper_orchestra.llm.base`; the SDK is imported lazily, inside functions.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ValidationError

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
    ToolError,
    ToolSpec,
    Usage,
    WebSearch,
    normalize_parts,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 64_000
#: Largest output any current Claude model produces; larger requests are capped.
MAX_OUTPUT_TOKENS = 128_000
DEFAULT_EFFORT: Effort = "high"

#: Opts every request into server-side fallbacks: a request the model declines
#: for policy reasons is re-run on the model Anthropic recommends for that case.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

MEMORY_TOOL_TYPE = "memory_20250818"
MEMORY_TOOL_NAME = "memory"
#: Dynamic-filtering web search, the current version for Claude Fable 5.2.
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
WEB_SEARCH_TOOL_NAME = "web_search"

#: Set to ``0`` in organizations with customer-managed encryption keys, where
#: ``output_config.format`` and strict tools are not available.
STRUCTURED_OUTPUTS_ENV = "PAPER_ORCHESTRA_STRUCTURED_OUTPUTS"

#: Fable-class models read at most this many pixels on the long edge and the
#: API downscales anything larger, so sending more only costs bytes.  Staying
#: below it also satisfies the 8000 px hard limit and the 3000 px limit that
#: applies once a request carries more than 20 images.
MAX_IMAGE_EDGE = 2576
#: Per-image size limit, measured on the base64 encoding.  5 MB is the
#: strictest documented value (the first-party API allows 10 MB) and keeps a
#: request with many figures clear of the request-size limit.
MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024

#: ``pause_turn`` continuations allowed within one ``send``.
MAX_PAUSES = 8
#: Times an interrupted response stream is re-issued.  Connection setup is
#: retried by the SDK itself; a stream that breaks mid-response is not.
STREAM_RESTARTS = 2
STREAM_RESTART_DELAY_S = 2.0
SDK_MAX_RETRIES = 4
#: A single request at high effort routinely runs for many minutes.
REQUEST_TIMEOUT_S = 1800.0
CONNECT_TIMEOUT_S = 10.0

_ENV_API_KEY = "ANTHROPIC_API_KEY"
_ENV_AUTH_TOKEN = "ANTHROPIC_AUTH_TOKEN"
_ENV_PROFILE = "ANTHROPIC_PROFILE"
_ENV_RULE_ID = "ANTHROPIC_FEDERATION_RULE_ID"
_ENV_ORGANIZATION_ID = "ANTHROPIC_ORGANIZATION_ID"
_ENV_SERVICE_ACCOUNT_ID = "ANTHROPIC_SERVICE_ACCOUNT_ID"
_ENV_WORKSPACE_ID = "ANTHROPIC_WORKSPACE_ID"
_ENV_IDENTITY_TOKEN = "ANTHROPIC_IDENTITY_TOKEN"
_ENV_IDENTITY_TOKEN_FILE = "ANTHROPIC_IDENTITY_TOKEN_FILE"
_ENV_MODAL_IDENTITY_TOKEN = "MODAL_IDENTITY_TOKEN"
_ENV_MODAL_IDENTITY_TOKEN_FILE = "MODAL_IDENTITY_TOKEN_FILE"

_SETUP_HELP = (
    "Configure one of:\n"
    f"  1. API key: set {_ENV_API_KEY} (for example in a .env file).\n"
    f"  2. Workload Identity Federation (keyless): set {_ENV_RULE_ID} and "
    f"{_ENV_ORGANIZATION_ID}, plus {_ENV_IDENTITY_TOKEN_FILE} (a file holding your identity "
    f"provider's token, re-read on every exchange) or {_ENV_IDENTITY_TOKEN}; add "
    f"{_ENV_SERVICE_ACCOUNT_ID} and {_ENV_WORKSPACE_ID} when the federation rule needs them. "
    "Inside a Modal function the identity token is taken from Modal automatically, and "
    f"{_ENV_API_KEY} must be unset because it takes precedence."
)

_STATIC_TOKEN_CAVEAT = (
    f"{_ENV_IDENTITY_TOKEN} is a static token. A token that carries a jti claim can be "
    "exchanged only once, so a run that outlives the first access token will fail to "
    f"refresh it; prefer {_ENV_IDENTITY_TOKEN_FILE} pointing at a token file that is refreshed."
)

#: Error types of a mid-stream ``error`` event after which the same request may be sent again.
_TRANSIENT_STREAM_ERRORS = frozenset(
    {"overloaded_error", "api_error", "rate_limit_error", "timeout_error"}
)

_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MEMORY_COMMANDS = ("view", "create", "str_replace", "insert", "delete", "rename")
_PIL_MEDIA_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _import_sdk() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMConfigError(
            f"The anthropic package is not installed ({exc}). Install it with `uv sync`."
        ) from exc
    return anthropic


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def bridge_identity_token(environ: Mapping[str, str]) -> dict[str, str]:
    """Environment additions that let the SDK's WIF chain use Modal's OIDC token.

    Returns ``{}`` unless the federation rule and organization are configured
    and no identity token has been chosen explicitly.  Modal's token file is
    preferred because Modal keeps refreshing it and the SDK re-reads the file
    on every token exchange.  ``MODAL_IDENTITY_TOKEN`` is the initial token and
    never changes: a token that carries a ``jti`` claim can be exchanged only
    once, so with it a run that outlives the first access token cannot renew
    its credentials.

    The bridge does not make WIF win: an ``ANTHROPIC_API_KEY`` (or auth token,
    or ``ANTHROPIC_PROFILE``) in the environment still takes precedence in the
    SDK, which :func:`describe_credentials` reports as a warning.
    """
    if not (environ.get(_ENV_RULE_ID) and environ.get(_ENV_ORGANIZATION_ID)):
        return {}
    if environ.get(_ENV_IDENTITY_TOKEN_FILE) or environ.get(_ENV_IDENTITY_TOKEN):
        return {}
    if path := environ.get(_ENV_MODAL_IDENTITY_TOKEN_FILE):
        return {_ENV_IDENTITY_TOKEN_FILE: path}
    if token := environ.get(_ENV_MODAL_IDENTITY_TOKEN):
        return {_ENV_IDENTITY_TOKEN: token}
    return {}


def _apply_identity_bridge() -> str | None:
    """Apply :func:`bridge_identity_token` to ``os.environ``; return the Modal variable used."""
    additions = bridge_identity_token(os.environ)
    if not additions:
        return None
    os.environ.update(additions)
    source = (
        _ENV_MODAL_IDENTITY_TOKEN_FILE
        if _ENV_IDENTITY_TOKEN_FILE in additions
        else _ENV_MODAL_IDENTITY_TOKEN
    )
    logger.info("Workload Identity Federation will use Modal's identity token from %s.", source)
    return source


def _without_userinfo(url: str) -> str:
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def describe_credentials(
    sdk_client: Any, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Report which credential an SDK client will send, without a network call.

    ``credential_source`` is one of ``api_key``, ``auth_token``,
    ``workload_identity``, ``profile``, ``custom`` or ``none``.  ``warnings``
    lists the precedence traps that apply (a static credential or a profile
    shadowing the WIF variables, a missing token file).  Identifiers are
    included; keys and tokens never are.
    """
    env = os.environ if environ is None else environ
    wif_ids = bool(env.get(_ENV_RULE_ID) and env.get(_ENV_ORGANIZATION_ID))
    warnings: list[str] = []
    info: dict[str, Any] = {
        "credential_source": "none",
        "base_url": _without_userinfo(str(getattr(sdk_client, "base_url", "") or "")),
        "warnings": warnings,
    }
    credentials = getattr(sdk_client, "credentials", None)

    if getattr(sdk_client, "api_key", None) is not None:
        info["credential_source"] = "api_key"
        if env.get(_ENV_AUTH_TOKEN):
            warnings.append(f"Both {_ENV_API_KEY} and {_ENV_AUTH_TOKEN} are set; the API key wins.")
        if wif_ids and env.get(_ENV_API_KEY):
            warnings.append(_shadow_warning(_ENV_API_KEY))
    elif getattr(sdk_client, "auth_token", None) is not None:
        info["credential_source"] = "auth_token"
        if wif_ids and env.get(_ENV_AUTH_TOKEN):
            warnings.append(_shadow_warning(_ENV_AUTH_TOKEN))
    elif credentials is not None:
        anthropic = _import_sdk()
        if isinstance(credentials, anthropic.WorkloadIdentityCredentials):
            info["credential_source"] = "workload_identity"
            info.update(_describe_workload_identity(credentials, env, warnings))
        elif isinstance(credentials, anthropic.CredentialsFile):
            info["credential_source"] = "profile"
            info["profile"] = getattr(credentials, "profile", None)
            config_path = getattr(credentials, "config_path", None)
            info["config_path"] = str(config_path) if config_path is not None else None
            if wif_ids and env.get(_ENV_PROFILE):
                warnings.append(_shadow_warning(_ENV_PROFILE))
            elif wif_ids:
                warnings.append(
                    "The Workload Identity Federation variables are set without an identity "
                    f"token ({_ENV_IDENTITY_TOKEN_FILE} or {_ENV_IDENTITY_TOKEN}), so the "
                    "profile on disk is used instead."
                )
        else:
            info["credential_source"] = "custom"
    return info


def _shadow_warning(variable: str) -> str:
    return (
        f"{variable} is set and takes precedence over the Workload Identity Federation "
        f"variables; unset {variable} to authenticate with WIF."
    )


def _describe_workload_identity(
    credentials: Any, env: Mapping[str, str], warnings: list[str]
) -> dict[str, Any]:
    # The SDK keeps these on private attributes (anthropic 2.8); fall back to the
    # environment they were read from if a later release renames them.
    details: dict[str, Any] = {
        "federation_rule_id": getattr(credentials, "_federation_rule_id", None)
        or env.get(_ENV_RULE_ID),
        "organization_id": getattr(credentials, "_organization_id", None)
        or env.get(_ENV_ORGANIZATION_ID),
        "service_account_id": getattr(credentials, "_service_account_id", None)
        or env.get(_ENV_SERVICE_ACCOUNT_ID)
        or None,
        "workspace_id": getattr(credentials, "_workspace_id", None)
        or env.get(_ENV_WORKSPACE_ID)
        or None,
    }
    provider = getattr(credentials, "_identity_token_provider", None)
    token_path = getattr(provider, "path", None) or env.get(_ENV_IDENTITY_TOKEN_FILE)
    if token_path:
        details["identity_token"] = f"file:{token_path}"
        if not Path(token_path).is_file():
            warnings.append(
                f"The identity token file {token_path} does not exist; the first request "
                "will fail until the identity provider writes it."
            )
    elif env.get(_ENV_IDENTITY_TOKEN):
        details["identity_token"] = f"env:{_ENV_IDENTITY_TOKEN}"
        warnings.append(_STATIC_TOKEN_CAVEAT)
    else:
        details["identity_token"] = "custom provider"
    return details


def _missing_credentials_message(env: Mapping[str, str]) -> str:
    message = f"No Anthropic credentials were found. {_SETUP_HELP}"
    wif_names = (_ENV_RULE_ID, _ENV_ORGANIZATION_ID, _ENV_IDENTITY_TOKEN_FILE, _ENV_IDENTITY_TOKEN)
    present = [name for name in wif_names if env.get(name)]
    if not present:
        return message
    missing = [name for name in (_ENV_RULE_ID, _ENV_ORGANIZATION_ID) if not env.get(name)]
    if not (env.get(_ENV_IDENTITY_TOKEN_FILE) or env.get(_ENV_IDENTITY_TOKEN)):
        missing.append(f"{_ENV_IDENTITY_TOKEN_FILE} (or {_ENV_IDENTITY_TOKEN})")
    if not missing:
        return message
    return (
        f"{message}\nWorkload Identity Federation is only partly configured: "
        f"{', '.join(present)} set, {', '.join(missing)} missing."
    )


def structured_outputs_enabled(environ: Mapping[str, str]) -> bool:
    """Whether native structured outputs may be used (see ``STRUCTURED_OUTPUTS_ENV``)."""
    value = environ.get(STRUCTURED_OUTPUTS_ENV, "").strip().lower()
    return value not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def normalize_fallback_turn(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make an assistant turn that crossed a server-side fallback safe to replay.

    Blocks before the last ``fallback`` block came from the model that
    declined; of those only text and completed server-tool exchanges may be
    echoed back.  The fallback block and everything after it stay as received.
    A turn without a fallback block is returned unchanged.
    """
    index = max(
        (i for i, block in enumerate(content) if block.get("type") == "fallback"), default=None
    )
    if index is None:
        return content
    head, tail = content[:index], content[index:]
    answered = {
        block.get("tool_use_id")
        for block in head
        if str(block.get("type", "")).endswith("_tool_result")
    }
    kept = [
        block
        for block in head
        if block.get("type") == "text"
        or str(block.get("type", "")).endswith("_tool_result")
        or (block.get("type") == "server_tool_use" and block.get("id") in answered)
    ]
    return kept + tail


def _plain_json(value: Any, what: str) -> Any:
    """A private, plain-JSON copy of ``value`` (so frozen request parts cannot be mutated)."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise LLMConfigError(f"The {what} is not JSON-serialisable: {exc}") from exc


def _open_objects(node: Any, path: str = "$") -> list[str]:
    """Paths of object schemas without declared properties.

    Structured outputs require ``additionalProperties: false`` everywhere, so
    the model can only ever return ``{}`` for such an object (this is what a
    pydantic ``dict[str, X]`` field turns into).
    """
    if not isinstance(node, dict):
        return []
    found: list[str] = []
    types = node.get("type")
    if (types == "object" or (isinstance(types, list) and "object" in types)) and not node.get(
        "properties"
    ):
        found.append(path)
    for key in ("properties", "$defs"):
        children = node.get(key)
        if isinstance(children, dict):
            for name, child in children.items():
                found.extend(_open_objects(child, f"{path}.{name}"))
    if "items" in node:
        found.extend(_open_objects(node["items"], f"{path}[]"))
    for key in ("anyOf", "allOf"):
        children = node.get(key)
        if isinstance(children, list):
            for position, child in enumerate(children):
                found.extend(_open_objects(child, f"{path}.{key}[{position}]"))
    return found


def json_output_format(schema: Schema) -> dict[str, Any]:
    """The ``output_config.format`` value for a pydantic model or a JSON schema.

    The schema goes through the SDK's own ``transform_schema``, so the request
    is the one the SDK's ``output_format=Model`` convenience would send.  The
    response is validated by this module instead, after ``stop_reason`` has
    been checked: the SDK helper validates every text block while the stream is
    still open, which turns a truncated or refused response into a bare
    validation error before the stop reason and usage are known.
    """
    anthropic = _import_sdk()
    try:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            transformed = anthropic.transform_schema(schema)
        elif isinstance(schema, Mapping):
            transformed = anthropic.transform_schema(_plain_json(dict(schema), "output schema"))
        else:
            raise TypeError(
                f"expected a pydantic model class or a JSON schema mapping, got {schema!r}"
            )
    except (TypeError, ValueError) as exc:
        raise LLMConfigError(
            f"The output schema cannot be used for native structured output: {exc}"
        ) from exc
    open_paths = _open_objects(transformed)
    if open_paths:
        raise LLMConfigError(
            "The output schema has open-ended objects (for example a dict[str, X] field) at "
            f"{', '.join(open_paths)}. Native structured output can only return {{}} for them; "
            "use a list of key/value objects, or parse the response text instead."
        )
    return {"type": "json_schema", "schema": transformed}


def _client_tool_definition(tool: ToolSpec) -> dict[str, Any]:
    schema = _plain_json(tool.input_schema, f"input_schema of tool '{tool.name}'")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise LLMConfigError(
            f"The input_schema of tool '{tool.name}' must be a JSON schema of type 'object'."
        )
    return {"name": tool.name, "description": tool.description, "input_schema": schema}


def _strict_tool_definition(definition: Mapping[str, Any]) -> dict[str, Any] | None:
    """The strict (grammar-constrained input) form of a client tool, if it has one.

    A closed schema (``additionalProperties: false``) opts a tool in.  Schemas
    that strict mode cannot express keep their plain definition.
    """
    schema = definition["input_schema"]
    if schema.get("additionalProperties") is not False:
        return None
    try:
        strict_schema = _import_sdk().transform_schema(schema)
    except (TypeError, ValueError):
        return None
    if _open_objects(strict_schema):
        return None
    return {**definition, "input_schema": strict_schema, "strict": True}


def _web_search_definition(web_search: WebSearch) -> dict[str, Any]:
    if web_search.allowed_domains and web_search.blocked_domains:
        raise LLMConfigError(
            "Web search accepts allowed_domains or blocked_domains, not both in one conversation."
        )
    if web_search.max_uses < 1:
        raise LLMConfigError("WebSearch.max_uses must be at least 1.")
    definition: dict[str, Any] = {
        "type": WEB_SEARCH_TOOL_TYPE,
        "name": WEB_SEARCH_TOOL_NAME,
        "max_uses": web_search.max_uses,
    }
    if web_search.allowed_domains:
        definition["allowed_domains"] = list(web_search.allowed_domains)
    if web_search.blocked_domains:
        definition["blocked_domains"] = list(web_search.blocked_domains)
    return definition


def _tool_definitions(
    tools: Sequence[ToolSpec],
    memory: MemoryBackend | None,
    web_search: WebSearch | None,
    *,
    strict_allowed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, ToolSpec]]:
    """Return ``(tools, tools without strict mode, handlers by name)``.

    The second list is ``None`` when no tool is strict, i.e. when it would equal the first.
    """
    definitions: list[dict[str, Any]] = []
    plain_definitions: list[dict[str, Any]] = []
    handlers: dict[str, ToolSpec] = {}
    if memory is not None:
        definitions.append({"type": MEMORY_TOOL_TYPE, "name": MEMORY_TOOL_NAME})
        plain_definitions.append(definitions[-1])
    for tool in tools:
        if not _TOOL_NAME.match(tool.name):
            raise LLMConfigError(
                f"Invalid tool name {tool.name!r}: use 1-64 letters, digits, '_' or '-'."
            )
        if tool.name in (MEMORY_TOOL_NAME, WEB_SEARCH_TOOL_NAME):
            raise LLMConfigError(f"The tool name '{tool.name}' is reserved for a built-in tool.")
        if tool.name in handlers:
            raise LLMConfigError(f"Two tools are named '{tool.name}'; tool names must be unique.")
        handlers[tool.name] = tool
        plain = _client_tool_definition(tool)
        strict = _strict_tool_definition(plain) if strict_allowed else None
        definitions.append(strict or plain)
        plain_definitions.append(plain)
    if web_search is not None:
        definitions.append(_web_search_definition(web_search))
        plain_definitions.append(definitions[-1])
    has_strict = any(definition.get("strict") for definition in definitions)
    return definitions, plain_definitions if has_strict else None, handlers


def _base64_size(data: bytes) -> int:
    return 4 * ((len(data) + 2) // 3)


def fit_image(image: Image) -> Image:
    """Return ``image`` if the API accepts it as is, else a downscaled or re-encoded copy.

    Called once, when a user turn is built.  Journaled turns are never
    re-encoded, because replayed media must stay byte-identical.
    """
    from PIL import Image as PILImage

    # Pillow reports undecodable data as OSError (unidentified or truncated
    # files), ValueError or SyntaxError (corrupt chunks), depending on the format.
    unreadable = (OSError, ValueError, SyntaxError, PILImage.DecompressionBombError)
    try:
        picture = PILImage.open(io.BytesIO(image.data))
    except unreadable as exc:
        raise LLMError(f"An image part could not be decoded: {exc}") from exc
    with picture:
        media_type = _PIL_MEDIA_TYPES.get(picture.format or "")
        if (
            media_type is not None
            and max(picture.size) <= MAX_IMAGE_EDGE
            and _base64_size(image.data) <= MAX_IMAGE_BASE64_BYTES
        ):
            return (
                image if media_type == image.media_type else replace(image, media_type=media_type)
            )
        try:
            return _reencode(picture)
        except unreadable as exc:
            raise LLMError(f"An image part could not be re-encoded: {exc}") from exc


def _reencode(picture: Any) -> Image:
    from PIL import Image as PILImage

    has_alpha = picture.mode in ("RGBA", "LA", "PA") or "transparency" in picture.info
    frame = picture.convert("RGBA" if has_alpha else "RGB")
    scale = min(1.0, MAX_IMAGE_EDGE / max(frame.size))
    for _ in range(8):
        size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
        candidate = frame if size == frame.size else frame.resize(size, PILImage.Resampling.LANCZOS)
        png = io.BytesIO()
        candidate.save(png, format="PNG")
        if _base64_size(png.getvalue()) <= MAX_IMAGE_BASE64_BYTES:
            return Image(png.getvalue(), "image/png")
        flat = candidate
        if has_alpha:
            flat = PILImage.new("RGB", candidate.size, "white")
            flat.paste(candidate, mask=candidate.getchannel("A"))
        jpeg = io.BytesIO()
        flat.save(jpeg, format="JPEG", quality=85)
        if _base64_size(jpeg.getvalue()) <= MAX_IMAGE_BASE64_BYTES:
            return Image(jpeg.getvalue(), "image/jpeg")
        scale *= 0.75
    raise LLMError(
        f"An image part could not be reduced below {MAX_IMAGE_BASE64_BYTES} base64 bytes."
    )


def _user_content(parts: Sequence[Text | Image | Pdf]) -> list[dict[str, Any]]:
    """Content blocks for a user turn: PDFs first, then images and text in the given order.

    Documents are read best ahead of the text that refers to them.  Images keep
    their position so that a caption or label next to an image stays next to it.
    """
    documents: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, Pdf):
            if not part.data:
                raise LLMError("A PDF part is empty.")
            block: dict[str, Any] = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": part.to_base64(),
                },
            }
            if part.title:
                block["title"] = part.title
            documents.append(block)
        elif isinstance(part, Image):
            fitted = fit_image(part)
            rest.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": fitted.media_type,
                        "data": fitted.to_base64(),
                    },
                }
            )
        elif part.text.strip():
            rest.append({"type": "text", "text": part.text})
    content = documents + rest
    if not content:
        raise LLMError("Cannot send an empty message: provide text, an image or a PDF.")
    return content


def _describe_parts(parts: Sequence[Text | Image | Pdf]) -> str:
    described = []
    for part in parts:
        if isinstance(part, Text):
            described.append(part.text)
        elif isinstance(part, Image):
            described.append(f"[image {part.media_type}, {len(part.data)} bytes]")
        else:
            described.append(f"[pdf {part.title or ''}, {len(part.data)} bytes]")
    return "\n".join(described)


def _text_of(content: Sequence[Mapping[str, Any]]) -> str:
    """The text of one assistant message.

    Adjacent text blocks are fragments of one passage (citations split prose
    into several blocks) and are joined directly; text separated by tool or
    thinking blocks is joined with a blank line.
    """
    chunks: list[str] = []
    separated = False
    for block in content:
        if block.get("type") == "text":
            text = block.get("text") or ""
            if chunks and separated and text:
                chunks.append("\n\n")
            chunks.append(text)
            separated = False
        elif chunks:
            separated = True
    return "".join(chunks)


def _count(source: Any, name: str) -> int:
    return int(getattr(source, name, 0) or 0)


def _usage_of(message: Any) -> Usage:
    """Usage of one request.

    ``usage.iterations`` lists every sampling attempt behind the response
    (server-tool loops, declined attempts before a fallback) and is the billing
    truth when present; the top-level counters only cover the last attempt.
    """
    usage = message.usage
    attempts = getattr(usage, "iterations", None) or [usage]
    total = Usage(requests=1)
    for attempt in attempts:
        total.input_tokens += _count(attempt, "input_tokens")
        total.output_tokens += _count(attempt, "output_tokens")
        total.cache_read_input_tokens += _count(attempt, "cache_read_input_tokens")
        total.cache_creation_input_tokens += _count(attempt, "cache_creation_input_tokens")
    total.web_searches = _count(getattr(usage, "server_tool_use", None), "web_search_requests")
    return total


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _field(tool_input: Mapping[str, Any], name: str, kind: type, command: str) -> Any:
    value = tool_input.get(name)
    # bool is an int subclass, and a line number of True is never what was meant.
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        expected = "a string" if kind is str else "an integer"
        raise ToolError(
            f"The memory command '{command}' needs '{name}' to be {expected}; "
            f"got {value!r}. Send the command again with a valid '{name}'."
        )
    return value


def run_memory_command(memory: MemoryBackend, tool_input: Any) -> str:
    """Dispatch one ``memory`` tool call to ``memory``.

    The API does not validate memory tool input, so every field is type-checked
    here and a malformed call comes back to the model as a :class:`ToolError`.
    """
    if not isinstance(tool_input, Mapping):
        raise ToolError("The memory tool input must be a JSON object with a 'command' field.")
    command = tool_input.get("command")
    if command == "view":
        path = _field(tool_input, "path", str, command)
        view_range = tool_input.get("view_range")
        if view_range is None:
            return memory.view(path)
        if (
            not isinstance(view_range, (list, tuple))
            or len(view_range) != 2
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in view_range)
        ):
            raise ToolError(
                "'view_range' must be a list of two integers [start_line, end_line] "
                f"(end_line -1 means the end of the file); got {view_range!r}."
            )
        return memory.view(path, (view_range[0], view_range[1]))
    if command == "create":
        return memory.create(
            _field(tool_input, "path", str, command),
            _field(tool_input, "file_text", str, command),
        )
    if command == "str_replace":
        path = _field(tool_input, "path", str, command)
        old_str = _field(tool_input, "old_str", str, command)
        # The documented contract makes new_str optional: leaving it out deletes old_str.
        new_str = (
            "" if tool_input.get("new_str") is None else _field(tool_input, "new_str", str, command)
        )
        return memory.str_replace(path, old_str, new_str)
    if command == "insert":
        return memory.insert(
            _field(tool_input, "path", str, command),
            _field(tool_input, "insert_line", int, command),
            _field(tool_input, "insert_text", str, command),
        )
    if command == "delete":
        return memory.delete(_field(tool_input, "path", str, command))
    if command == "rename":
        return memory.rename(
            _field(tool_input, "old_path", str, command),
            _field(tool_input, "new_path", str, command),
        )
    raise ToolError(
        f"Unknown memory command {command!r}; use one of {', '.join(_MEMORY_COMMANDS)}."
    )


def _check_tool_input(tool: ToolSpec, tool_input: Any) -> None:
    """Reject input that misses the top level of the tool's schema.

    Non-strict tool input is not validated by the API.  This checks what a
    handler relies on most (an object, the required keys, primitive types and
    enums of top-level properties); deeper validation is the handler's own.
    Enum values are put back into their declared casing, which even
    grammar-constrained (strict) input does not guarantee.
    """
    if not isinstance(tool_input, dict):
        raise ToolError(f"The input of tool '{tool.name}' must be a JSON object.")
    schema = tool.input_schema
    required = [name for name in schema.get("required") or [] if isinstance(name, str)]
    missing = [name for name in required if name not in tool_input]
    if missing:
        raise ToolError(
            f"Tool '{tool.name}' is missing required parameter(s): {', '.join(missing)}."
        )
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    if schema.get("additionalProperties") is False:
        unknown = [name for name in tool_input if name not in properties]
        if unknown:
            raise ToolError(
                f"Tool '{tool.name}' does not accept parameter(s): {', '.join(unknown)}. "
                f"Accepted parameters: {', '.join(properties) or 'none'}."
            )
    for name, value in list(tool_input.items()):
        rule = properties.get(name)
        if not isinstance(rule, Mapping) or (value is None and name not in required):
            continue
        declared = rule.get("type")
        names = [declared] if isinstance(declared, str) else declared
        names = [n for n in names if isinstance(n, str)] if isinstance(names, list) else []
        accepted = tuple(kind for n in names for kind in _JSON_TYPES.get(n, ()))
        is_bool_as_number = isinstance(value, bool) and "boolean" not in names
        if accepted and (not isinstance(value, accepted) or is_bool_as_number):
            raise ToolError(
                f"Parameter '{name}' of tool '{tool.name}' must be of type "
                f"{' or '.join(names)}; got {value!r}."
            )
        choices = rule.get("enum")
        if isinstance(choices, list) and value not in choices:
            same = [
                choice
                for choice in choices
                if isinstance(choice, str)
                and isinstance(value, str)
                and choice.lower() == value.lower()
            ]
            if not same:
                raise ToolError(
                    f"Parameter '{name}' of tool '{tool.name}' must be one of "
                    f"{', '.join(repr(choice) for choice in choices)}; got {value!r}."
                )
            tool_input[name] = same[0]


def _result_text(output: Any) -> str:
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False, default=str)
    # The API rejects empty text content in a tool result.
    return output if output.strip() else "(no output)"


# ---------------------------------------------------------------------------
# Structured output and text selection
# ---------------------------------------------------------------------------


def _parse_structured(schema: Schema, text: str) -> tuple[Any, str]:
    """Validate ``text`` against ``schema``; return ``(parsed, problem)``."""
    if isinstance(schema, type):
        try:
            return schema.model_validate_json(text), ""
        except ValidationError as exc:
            return None, str(exc)
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _select_text(closing: str, everything: str, parser: Parser | None) -> tuple[str, Any]:
    """Pick the response text and parse it.

    The answer normally is the closing turn (what the model wrote after its
    last tool result).  A model sometimes delivers the answer together with a
    final tool call, such as saving a note, and then closes with a short
    confirmation; when the closing turn alone is empty or does not parse,
    everything the model wrote during the exchange is used instead.
    """
    candidates = [closing] if closing == everything else [closing, everything]
    for candidate in candidates:
        if not candidate.strip():
            continue
        if parser is None:
            return candidate, None
        parsed = parser(candidate)
        if parsed is not None:
            return candidate, parsed
    return closing, None


def _resolve_max_tokens(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LLMConfigError(f"max_tokens must be a positive integer, got {value!r}.")
    return min(value, MAX_OUTPUT_TOKENS)


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class AnthropicConversation(Conversation):
    """An append-only conversation with a manual tool loop.

    ``send`` is atomic with respect to the journal: the user turn, the
    assistant turns and the tool results of one exchange are committed together
    when the model reaches its end of turn.  A refusal, a truncation or an API
    error leaves the journal exactly as it was, so the caller may retry or
    rephrase.  (Effects of tool calls that already ran are not undone.)
    """

    def __init__(
        self,
        sdk_client: Any,
        *,
        model: str,
        system: str | None,
        tools: Sequence[ToolSpec],
        memory: MemoryBackend | None,
        web_search: WebSearch | None,
        effort: Effort | None,
        max_tokens: int | None,
        max_turns: int,
        label: str,
        on_event: EventSink | None,
        structured_outputs: bool,
    ) -> None:
        if not model or not isinstance(model, str):
            raise LLMConfigError("A conversation needs a model id.")
        if max_turns < 1:
            raise LLMConfigError("max_turns must be at least 1.")
        self.model = model
        self._sdk = sdk_client
        self._system = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if system and system.strip()
            else None
        )
        # `_plain_tools` is the fallback without strict mode.  It exists only
        # until the API has accepted a request of this conversation: from then
        # on the tool definitions are part of the history and must not change.
        self._tools, self._plain_tools, self._handlers = _tool_definitions(
            tools, memory, web_search, strict_allowed=structured_outputs
        )
        self._memory = memory
        self._web_search = web_search
        self._effort: Effort = effort or DEFAULT_EFFORT
        self._max_tokens = _resolve_max_tokens(
            DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        )
        self._max_turns = max_turns
        self._label = label
        self._on_event = on_event
        self._structured_outputs = structured_outputs
        self._journal: list[dict[str, Any]] = []
        self._usage = Usage()

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def journal(self) -> list[dict[str, Any]]:
        """A copy of the committed wire-JSON messages, for inspection or persistence."""
        return copy.deepcopy(self._journal)

    def send(
        self,
        parts: Part | Sequence[Part],
        *,
        parser: Parser | None = None,
        schema: Schema | None = None,
        require_parsed: bool = True,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Append a user turn and run the tool loop to the model's end of turn.

        ``LLMResponse.text`` is what the model wrote after its last tool result
        (see :func:`_select_text` for the one exception).  ``usage`` covers the
        requests of this call; :attr:`usage` keeps the running total, including
        requests that ended in a refusal or a truncation.
        """
        # Checked before anything is sent or recorded, so that the caller can
        # fall back to text parsing on the same, untouched conversation.
        output_format = self._output_format(schema)
        typed = normalize_parts(parts)
        user_message = {"role": "user", "content": _user_content(typed)}
        limit = self._max_tokens if max_tokens is None else _resolve_max_tokens(max_tokens)

        self._emit(
            {
                "type": "request",
                "label": self._label,
                "model": self.model,
                "user": _describe_parts(typed),
            }
        )
        try:
            return self._exchange(
                user_message, limit, output_format, parser, schema, require_parsed
            )
        except LLMError as exc:
            self._emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            raise

    def _output_format(self, schema: Schema | None) -> dict[str, Any] | None:
        if schema is None:
            return None
        if not self._structured_outputs:
            raise LLMConfigError(
                f"Native structured outputs are disabled ({STRUCTURED_OUTPUTS_ENV}=0); "
                "ask for JSON in the prompt and parse the response text instead."
            )
        if self._web_search is not None:
            raise LLMConfigError(
                "Native structured output cannot be combined with web search in one "
                "conversation (search results carry citations, which a response format "
                "rejects); ask for JSON in the prompt and parse the response text instead."
            )
        return json_output_format(schema)

    def _exchange(
        self,
        user_message: dict[str, Any],
        max_tokens: int,
        output_format: dict[str, Any] | None,
        parser: Parser | None,
        schema: Schema | None,
        require_parsed: bool,
    ) -> LLMResponse:
        turn: list[dict[str, Any]] = [user_message]
        sent = Usage()
        texts: list[str] = []
        closing_from = 0
        tool_calls = 0
        pauses = 0
        for step in range(self._max_turns):
            message, request_id = self._request(
                [*self._journal, *turn],
                max_tokens,
                output_format,
                format_is_untested=output_format is not None and step == 0,
            )
            usage = _usage_of(message)
            sent.add(usage)
            self._usage.add(usage)
            content = normalize_fallback_turn(message.to_dict(warnings=False)["content"])
            text = _text_of(content)
            stop_reason = message.stop_reason
            self._emit(
                {
                    "type": "assistant",
                    "text": text,
                    "stop_reason": stop_reason,
                    "model": message.model,
                    "request_id": request_id,
                }
            )
            self._emit({"type": "usage", **usage.to_dict()})
            logger.debug(
                "%s: %s after %d output tokens%s",
                self._label or self.model,
                stop_reason,
                usage.output_tokens,
                self._where(request_id),
            )

            if stop_reason == "refusal":
                raise self._refusal(message, request_id)
            if stop_reason in ("max_tokens", "model_context_window_exceeded"):
                raise self._truncated(stop_reason, max_tokens, content, text, request_id)

            texts.append(text)
            if content:
                turn.append({"role": "assistant", "content": content})
            if stop_reason == "pause_turn":
                # A server-tool loop paused: the turn is re-sent as it is and the
                # model continues it.  No user message goes in between.
                pauses += 1
                if pauses > MAX_PAUSES:
                    raise LLMError(
                        f"The model paused its turn more than {MAX_PAUSES} times"
                        f"{self._where(request_id)}."
                    )
                continue
            if stop_reason != "tool_use":
                break
            calls = [block for block in content if block.get("type") == "tool_use"]
            if not calls:
                raise LLMError(
                    f"The response stopped for tool use without a tool call"
                    f"{self._where(request_id)}."
                )
            # Every result of the turn goes back in one user message; splitting
            # them teaches the model to stop calling tools in parallel.
            turn.append({"role": "user", "content": [self._run_tool(block) for block in calls]})
            tool_calls += len(calls)
            closing_from = len(texts)
        else:
            raise LLMError(
                f"The model did not finish within max_turns={self._max_turns} requests"
                f"{self._where(request_id)}."
            )

        self._journal.extend(turn)

        closing = "\n\n".join(t for t in texts[closing_from:] if t)
        parsed: Any = None
        problem = ""
        if schema is not None:
            text = closing
            parsed, problem = _parse_structured(schema, closing)
        else:
            text, parsed = _select_text(closing, "\n\n".join(t for t in texts if t), parser)
        if parsed is None and (schema is not None or parser is not None):
            reason = f": {problem}" if problem else ""
            if require_parsed:
                raise LLMParseError(
                    f"The response could not be parsed into the requested shape"
                    f"{self._where(request_id)}{reason}",
                    raw_text=text,
                )
            logger.info(
                "%s: response did not parse%s%s",
                self._label or self.model,
                self._where(request_id),
                reason,
            )
        elif not text.strip():
            logger.warning("The model ended its turn without any text%s.", self._where(request_id))
        return LLMResponse(
            text=text,
            parsed=parsed,
            model=message.model,
            usage=sent,
            stop_reason=stop_reason,
            request_id=request_id,
            tool_calls=tool_calls,
            raw=message.to_dict(warnings=False),
        )

    # -- one HTTP request ---------------------------------------------------

    def _request(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        output_format: dict[str, Any] | None,
        *,
        format_is_untested: bool,
    ) -> tuple[Any, str | None]:
        """Send one request and map SDK failures onto :class:`LLMError`."""
        anthropic = _import_sdk()
        output_config: dict[str, Any] = {"effort": self._effort}
        if output_format is not None:
            output_config["format"] = output_format
        # No `thinking` (adaptive is the default and the only mode Fable 5.2
        # accepts), no sampling parameters, no forced tool choice, no prefill:
        # each of those is rejected by the current models.
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "output_config": output_config,
            # Moves a cache breakpoint along the growing conversation; the
            # system block carries its own, so tools and system stay cached.
            "cache_control": {"type": "ephemeral"},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if self._system is not None:
            request["system"] = self._system
        if self._tools:
            request["tools"] = self._tools
        try:
            response = self._stream(request)
        except anthropic.APIStatusError as exc:
            if exc.status_code == 400 and self._plain_tools is not None:
                # Strict tools are not available for every model and organization
                # and share a complexity budget with the response format.  Nothing
                # has been answered yet, so the tools can still be declared plainly.
                logger.warning(
                    "The first request was rejected%s (%s); retrying without strict tools.",
                    self._where(getattr(exc, "request_id", None)),
                    getattr(exc, "api_message", None) or exc,
                )
                self._tools, self._plain_tools = self._plain_tools, None
                return self._request(
                    messages, max_tokens, output_format, format_is_untested=format_is_untested
                )
            raise self._status_error(exc, format_is_untested=format_is_untested) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(
                f"Could not reach the Anthropic API{self._where(None)}: {type(exc).__name__}: {exc}"
            ) from exc
        except (anthropic.CredentialsError, anthropic.WorkloadIdentityError) as exc:
            raise LLMConfigError(
                f"Anthropic credentials could not be used{self._where(None)}: "
                f"{type(exc).__name__}: {exc}. For Workload Identity Federation, the reason "
                "for a denied token exchange is in the Console's federation history."
            ) from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(
                f"The Anthropic SDK failed{self._where(None)}: {type(exc).__name__}: {exc}"
            ) from exc
        # Accepted: the tool definitions now belong to the history.
        self._plain_tools = None
        return response

    def _stream(self, request: dict[str, Any]) -> tuple[Any, str | None]:
        """Stream one response to completion, re-issuing it if the stream breaks.

        Nothing is journaled until a response is complete, so sending the same
        request again is safe.
        """
        import httpx2

        anthropic = _import_sdk()
        interruption = ""
        for attempt in range(STREAM_RESTARTS + 1):
            if attempt:
                logger.warning(
                    "Response stream interrupted%s (%s); sending the same request again "
                    "(%d of %d).",
                    self._where(None),
                    interruption,
                    attempt,
                    STREAM_RESTARTS,
                )
                time.sleep(STREAM_RESTART_DELAY_S)
            message = None
            request_id: str | None = None
            try:
                with self._sdk.beta.messages.stream(**request) as stream:
                    request_id = stream.request_id
                    message = stream.get_final_message()
            except (httpx2.TransportError, ValueError) as exc:
                # A dropped connection, or stream data the SDK could not decode
                # (it raises ValueError for malformed event or tool-input JSON).
                interruption = f"{type(exc).__name__}: {exc}"
                continue
            except AssertionError:
                # The SDK asserts when the stream ended before `message_start`.
                interruption = "the stream ended before the response started"
                continue
            except anthropic.APIStatusError as exc:
                # An `error` event inside a 200 stream; only transient kinds are retried.
                if exc.status_code >= 400 or getattr(exc, "type", None) not in (
                    _TRANSIENT_STREAM_ERRORS
                ):
                    raise
                interruption = f"{getattr(exc, 'type', 'error')} event"
                continue
            if message is None or message.stop_reason is None:
                interruption = "the stream ended before the response finished"
                continue
            return message, request_id
        raise LLMError(
            f"The response stream broke {STREAM_RESTARTS + 1} times in a row"
            f"{self._where(None)}: {interruption}."
        )

    # -- tools --------------------------------------------------------------

    def _run_tool(self, block: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one client tool call and return its ``tool_result`` block."""
        name = block.get("name")
        # A copy: the block itself is part of the journal and must not change
        # if a handler modifies its input.
        tool_input = copy.deepcopy(block.get("input"))
        self._emit({"type": "tool_call", "name": name, "input": copy.deepcopy(tool_input)})
        is_error = False
        try:
            if name == MEMORY_TOOL_NAME and self._memory is not None:
                output = run_memory_command(self._memory, tool_input)
            elif name in self._handlers:
                tool = self._handlers[name]
                _check_tool_input(tool, tool_input)
                output = tool.handler(tool_input)
            else:
                available = [*self._handlers]
                if self._memory is not None:
                    available.insert(0, MEMORY_TOOL_NAME)
                raise ToolError(
                    f"Unknown tool '{name}'. Available tools: {', '.join(available) or 'none'}."
                )
        except ToolError as exc:
            output, is_error = str(exc) or "The tool reported an error.", True
        except Exception as exc:
            logger.warning("Tool '%s' raised %s: %s", name, type(exc).__name__, exc, exc_info=True)
            output, is_error = f"{type(exc).__name__}: {exc}", True
        output = _result_text(output)
        self._emit({"type": "tool_result", "name": name, "is_error": is_error, "content": output})
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.get("id"),
            "content": output,
        }
        if is_error:
            result["is_error"] = True
        return result

    # -- errors and events --------------------------------------------------

    def _where(self, request_id: str | None) -> str:
        details = [f"model={self.model}"]
        if self._label:
            details.insert(0, f"label={self._label}")
        if request_id:
            details.append(f"request_id={request_id}")
        return f" [{', '.join(details)}]"

    def _refusal(self, message: Any, request_id: str | None) -> LLMRefusal:
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        text = f"{message.model} declined the request"
        if category:
            text += f" (category: {category})"
        text += self._where(request_id)
        if explanation:
            text += f": {explanation}"
        return LLMRefusal(text, category=category)

    def _truncated(
        self,
        stop_reason: str,
        max_tokens: int,
        content: Sequence[Mapping[str, Any]],
        text: str,
        request_id: str | None,
    ) -> LLMTruncated:
        if stop_reason == "model_context_window_exceeded":
            cause = "The response ran into the model's context window"
        elif any(block.get("type") == "tool_use" for block in content):
            cause = f"The response hit max_tokens={max_tokens} in the middle of a tool call"
        else:
            cause = f"The response hit max_tokens={max_tokens} before it was finished"
        return LLMTruncated(
            f"{cause}{self._where(request_id)}. Thinking counts towards max_tokens; raise it "
            f"(up to {MAX_OUTPUT_TOKENS}) or ask for less output.",
            partial_text=text,
        )

    def _status_error(self, exc: Any, *, format_is_untested: bool) -> LLMError:
        status = exc.status_code
        kind = getattr(exc, "type", None) or "error"
        detail = getattr(exc, "api_message", None) or str(exc)
        where = self._where(getattr(exc, "request_id", None))
        if status == 400 and format_is_untested:
            return LLMConfigError(
                f"The API rejected a request that asked for native structured output{where}: "
                f"{detail}. Ask for JSON in the prompt and parse the response text instead "
                f"({STRUCTURED_OUTPUTS_ENV}=0 turns native structured output off for good)."
            )
        if status in (401, 403):
            return LLMConfigError(
                f"The Anthropic API rejected the credentials or their permissions "
                f"({status} {kind}){where}: {detail}"
            )
        return LLMError(f"Anthropic API error {status} ({kind}){where}: {detail}")

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            # The transcript is an observer; losing an event must not lose a response.
            logger.warning(
                "The event sink failed on a '%s' event.", event.get("type"), exc_info=True
            )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    """Claude models through the first-party Anthropic API."""

    provider = "anthropic"
    capabilities = Capabilities(
        tools=True,
        memory_tool=True,
        web_search=True,
        pdf_input=True,
        image_input=True,
        structured_output=True,
    )

    def __init__(
        self, *, http_client: Any | None = None, max_retries: int = SDK_MAX_RETRIES
    ) -> None:
        """Build the SDK client and make sure a credential resolves.

        ``http_client`` is an ``httpx2.Client`` (for example
        ``anthropic.DefaultHttpxClient(proxy=...)``); tests pass one with a mock
        transport.  Raises :class:`LLMConfigError` when no credential is found.
        """
        anthropic = _import_sdk()
        self._identity_bridge = _apply_identity_bridge()
        options: dict[str, Any] = {
            "max_retries": max_retries,
            "timeout": anthropic.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        }
        if http_client is not None:
            options["http_client"] = http_client
        try:
            # No explicit key: the SDK's own resolution order applies.
            self._sdk = anthropic.Anthropic(**options)
        except anthropic.AnthropicError as exc:
            raise LLMConfigError(
                f"The Anthropic credentials could not be loaded: {exc}\n{_SETUP_HELP}"
            ) from exc

        # Without a credential the SDK only fails on the first request, with a
        # TypeError; say what to configure now instead.
        credentials = describe_credentials(self._sdk)
        if credentials["credential_source"] == "none":
            raise LLMConfigError(_missing_credentials_message(os.environ))
        for warning in credentials["warnings"]:
            logger.warning("%s", warning)
        logger.info(
            "Anthropic client ready (credential source: %s).", credentials["credential_source"]
        )

        self._structured_outputs = structured_outputs_enabled(os.environ)
        if not self._structured_outputs:
            self.capabilities = replace(self.capabilities, structured_output=False)

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
    ) -> Conversation:
        if temperature is not None:
            logger.debug(
                "temperature=%s is ignored: Claude rejects sampling parameters.", temperature
            )
        return AnthropicConversation(
            self._sdk,
            model=model,
            system=system,
            tools=tools,
            memory=memory,
            web_search=web_search,
            effort=effort,
            max_tokens=max_tokens,
            max_turns=max_turns,
            label=label,
            on_event=on_event,
            structured_outputs=self._structured_outputs,
        )

    def describe(self) -> dict[str, Any]:
        """Credential source and settings for ``paper-orchestra doctor`` (no network, no secrets)."""
        return {
            "provider": self.provider,
            **describe_credentials(self._sdk),
            "identity_token_bridge": self._identity_bridge,
            "structured_outputs": self._structured_outputs,
            "sdk_version": _import_sdk().__version__,
        }
