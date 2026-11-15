"""The context harness: what a model sees, and what it leaves behind.

Every model call of the pipeline goes through :class:`ContextHarness`.  It
decides what memory is in context (the injected index, pinned and retrieved
pages), what memory tools the session gets, and it records the session in the
transcript and the usage ledger.  Agents state *what* they need
(``MemoryPolicy``, ``pinned``, ``retrieve``); how that reaches the model
depends on the provider:

* **pull**: providers with a native memory tool get the wiki as ``/memories``
  plus search tools, and open pages on demand;
* **push**: other providers get the relevant pages placed in the prompt.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from paper_orchestra.llm.base import (
    Conversation,
    Effort,
    LLMClient,
    LLMConfigError,
    LLMParseError,
    LLMResponse,
    Parser,
    Part,
    Schema,
    Text,
    ToolSpec,
    Usage,
    UsageTracker,
    WebSearch,
    normalize_parts,
)
from paper_orchestra.llm.parsing import parse_json
from paper_orchestra.memory import prompts
from paper_orchestra.memory.tools import (
    WikiMemoryBackend,
    transcript_search_tool,
    virtual_path,
    wiki_search_tool,
)
from paper_orchestra.memory.transcript import TranscriptLog
from paper_orchestra.memory.wiki import INDEX_FILE, WIKI_DIR, Wiki, WikiError

if TYPE_CHECKING:
    from paper_orchestra.config import DreamMode, PipelineConfig
    from paper_orchestra.memory.dreaming import DreamReport

log = logging.getLogger(__name__)

WriteScope = Literal["none", "notes", "all"]


@dataclass(frozen=True)
class MemoryPolicy:
    """What a session may do with memory."""

    #: Inject the index and offer read/search access.
    read: bool = True
    write: WriteScope = "notes"
    lab_write: Literal["none", "all"] = "none"
    #: Offer ``transcript_search`` (consolidation sessions).
    transcripts: bool = False

    @classmethod
    def none(cls) -> MemoryPolicy:
        """A plain model call: nothing injected, no tools."""
        return cls(read=False, write="none")

    @classmethod
    def default(cls) -> MemoryPolicy:
        return cls()

    @classmethod
    def readonly(cls) -> MemoryPolicy:
        return cls(write="none")

    @classmethod
    def dream(cls, *, lab: bool = False) -> MemoryPolicy:
        if lab:
            return cls(write="none", lab_write="all", transcripts=True)
        return cls(write="all", transcripts=True)


def _schema_json(schema: Schema) -> str:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return json.dumps(schema.model_json_schema(), indent=1, sort_keys=True)
    return json.dumps(dict(schema), indent=1, sort_keys=True)


def _schema_parser(schema: Schema) -> Parser:
    """Text -> validated object, or ``None`` when it does not fit the schema."""

    def parse(text: str) -> Any:
        data = parse_json(text)
        if data is None:
            return None
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                return schema.model_validate(data)
            except ValidationError:
                return None
        return data

    return parse


class HarnessConversation:
    """A conversation opened through the harness (memory, transcript, usage wired in)."""

    def __init__(
        self,
        harness: ContextHarness,
        conversation: Conversation,
        *,
        client: LLMClient,
        label: str,
        writes: list[tuple[str, str]],
        text_only_schema: bool = False,
    ) -> None:
        self._harness = harness
        self._conversation = conversation
        self._client = client
        self._label = label
        self._first = True
        self._text_schema = text_only_schema
        #: ``(scope, path)`` of every memory file this session wrote.
        self.writes = writes
        self.model = conversation.model

    @property
    def usage(self) -> Usage:
        return self._conversation.usage

    def send(
        self,
        parts: Part | Sequence[Part],
        *,
        parser: Parser | None = None,
        schema: Schema | None = None,
        require_parsed: bool = True,
        max_tokens: int | None = None,
        pinned: Sequence[str] = (),
        retrieve: str | None = None,
        k: int = 6,
    ) -> LLMResponse:
        """Send a user turn.  ``pinned``/``retrieve`` add memory pages to the first turn."""
        content = normalize_parts(parts)
        if self._first and (pinned or retrieve):
            block = self._harness.context_block(pinned=pinned, query=retrieve, k=k)
            if block:
                content = [Text(block), *content]
        self._first = False
        response = self._send(content, parser, schema, require_parsed, max_tokens)
        self._harness.usage.record(self._label, self.model, response.usage)
        return response

    def _send(
        self,
        content: list,
        parser: Parser | None,
        schema: Schema | None,
        require_parsed: bool,
        max_tokens: int | None,
    ) -> LLMResponse:
        if schema is None:
            return self._conversation.send(
                content, parser=parser, require_parsed=require_parsed, max_tokens=max_tokens
            )
        if self._client.capabilities.structured_output and not self._text_schema:
            try:
                return self._conversation.send(
                    content, schema=schema, require_parsed=require_parsed, max_tokens=max_tokens
                )
            except LLMConfigError as exc:
                # The provider cannot honour this schema here (feature off, or it does not
                # combine with another option).  Ask for JSON in text for the rest of this
                # conversation instead.
                log.info("%s: native structured output unavailable (%s)", self._label, exc)
                self._text_schema = True
        instructions = Text(prompts.schema_instructions(_schema_json(schema)))
        return self._conversation.send(
            [*content, instructions],
            parser=_schema_parser(schema),
            require_parsed=require_parsed,
            max_tokens=max_tokens,
        )


class ContextHarness:
    """Shared by every agent of a run.  Thread-safe."""

    def __init__(
        self,
        *,
        project: Wiki | None = None,
        lab: Wiki | None = None,
        transcripts: TranscriptLog | None = None,
        client_for: Callable[[str], LLMClient] | None = None,
        usage: UsageTracker | None = None,
        default_effort: Effort | None = None,
        dream_mode: DreamMode = "stages",
        dream_model: str | None = None,
    ) -> None:
        if client_for is None:
            from paper_orchestra.llm.router import get_client

            client_for = get_client
        self._client_for = client_for
        self.project = project
        self.lab = lab if project is not None else None
        self.transcripts = transcripts
        self.usage = usage or UsageTracker()
        self.default_effort = default_effort
        self.dream_mode: DreamMode = dream_mode
        self.dream_model = dream_model
        self._lock = threading.RLock()
        self._dirty: set[str] = set()

    @classmethod
    def create(
        cls,
        cfg: PipelineConfig,
        *,
        client_for: Callable[[str], LLMClient] | None = None,
        usage: UsageTracker | None = None,
    ) -> ContextHarness:
        """Build the harness (wikis and transcript log) a run's configuration asks for."""
        common: dict[str, Any] = {
            "client_for": client_for,
            "usage": usage,
            "default_effort": cfg.models.effort,
            "dream_mode": cfg.memory.dream,
            "dream_model": cfg.models.reflection,
        }
        if not cfg.memory.enabled:
            return cls(**common)
        project = Wiki(
            cfg.project_memory_dir,
            name="project",
            description="What this run has learned about the paper being written.",
        )
        lab_dir = cfg.memory.resolved_lab_dir()
        lab = (
            Wiki(
                lab_dir,
                name="lab",
                description="Process lessons shared across papers. Never paper content.",
                read_only=(),
            )
            if lab_dir is not None
            else None
        )
        transcripts = TranscriptLog(project.root / "transcripts")
        return cls(project=project, lab=lab, transcripts=transcripts, **common)

    # -- basics ----------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.project is not None

    def client_for(self, model: str) -> LLMClient:
        return self._client_for(model)

    def _wiki(self, scope: str) -> Wiki | None:
        return self.lab if scope == "lab" else self.project

    def _mark_dirty(self, scope: str, rel: str = "") -> None:
        with self._lock:
            self._dirty.add(scope)

    def _index(self, scope: str) -> str:
        """The injected index of a wiki, regenerated first if pages changed."""
        wiki = self._wiki(scope)
        assert wiki is not None
        with self._lock:
            if scope in self._dirty:
                wiki.rebuild_index()
                self._dirty.discard(scope)
        return wiki.injected_context()

    # -- model calls -----------------------------------------------------------

    def open(
        self,
        *,
        agent: str,
        stage: str,
        model: str,
        system: str | None = None,
        memory: MemoryPolicy | None = None,
        tools: Sequence[ToolSpec] = (),
        web_search: WebSearch | None = None,
        effort: Effort | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_turns: int = 60,
        label: str | None = None,
    ) -> HarnessConversation:
        """Open a conversation with memory, transcript and usage accounting attached."""
        client = self.client_for(model)
        policy = memory or MemoryPolicy.default()
        label = label or f"{stage}.{agent}"
        writes: list[tuple[str, str]] = []
        all_tools = list(tools)
        backend: WikiMemoryBackend | None = None

        if self.project is not None and policy.read:
            pull = client.capabilities.memory_tool
            lab_index = self._index("lab") if self.lab is not None else None
            block = prompts.memory_block(
                index=self._index("project"),
                lab_index=lab_index,
                pull=pull,
                write=policy.write if policy.lab_write == "none" else "all",
            )
            system = f"{system}\n\n{block}" if system else block
            if pull:

                def on_write(scope: str, rel: str) -> None:
                    writes.append((scope, rel))
                    self._mark_dirty(scope, rel)

                backend = WikiMemoryBackend(
                    self.project,
                    self.lab,
                    write=policy.write,
                    lab_write=policy.lab_write,
                    author=f"{stage}.{agent}",
                    on_write=on_write,
                )
                all_tools.append(wiki_search_tool(self.project, self.lab))
                if policy.transcripts and self.transcripts is not None:
                    all_tools.append(transcript_search_tool(self.transcripts))

        if all_tools and not client.capabilities.tools:
            raise LLMConfigError(
                f"model '{model}' ({client.provider}) cannot use tools, which {label} requires"
            )
        on_event = (
            self.transcripts.sink(stage=stage, agent=agent)
            if self.transcripts is not None
            else None
        )
        conversation = client.conversation(
            model=model,
            system=system,
            tools=all_tools,
            memory=backend,
            web_search=web_search if client.capabilities.web_search else None,
            effort=effort or self.default_effort,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            label=label,
            on_event=on_event,
        )
        return HarnessConversation(
            self,
            conversation,
            client=client,
            label=label,
            writes=writes,
            # A response format and search citations do not combine: go straight to text.
            text_only_schema=web_search is not None and client.capabilities.web_search,
        )

    def generate(
        self,
        parts: Part | Sequence[Part],
        *,
        agent: str,
        stage: str,
        model: str,
        system: str | None = None,
        parser: Parser | None = None,
        schema: Schema | None = None,
        require_parsed: bool = True,
        memory: MemoryPolicy | None = None,
        pinned: Sequence[str] = (),
        retrieve: str | None = None,
        k: int = 6,
        web_search: WebSearch | None = None,
        effort: Effort | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retries: int = 2,
        label: str | None = None,
    ) -> LLMResponse:
        """One-shot call.  A response that fails to parse is asked for again from scratch."""
        last_error: LLMParseError | None = None
        for _ in range(max(1, parse_retries + 1)):
            conversation = self.open(
                agent=agent,
                stage=stage,
                model=model,
                system=system,
                memory=memory,
                web_search=web_search,
                effort=effort,
                temperature=temperature,
                max_tokens=max_tokens,
                label=label,
            )
            try:
                return conversation.send(
                    parts,
                    parser=parser,
                    schema=schema,
                    require_parsed=require_parsed,
                    pinned=pinned,
                    retrieve=retrieve,
                    k=k,
                )
            except LLMParseError as exc:
                last_error = exc
                log.warning("%s: response did not parse; asking again", label or f"{stage}.{agent}")
        assert last_error is not None
        raise last_error

    # -- deterministic memory access for pipeline code -------------------------

    def context_block(
        self,
        *,
        pinned: Sequence[str] = (),
        query: str | None = None,
        k: int = 6,
        budget_chars: int = 60_000,
    ) -> str:
        """Render pinned pages plus the top ``k`` search hits for ``query`` as one block.

        Pages are included whole, in order, until the character budget is used;
        a page that would overflow is replaced by a pointer to its path.
        """
        if self.project is None:
            return ""
        chosen: list[str] = []
        for rel in pinned:
            if rel not in chosen and self.project.exists(rel):
                chosen.append(rel)
        if query:
            for hit in self.project.search(query, k=k, prefix=WIKI_DIR):
                if hit.path not in chosen:
                    chosen.append(hit.path)
        if not chosen:
            return ""
        sections: list[str] = []
        used = 0
        for rel in chosen:
            try:
                text = self.project.read_text(rel).strip()
            except WikiError:
                continue
            path = virtual_path(rel)
            if used + len(text) > budget_chars:
                sections.append(f'<page path="{path}" omitted="too long for this prompt"/>')
                continue
            used += len(text)
            sections.append(f'<page path="{path}">\n{text}\n</page>')
        return "<memory>\n" + "\n".join(sections) + "\n</memory>"

    def write_page(
        self,
        rel: str,
        body: str,
        meta: dict[str, Any] | None = None,
        *,
        scope: Literal["project", "lab"] = "project",
    ) -> None:
        """Write a wiki page from pipeline code (allowed to write the source pages)."""
        wiki = self._wiki(scope)
        if wiki is None:
            return
        if rel == INDEX_FILE:
            raise WikiError(f"{INDEX_FILE} is maintained by the wiki itself")
        wiki.write(rel, body, meta, author="pipeline", force=True)
        self._mark_dirty(scope, rel)

    def read_page(self, rel: str, *, scope: Literal["project", "lab"] = "project") -> str | None:
        """Body of a page, or ``None`` when memory is off or the page does not exist."""
        wiki = self._wiki(scope)
        if wiki is None or not wiki.exists(rel):
            return None
        return wiki.read(rel).body

    def note(
        self,
        text: str,
        *,
        stage: str,
        agent: str = "pipeline",
        kind: str = "observation",
        title: str | None = None,
    ) -> str | None:
        """Record a salient observation for later consolidation."""
        if self.project is None:
            return None
        rel = self.project.add_note(text, stage=stage, agent=agent, kind=kind, title=title)
        self.record("note", stage=stage, agent=agent, kind=kind, path=rel, text=text)
        return rel

    def record(self, type: str, *, stage: str = "", agent: str = "pipeline", **fields: Any) -> None:
        """Append a pipeline-level event (stage boundaries, compiles, scores) to the transcript."""
        if self.transcripts is None:
            return
        self.transcripts.record({"type": type, "stage": stage, "agent": agent, **fields})

    def refresh(self) -> None:
        """Repair links and regenerate the index of each wiki."""
        for scope in ("project", "lab"):
            wiki = self._wiki(scope)
            if wiki is None:
                continue
            wiki.repair_links()
            wiki.rebuild_index()
            with self._lock:
                self._dirty.discard(scope)

    # -- consolidation ---------------------------------------------------------

    def dream(
        self,
        *,
        stage: str,
        focus: str,
        model: str | None = None,
        build: bool = False,
        force: bool = False,
    ) -> DreamReport | None:
        """Consolidate the project memory at a stage boundary.

        The model only takes part when dreaming is set to ``stages`` (or
        ``force``); otherwise this is deterministic maintenance.  ``build``
        asks for a session even when no notes are pending, because ``focus``
        calls for new pages.
        """
        if self.project is None:
            return None
        from paper_orchestra.memory.dreaming import dream

        return dream(
            self,
            stage=stage,
            focus=focus,
            model=model or self.dream_model,
            scope="project",
            use_model=force or self.dream_mode == "stages",
            build=build,
        )

    def finish(self, *, model: str | None = None) -> list[DreamReport]:
        """End-of-run consolidation: the project memory, then promotion to the lab memory."""
        if self.project is None:
            return []
        from paper_orchestra.memory.dreaming import END_OF_RUN_FOCUS, LAB_FOCUS, dream

        use_model = self.dream_mode in ("stages", "end")
        model = model or self.dream_model
        reports = [
            dream(
                self,
                stage="finalize",
                focus=END_OF_RUN_FOCUS,
                model=model,
                scope="project",
                use_model=use_model,
            )
        ]
        if self.lab is not None:
            reports.append(
                dream(
                    self,
                    stage="finalize",
                    focus=LAB_FOCUS,
                    model=model,
                    scope="lab",
                    use_model=use_model,
                )
            )
        return reports
