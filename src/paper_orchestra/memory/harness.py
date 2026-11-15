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
from pathlib import Path
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
from paper_orchestra.memory.cache import KeyValueCache
from paper_orchestra.memory.tools import (
    WikiMemoryBackend,
    transcript_search_tool,
    virtual_path,
    wiki_search_tool,
)
from paper_orchestra.memory.transcript import TranscriptLog
from paper_orchestra.memory.wiki import INDEX_FILE, WIKI_DIR, Wiki, WikiError, link_for

if TYPE_CHECKING:
    from paper_orchestra.config import DreamMode, PipelineConfig
    from paper_orchestra.memory.dreaming import DreamReport, LeakScanner

log = logging.getLogger(__name__)

WriteScope = Literal["none", "notes", "all"]


class RunCancelled(RuntimeError):
    """The run was stopped (see :meth:`ContextHarness.cancel`).

    Deliberately not an ``LLMError``: nothing should retry it or treat it as a
    model failure.
    """


#: Sections of the project wiki that pipeline code owns.  They are records (inputs, verified
#: bibliography, review results), so no model session may edit them.
CODE_OWNED_SECTIONS = (
    "wiki/sources/",
    "wiki/literature/",
    "wiki/reviews/",
    "wiki/figures/",
    "wiki/paper/",
    "wiki/drafts/",
)

#: What a session that cannot open pages is given: the knowledge earlier sessions wrote
#: down (project and lab), and the lessons that are still waiting to be consolidated.
PUSH_PROJECT_SECTIONS = ("wiki/lessons", "wiki/synthesis")
PUSH_LAB_SECTIONS = ("wiki/lessons", "wiki/venues")
PUSH_MAX_NOTES = 12
PUSH_BUDGET_CHARS = 40_000

#: Seeded into a new project wiki: the facts every session should start from.
SEED_STANDING_NOTES = """\
- Numbers in the paper come only from the experimental log -> [[sources/experimental_log]]
- Method details come only from the idea document -> [[sources/idea]]
- The source pages outrank every other page and note in this memory."""


@dataclass(frozen=True)
class MemoryPolicy:
    """What a session may do with memory."""

    #: Inject the index and offer read/search access.
    read: bool = True
    write: WriteScope = "notes"
    lab_write: Literal["none", "all"] = "none"
    #: Offer ``transcript_search`` (consolidation sessions).
    transcripts: bool = False
    #: Show the lab memory beside the project memory.  Off for sessions whose output
    #: becomes material for the paper, which must come from this run alone.
    lab: bool = True

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
    def dream(cls) -> MemoryPolicy:
        """Consolidation: may edit pages and standing notes, and search transcripts."""
        return cls(write="all", transcripts=True, lab=False)


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
        reads: list[tuple[str, str]] | None = None,
        text_only_schema: bool = False,
        push: bool = False,
        push_lab: bool = False,
    ) -> None:
        self._harness = harness
        self._conversation = conversation
        self._client = client
        self._label = label
        self._first = True
        self._text_schema = text_only_schema
        self._recorded = Usage()
        #: Push mode: the session cannot open pages, so the harness hands it some.
        self._push = push
        self._push_lab = push_lab
        #: ``(scope, path)`` of every memory file this session wrote.
        self.writes = writes
        #: ``(scope, path)`` of every memory file this session opened.
        self.reads = reads if reads is not None else []
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
        """Send a user turn.  ``pinned``/``retrieve`` add memory pages to the first turn.

        A session that cannot open pages itself (push mode) is given the lessons
        and syntheses in memory when the caller names nothing.
        """
        self._harness._check_cancelled()
        content = normalize_parts(parts)
        if self._first:
            block = ""
            if pinned or retrieve:
                block = self._harness.context_block(pinned=pinned, query=retrieve, k=k)
            elif self._push:
                block = self._harness.default_push_block(lab=self._push_lab)
            if block:
                content = [Text(block), *content]
        self._first = False
        try:
            return self._send(content, parser, schema, require_parsed, max_tokens)
        finally:
            # Record what was spent even when the turn failed to parse or was refused.
            self._record_usage()

    def _record_usage(self) -> None:
        total = self._conversation.usage
        delta = Usage(
            input_tokens=total.input_tokens - self._recorded.input_tokens,
            output_tokens=total.output_tokens - self._recorded.output_tokens,
            cache_read_input_tokens=(
                total.cache_read_input_tokens - self._recorded.cache_read_input_tokens
            ),
            cache_creation_input_tokens=(
                total.cache_creation_input_tokens - self._recorded.cache_creation_input_tokens
            ),
            requests=total.requests - self._recorded.requests,
            web_searches=total.web_searches - self._recorded.web_searches,
        )
        if delta.requests or delta.input_tokens or delta.output_tokens:
            self._harness.usage.record(self._label, self.model, delta)
            self._recorded.add(delta)

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
        consolidate_lab: bool = True,
        cache: KeyValueCache | None = None,
        protected: Sequence[Path] = (),
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
        #: When False, finish() only adds this run's lessons to the lab memory as notes and
        #: leaves folding them into pages to a separate, single consolidation (used where
        #: several runs share the lab memory on storage without file locking).
        self.consolidate_lab = consolidate_lab
        #: Data kept across runs for pipeline code only (never shown to a model).
        self.cache = cache
        #: Files and directories that hold this paper's content, beside the wiki pages.
        self._protected: list[Path] = [Path(path) for path in protected]
        self._scanner: LeakScanner | None = None
        self._scanner_signature: tuple[Any, ...] | None = None
        self._lock = threading.RLock()
        self._dirty: set[str] = set()
        self._cancelled = threading.Event()

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
            "consolidate_lab": cfg.memory.lab_consolidate,
        }
        if not cfg.memory.enabled:
            return cls(**common)
        project = Wiki(
            cfg.project_memory_dir,
            name="project",
            description="What this run has learned about the paper being written.",
            read_only=CODE_OWNED_SECTIONS,
        )
        if not project.standing_notes():
            project.set_standing_notes(SEED_STANDING_NOTES)
        lab_dir = cfg.memory.resolved_lab_dir()
        lab = (
            Wiki(
                lab_dir,
                name="lab",
                description="Process lessons shared across papers. Never paper content.",
                read_only=(),
                # Both indexes are injected into the same prompt, so the shared one is kept
                # much smaller.
                index_max_lines=80,
                index_max_bytes=8_000,
                standing_max_lines=15,
            )
            if lab_dir is not None
            else None
        )
        transcripts = TranscriptLog(project.root / "transcripts")
        from paper_orchestra.context import RunPaths

        paths = RunPaths(cfg.base_dir, cfg.idea_filename, cfg.experimental_log_filename)
        return cls(
            project=project,
            lab=lab,
            transcripts=transcripts,
            # Beside the wiki, not in it: no memory tool, search or index reaches it.
            cache=KeyValueCache(lab_dir / "cache") if lab_dir is not None else None,
            protected=(
                paths.materials_dir,
                paths.latex_dir,
                paths.refinement_dir,
                paths.outline,
                paths.plotting_results,
                paths.final_tex,
            ),
            **common,
        )

    # -- basics ----------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.project is not None

    def client_for(self, model: str) -> LLMClient:
        return self._client_for(model)

    def cancel(self) -> None:
        """Stop the run at the next model call.

        After this, opening a conversation or sending a turn raises
        :class:`RunCancelled`.  A call that is already in flight finishes; work on
        other threads then stops at its next call instead of running to the end.
        """
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def _check_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise RunCancelled("the run was stopped")

    def _wiki(self, scope: str) -> Wiki | None:
        return self.lab if scope == "lab" else self.project

    def _mark_dirty(self, scope: str, rel: str = "") -> None:
        with self._lock:
            self._dirty.add(scope)

    def _index(self, scope: str) -> str:
        """The injected index of a wiki, regenerated first if pages changed.

        Only the project index is regenerated here.  The lab memory is shared
        between runs, possibly on storage where the last writer wins, so a run
        reads its index as it is and only ever adds notes; the index is rewritten
        by the lab consolidation alone, under its lock.
        """
        wiki = self._wiki(scope)
        assert wiki is not None
        with self._lock:
            if scope == "project" and scope in self._dirty:
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
        memory_root: Literal["project", "lab"] = "project",
    ) -> HarnessConversation:
        """Open a conversation with memory, transcript and usage accounting attached.

        ``memory_root="lab"`` makes the lab wiki the session's whole memory (no project
        pages, no transcripts): used to consolidate the shared memory without exposing any
        paper content to that session.
        """
        self._check_cancelled()
        client = self.client_for(model)
        policy = memory or MemoryPolicy.default()
        label = label or f"{stage}.{agent}"
        writes: list[tuple[str, str]] = []
        all_tools = list(tools)
        backend: WikiMemoryBackend | None = None
        reads: list[tuple[str, str]] = []
        lab_only = memory_root == "lab"
        root_wiki = self.lab if lab_only else self.project
        scope_name = "lab" if lab_only else "project"
        push = False
        mounted_lab: Wiki | None = None

        if self.project is not None and root_wiki is not None and policy.read:
            pull = client.capabilities.memory_tool
            push = not pull and not lab_only
            mounted_lab = None if lab_only or not policy.lab else self.lab
            if lab_only:
                block = prompts.lab_memory_block(self._index("lab"))
            else:
                block = prompts.memory_block(
                    index=self._index("project"),
                    lab_index=self._index("lab") if mounted_lab is not None else None,
                    pull=pull,
                    write=policy.write if policy.lab_write == "none" else "all",
                )
            system = f"{system}\n\n{block}" if system else block
            if pull:

                def on_write(scope: str, rel: str) -> None:
                    scope = scope_name if lab_only else scope
                    writes.append((scope, rel))
                    self._mark_dirty(scope, rel)

                def on_read(scope: str, rel: str) -> None:
                    scope = scope_name if lab_only else scope
                    reads.append((scope, rel))
                    self.record("memory_read", stage=stage, agent=agent, scope=scope, path=rel)

                backend = WikiMemoryBackend(
                    root_wiki,
                    mounted_lab,
                    write=policy.write,
                    lab_write=policy.lab_write,
                    author=f"{stage}.{agent}",
                    on_write=on_write,
                    on_read=on_read,
                    guard=self._lab_guard if lab_only else None,
                )
                all_tools.append(wiki_search_tool(root_wiki, mounted_lab))
                if policy.transcripts and self.transcripts is not None and not lab_only:
                    # A session that checks claims must not find its own questions.
                    all_tools.append(
                        transcript_search_tool(self.transcripts, exclude_agents=(agent,))
                    )

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
            reads=reads,
            # A response format and search citations do not combine: go straight to text.
            text_only_schema=web_search is not None and client.capabilities.web_search,
            push=push,
            push_lab=mounted_lab is not None,
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
        lab_pinned: Sequence[str] = (),
    ) -> str:
        """Render pinned pages plus the top ``k`` search hits for ``query`` as one block.

        Pages are included whole, in order, until the character budget is used;
        a page that would overflow is replaced by a pointer to its path.
        ``lab_pinned`` names pages of the lab memory to include after them.
        """
        if self.project is None:
            return ""
        chosen: list[tuple[Wiki, str, bool]] = []
        seen: set[str] = set()
        for rel in pinned:
            if rel not in seen and self.project.exists(rel):
                seen.add(rel)
                chosen.append((self.project, rel, False))
        if query:
            for hit in self.project.search(query, k=k, prefix=WIKI_DIR):
                if hit.path not in seen:
                    seen.add(hit.path)
                    chosen.append((self.project, hit.path, False))
        if self.lab is not None:
            chosen.extend((self.lab, rel, True) for rel in lab_pinned if self.lab.exists(rel))
        if not chosen:
            return ""
        sections: list[str] = []
        used = 0
        for wiki, rel, is_lab in chosen:
            try:
                text = wiki.read_text(rel).strip()
            except WikiError:
                continue
            path = virtual_path(rel, lab=is_lab)
            if used + len(text) > budget_chars:
                sections.append(f'<page path="{path}" omitted="too long for this prompt"/>')
                continue
            used += len(text)
            sections.append(f'<page path="{path}">\n{text}\n</page>')
        return "<memory>\n" + "\n".join(sections) + "\n</memory>"

    def default_push_block(self, *, lab: bool = True) -> str:
        """What a session that cannot open pages is given when its caller names nothing.

        The lessons and syntheses that consolidation wrote (newest first), the
        lesson notes still waiting to be consolidated, and, with ``lab``, the
        lab memory's lessons and venue rules.  The inputs are not repeated: the
        agents already put them in their prompts.
        """
        if self.project is None:
            return ""
        pages = [
            info
            for section in PUSH_PROJECT_SECTIONS
            for info in self.project.list(section)
            if not self.project.read(info.path).meta.get("auto")
        ]
        pages.sort(key=lambda info: info.updated, reverse=True)
        notes = [
            info.path for info in reversed(self.project.pending_notes()) if info.kind == "lesson"
        ][:PUSH_MAX_NOTES]
        lab_pages: list[str] = []
        if lab and self.lab is not None:
            lab_pages = [
                info.path
                for section in PUSH_LAB_SECTIONS
                for info in self.lab.list(section)
                if not self.lab.read(info.path).meta.get("auto")
            ]
        return self.context_block(
            pinned=[*(info.path for info in pages), *notes],
            lab_pinned=lab_pages,
            budget_chars=PUSH_BUDGET_CHARS,
        )

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
        limit = wiki.max_page_bytes - 4_000  # room for front matter
        if len(body.encode("utf-8")) <= limit:
            wiki.write(rel, body, meta, author="pipeline", force=True)
        else:
            self._write_in_parts(wiki, rel, body, dict(meta or {}), limit)
        self._mark_dirty(scope, rel)

    @staticmethod
    def _write_in_parts(wiki: Wiki, rel: str, body: str, meta: dict[str, Any], limit: int) -> None:
        """Store a page that exceeds the page limit as linked parts, split on line breaks."""
        chunks: list[str] = []
        current: list[str] = []
        size = 0
        for line in body.splitlines(keepends=True):
            encoded = len(line.encode("utf-8"))
            if current and size + encoded > limit:
                chunks.append("".join(current))
                current, size = [], 0
            current.append(line)
            size += encoded
        if current:
            chunks.append("".join(current))
        stem = rel[:-3]
        links = []
        for number, chunk in enumerate(chunks, 1):
            part_rel = f"{stem}-part-{number}.md"
            part_meta = {
                **meta,
                "summary": f"{meta.get('summary') or stem} (part {number} of {len(chunks)})",
            }
            wiki.write(part_rel, chunk, part_meta, author="pipeline", force=True)
            links.append(f"- [[{link_for(part_rel)}]]")
        wiki.write(
            rel,
            f"This page is stored in {len(chunks)} parts because of its size:\n\n"
            + "\n".join(links),
            meta,
            author="pipeline",
            force=True,
        )

    def delete_page(self, rel: str, *, scope: Literal["project", "lab"] = "project") -> bool:
        """Remove a page from pipeline code; returns whether it existed."""
        wiki = self._wiki(scope)
        if wiki is None or not wiki.exists(rel):
            return False
        wiki.delete(rel, force=True)
        self._mark_dirty(scope, rel)
        return True

    def list_pages(
        self, prefix: str = WIKI_DIR, *, scope: Literal["project", "lab"] = "project"
    ) -> list[str]:
        """Paths of the pages under ``prefix`` (empty when memory is off)."""
        wiki = self._wiki(scope)
        return [info.path for info in wiki.list(prefix)] if wiki is not None else []

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
        """Repair links and regenerate the index of the project wiki.

        The lab wiki is left alone: it is maintained only by its consolidation
        (see :meth:`_index`).
        """
        if self.project is None:
            return
        self.project.repair_links()
        self.project.rebuild_index()
        with self._lock:
            self._dirty.discard("project")

    def protect(self, *paths: str | Path) -> None:
        """Declare files or directories whose text is this paper's content.

        Whatever they contain is kept out of the lab memory by the leak scan,
        in addition to the content pages of the project wiki.
        """
        with self._lock:
            for path in paths:
                if Path(path) not in self._protected:
                    self._protected.append(Path(path))
            self._scanner = None

    def leak_scanner(self) -> LeakScanner:
        """The scanner for text headed to the lab memory, rebuilt when the materials change."""
        from paper_orchestra.memory.dreaming import (
            CONTENT_SECTIONS,
            LeakScanner,
            content_sources,
            protected_files,
            venue_materials,
        )

        if self.project is None:
            return LeakScanner(())
        with self._lock:
            entries: list[tuple[Any, ...]] = [
                (info.path, info.size, info.updated)
                for section in CONTENT_SECTIONS
                for info in self.project.list(section.rstrip("/"))
            ]
            for file in protected_files(self._protected):
                try:
                    stat = file.stat()
                except OSError:
                    continue
                entries.append((str(file), stat.st_size, stat.st_mtime_ns))
            signature = tuple(entries)
            if self._scanner is None or signature != self._scanner_signature:
                self._scanner = LeakScanner(
                    content_sources(self.project, self._protected),
                    venue_materials(self.project, self._protected),
                )
                self._scanner_signature = signature
            return self._scanner

    def consolidate_lab_memory(self, *, model: str | None = None) -> DreamReport | None:
        """Fold pending lab notes into lab pages (the second half of promotion)."""
        if self.project is None or self.lab is None:
            return None
        from paper_orchestra.memory.dreaming import LAB_FOCUS, dream

        return dream(
            self,
            stage="finalize",
            focus=LAB_FOCUS,
            model=model or self.dream_model,
            scope="lab",
            use_model=self.dream_mode in ("stages", "end"),
        )

    def _lab_guard(self, scope: str, text: str) -> str | None:
        """Veto text headed for the shared memory if it carries this paper's content."""
        if self.project is None:
            return None
        return self.leak_scanner().reason(text)

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
        from paper_orchestra.memory.dreaming import (
            END_OF_RUN_FOCUS,
            LAB_FOCUS,
            dream,
            promote_lessons,
        )

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
            # Lessons reach the shared memory as notes that passed a leak scan; the lab
            # session then works from those notes alone and never sees this paper.
            promote_lessons(self)
        if self.lab is not None and self.consolidate_lab:
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
