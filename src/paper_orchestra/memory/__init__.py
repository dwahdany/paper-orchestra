"""Agent memory: a wiki behind a context harness."""

from paper_orchestra.memory.dreaming import DreamReport, dream
from paper_orchestra.memory.harness import ContextHarness, HarnessConversation, MemoryPolicy
from paper_orchestra.memory.tools import WikiMemoryBackend, transcript_search_tool, wiki_search_tool
from paper_orchestra.memory.transcript import TranscriptHit, TranscriptLog
from paper_orchestra.memory.wiki import LintReport, Page, PageInfo, SearchHit, Wiki, WikiError

__all__ = [
    "ContextHarness",
    "DreamReport",
    "HarnessConversation",
    "LintReport",
    "MemoryPolicy",
    "Page",
    "PageInfo",
    "SearchHit",
    "TranscriptHit",
    "TranscriptLog",
    "Wiki",
    "WikiError",
    "WikiMemoryBackend",
    "dream",
    "transcript_search_tool",
    "wiki_search_tool",
]
