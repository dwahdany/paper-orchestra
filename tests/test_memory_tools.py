from __future__ import annotations

import pytest

from paper_orchestra.llm.base import MemoryBackend, ToolError
from paper_orchestra.memory.tools import (
    MAX_VIEW_CHARS,
    WikiMemoryBackend,
    transcript_search_tool,
    wiki_search_tool,
)
from paper_orchestra.memory.transcript import TranscriptLog
from paper_orchestra.memory.wiki import Wiki


@pytest.fixture
def project(tmp_path) -> Wiki:
    wiki = Wiki(tmp_path / "project", name="project")
    wiki.write("wiki/sources/idea.md", "We propose FooNet.", {"summary": "idea"}, force=True)
    wiki.write(
        "wiki/literature/Vaswani2017Attention.md",
        "line one\nline two\nline three",
        {"title": "Attention Is All You Need", "summary": "Transformer"},
    )
    wiki.add_note("pdflatex choked on cleverref", stage="section_writing", kind="lesson")
    wiki.rebuild_index()
    return wiki


@pytest.fixture
def lab(tmp_path) -> Wiki:
    wiki = Wiki(tmp_path / "lab", name="lab", read_only=())
    wiki.write("wiki/lessons/latex.md", "Use cleveref, not cleverref.", {"summary": "LaTeX"})
    wiki.rebuild_index()
    return wiki


def test_backend_satisfies_the_provider_protocol(project):
    assert isinstance(WikiMemoryBackend(project), MemoryBackend)


def test_view_root_lists_two_levels_and_hides_transcripts(project, lab, tmp_path):
    (project.root / "transcripts" / "s.jsonl").write_text("{}\n")
    listing = WikiMemoryBackend(project, lab).view("/memories")
    lines = listing.splitlines()
    assert lines[0].startswith("Here're the files and directories up to 2 levels deep in /memories")
    paths = [line.split("\t")[1] for line in lines[1:]]
    assert "/memories/MEMORY.md" in paths
    assert "/memories/wiki/literature" in paths and "/memories/wiki/sources" in paths
    assert "/memories/lab" in paths and "/memories/lab/MEMORY.md" in paths
    assert not any("transcripts" in p for p in paths)
    # Two levels only: files inside wiki/literature are not listed from the root.
    assert "/memories/wiki/literature/Vaswani2017Attention.md" not in paths
    deeper = WikiMemoryBackend(project, lab).view("/memories/wiki")
    assert "/memories/wiki/literature/Vaswani2017Attention.md" in deeper


def test_view_file_numbers_lines_and_honours_ranges(project):
    backend = WikiMemoryBackend(project)
    text = backend.view("/memories/wiki/literature/Vaswani2017Attention.md")
    assert text.startswith(
        "Here's the content of /memories/wiki/literature/Vaswani2017Attention.md with line numbers:"
    )
    assert "     1\t---" in text
    body_line = next(line for line in text.splitlines() if line.endswith("line two"))
    number = int(body_line.split("\t")[0])
    ranged = backend.view("/memories/wiki/literature/Vaswani2017Attention.md", (number, number))
    assert ranged.splitlines()[1:] == [f"{number:>6}\tline two"]
    to_end = backend.view("/memories/wiki/literature/Vaswani2017Attention.md", (number, -1))
    assert to_end.splitlines()[-1].endswith("line three") or to_end.splitlines()[-1].endswith("\t")
    with pytest.raises(ToolError, match="Invalid view_range"):
        backend.view("/memories/wiki/literature/Vaswani2017Attention.md", (999, 1000))
    with pytest.raises(ToolError, match="two integers"):
        backend.view("/memories/wiki/literature/Vaswani2017Attention.md", ("a", None))


def test_view_truncates_long_pages_with_a_continuation_hint(project):
    project.write("wiki/big.md", "\n".join(f"row {i} " + "x" * 80 for i in range(600)))
    text = WikiMemoryBackend(project).view("/memories/wiki/big.md")
    assert len(text) < MAX_VIEW_CHARS + 400
    assert "Use view_range [" in text


@pytest.mark.parametrize(
    "path",
    [
        "wiki/x.md",
        "/etc/passwd",
        "/memories/../etc/passwd",
        "/memories/wiki/../../x.md",
        "/memories/%2e%2e/x.md",
        "/memories/wiki\\x.md",
        "/memories/transcripts/s.jsonl",
        "/memories/.dream.lock",
        "/memories/wiki/nope.md",
        "/memories/lab/wiki/lessons/latex.md",  # no lab mounted
        None,
        "",
    ],
)
def test_bad_paths_are_tool_errors(project, path):
    with pytest.raises(ToolError):
        WikiMemoryBackend(project).view(path)


def test_notes_scope_allows_only_new_notes(project):
    backend = WikiMemoryBackend(project, write="notes")
    writes: list[tuple[str, str]] = []
    backend._on_write = lambda scope, rel: writes.append((scope, rel))
    assert backend.create("/memories/notes/compile-fix.md", "Lesson: use cleveref.").startswith(
        "File created successfully"
    )
    assert writes == [("project", "notes/compile-fix.md")]
    for operation in (
        lambda: backend.create("/memories/wiki/lessons/x.md", "text"),
        lambda: backend.create("/memories/MEMORY.md", "text"),
        lambda: backend.str_replace(
            "/memories/wiki/literature/Vaswani2017Attention.md", "one", "1"
        ),
        lambda: backend.delete("/memories/wiki/literature/Vaswani2017Attention.md"),
        lambda: backend.create("/memories/notes/consolidated/d/x.md", "text"),
        lambda: backend.create("/memories/notes/x.txt", "text"),
        lambda: backend.rename("/memories/notes/compile-fix.md", "/memories/wiki/x.md"),
    ):
        with pytest.raises(ToolError):
            operation()
    assert project.read("wiki/literature/Vaswani2017Attention.md").body.startswith("line one")


def test_read_only_scope_rejects_every_mutation(project):
    backend = WikiMemoryBackend(project, write="none")
    with pytest.raises(ToolError, match="read-only"):
        backend.create("/memories/notes/x.md", "text")
    assert "idea" in backend.view("/memories/wiki/sources/idea.md")


def test_full_scope_edits_pages_but_never_sources(project):
    backend = WikiMemoryBackend(project, write="all")
    page = "/memories/wiki/literature/Vaswani2017Attention.md"
    assert backend.str_replace(page, "line two", "LINE 2") == "The memory file has been edited."
    assert "LINE 2" in project.read_text("wiki/literature/Vaswani2017Attention.md")
    with pytest.raises(ToolError, match="did not appear"):
        backend.str_replace(page, "absent", "x")
    project.write("wiki/dup.md", "same same")
    with pytest.raises(ToolError, match="Multiple occurrences"):
        backend.str_replace("/memories/wiki/dup.md", "same", "x")
    # new_str may arrive missing/None from an unvalidated tool input: treated as deletion.
    backend.str_replace("/memories/wiki/dup.md", "same same", None)

    backend.create("/memories/wiki/lessons/latex.md", "a\nc")
    stored = project.read_text("wiki/lessons/latex.md").split("\n")
    after_a = stored.index("a") + 1
    assert backend.insert("/memories/wiki/lessons/latex.md", after_a, "b").endswith(
        "has been edited."
    )
    assert "a\nb\nc" in project.read_text("wiki/lessons/latex.md")
    # Free text placed inside the front matter block would be lost on save: refuse it.
    with pytest.raises(ToolError, match="front matter"):
        backend.insert("/memories/wiki/lessons/latex.md", 1, "stray prose")
    backend.insert("/memories/wiki/lessons/latex.md", 1, "summary: LaTeX lessons")
    assert project.read("wiki/lessons/latex.md").meta["summary"] == "LaTeX lessons"
    with pytest.raises(ToolError, match="insert_line"):
        backend.insert("/memories/wiki/lessons/latex.md", 99, "z")
    with pytest.raises(ToolError, match="integer"):
        backend.insert("/memories/wiki/lessons/latex.md", "1", "z")

    backend.rename("/memories/wiki/lessons/latex.md", "/memories/wiki/lessons/tex.md")
    assert project.exists("wiki/lessons/tex.md") and not project.exists("wiki/lessons/latex.md")
    with pytest.raises(ToolError, match="already exists"):
        backend.rename("/memories/wiki/dup.md", "/memories/wiki/lessons/tex.md")
    assert backend.delete("/memories/wiki/lessons/tex.md").startswith("Successfully deleted")

    for operation in (
        lambda: backend.create("/memories/wiki/sources/idea.md", "tampered"),
        lambda: backend.str_replace("/memories/wiki/sources/idea.md", "FooNet", "BarNet"),
        lambda: backend.delete("/memories/wiki/sources/idea.md"),
        lambda: backend.delete("/memories"),
        lambda: backend.delete("/memories/MEMORY.md"),
    ):
        with pytest.raises(ToolError):
            operation()
    assert "FooNet" in project.read_text("wiki/sources/idea.md")


def test_standing_notes_edit_survives_index_rebuild(project):
    backend = WikiMemoryBackend(project, write="all")
    text = project.read_text("MEMORY.md")
    backend.str_replace("/memories/MEMORY.md", "_None yet._", "- Use cleveref.")
    assert "- Use cleveref." in project.rebuild_index()
    assert text != project.read_text("MEMORY.md")


def test_lab_is_separate_and_read_only_unless_granted(project, lab):
    reader = WikiMemoryBackend(project, lab, write="all")
    assert "cleveref" in reader.view("/memories/lab/wiki/lessons/latex.md")
    with pytest.raises(ToolError, match="lab memory is read-only"):
        reader.create("/memories/lab/wiki/lessons/new.md", "x")
    writer = WikiMemoryBackend(project, lab, write="none", lab_write="all")
    writer.create("/memories/lab/wiki/lessons/new.md", "Lesson")
    assert lab.exists("wiki/lessons/new.md")
    with pytest.raises(ToolError, match="read-only"):
        writer.create("/memories/notes/x.md", "x")
    with pytest.raises(ToolError, match="between"):
        WikiMemoryBackend(project, lab, write="all", lab_write="all").rename(
            "/memories/wiki/literature/Vaswani2017Attention.md", "/memories/lab/wiki/x.md"
        )


def test_page_size_limit_surfaces_as_tool_error(tmp_path):
    wiki = Wiki(tmp_path, max_page_bytes=200)
    with pytest.raises(ToolError, match="Split it"):
        WikiMemoryBackend(wiki, write="all").create("/memories/wiki/big.md", "x" * 1000)


def test_wiki_search_tool(project, lab):
    tool = wiki_search_tool(project, lab)
    assert tool.name == "wiki_search" and tool.input_schema["required"] == ["query"]
    result = tool.handler({"query": "transformer attention"})
    assert "/memories/wiki/literature/Vaswani2017Attention.md: Attention Is All You Need" in result
    assert "/memories/lab/wiki/lessons/latex.md" in tool.handler(
        {"query": "cleveref", "scope": "lab"}
    )
    assert "notes/" in tool.handler({"query": "cleverref pdflatex", "scope": "project"})
    assert tool.handler({"query": "attention", "section": "sources"}) == "No matching pages."
    assert tool.handler({"query": "zzzzzz"}) == "No matching pages."
    for bad in (
        {},
        {"query": ""},
        {"query": "x", "scope": "elsewhere"},
        {"query": "x", "k": "many"},
    ):
        with pytest.raises(ToolError):
            tool.handler(bad)
    assert "scope" not in wiki_search_tool(project).input_schema["properties"]


def test_transcript_search_tool(tmp_path):
    log = TranscriptLog(tmp_path, session_id="s1")
    log.record(
        {"type": "compile", "stage": "section_writing", "errors": ["Undefined control sequence"]}
    )
    tool = transcript_search_tool(log)
    result = tool.handler({"query": "undefined control sequence"})
    assert "section_writing" in result and "compile" in result and "seq=1" in result
    assert (
        tool.handler({"query": "undefined", "stage": "refinement"})
        == "No matching transcript events."
    )
    with pytest.raises(ToolError):
        tool.handler({"query": None})
