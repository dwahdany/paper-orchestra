from __future__ import annotations

import json
import os

import pytest

from paper_orchestra.llm.base import MemoryBackend, ToolError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, MemoryPolicy, TranscriptLog
from paper_orchestra.research.access import (
    ProjectMemoryBackend,
    propose_edit_tool,
    search_tool,
)
from paper_orchestra.research.model import NotAllowed, ProjectError
from paper_orchestra.research.project import Project

BRIEF = "Does learning-rate warmup reduce the variance of early updates?"
HYPOTHESIS = """---
title: Warmup reduces variance
summary: Linear warmup lowers the variance of early updates
---
## Statement

Warmup lowers the variance of the first thousand updates.
"""


@pytest.fixture
def project(tmp_path) -> Project:
    attached = tmp_path / "attached"
    attached.mkdir()
    (attached / "notes.txt").write_text("line one\nline two\nline three\n")
    (attached / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    return Project.create(
        tmp_path / "p",
        title="Warmup and variance",
        brief=BRIEF,
        files=[attached / "notes.txt", attached / "figure.png"],
    )


@pytest.fixture
def theorist(project) -> ProjectMemoryBackend:
    return ProjectMemoryBackend(project, "theorist", 1, "s-theorist-1")


def backend(project, role: str, session: str | None = None, round: int = 1) -> ProjectMemoryBackend:
    return ProjectMemoryBackend(project, role, round, session or f"s-{role}")


def listed(text: str) -> list[str]:
    return [line.split("\t")[1] for line in text.splitlines()[1:] if "\t" in line]


def shown(text: str) -> str:
    """The content of a file view without the header and the line numbers."""
    return "\n".join(line.split("\t", 1)[1] for line in text.splitlines()[1:])


def test_the_backend_is_a_memory_backend(theorist):
    assert isinstance(theorist, MemoryBackend)
    assert (theorist.author, theorist.role, theorist.round) == ("agent:theorist", "theorist", 1)
    assert ProjectMemoryBackend(theorist.project, "agent:scout", 2, "s").role == "scout"
    with pytest.raises(ProjectError, match="session id"):
        ProjectMemoryBackend(theorist.project, "scout", 1, " ")
    with pytest.raises(ProjectError, match="not an author"):
        ProjectMemoryBackend(theorist.project, "Not A Role", 1, "s")


# -- reading --------------------------------------------------------------------


def test_the_root_shows_the_wiki_and_the_inputs_and_nothing_else(project, theorist):
    project.add_request("Look at small batches")
    (project.root / "transcripts" / "s.jsonl").write_text("{}\n")
    project.wiki.add_note("a working note", stage="research")
    project.review(project.create_page("topic", "T", "x", author="agent:scout").path, "accept")
    text = theorist.view("/memories")
    assert text.startswith("Here're the files and directories up to 2 levels deep in /memories")
    assert listed(text) == [
        "/memories",
        "/memories/MEMORY.md",
        "/memories/inputs",
        "/memories/inputs/brief.md",
        "/memories/inputs/figure.png",
        "/memories/inputs/manifest.json",
        "/memories/inputs/notes.txt",
        "/memories/wiki",
        "/memories/wiki/idea.md",
        "/memories/wiki/topics",
    ]
    assert listed(theorist.view("/memories/wiki")) == [
        "/memories/wiki",
        "/memories/wiki/idea.md",
        "/memories/wiki/topics",
        "/memories/wiki/topics/t.md",
    ]
    # A section exists before anything is in it; a made-up one does not.
    assert listed(theorist.view("/memories/wiki/hypotheses")) == ["/memories/wiki/hypotheses"]
    with pytest.raises(ToolError, match="does not exist"):
        theorist.view("/memories/wiki/essays")


@pytest.mark.parametrize(
    "path",
    [
        "/memories/transcripts",
        "/memories/transcripts/s.jsonl",
        "/memories/notes",
        "/memories/proposals",
        "/memories/history",
        "/memories/requests",
        "/memories/experiments",
        "/memories/project.json",
        "/memories/state.json",
        "/memories/events.jsonl",
        "/memories/lab",
        "/memories/inputs/nothing-here",
        "/memories/wiki/idea.md/x",
    ],
)
def test_what_is_not_an_agents_to_read_does_not_exist(project, theorist, path):
    (project.root / "transcripts" / "s.jsonl").write_text("{}\n")
    project.propose_edit("wiki/idea.md", author="agent:analyst", summary="s", body="b")
    with pytest.raises(ToolError, match="does not exist"):
        theorist.view(path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        None,
        "wiki/idea.md",
        "/etc/passwd",
        "/memories/../project.json",
        "/memories/wiki/../../project.json",
        "/memories/wiki/hypotheses/../../../outside.md",
        "/memories/wiki/.hidden.md",
        "/memories/.project.lock",
        "/memories\\wiki\\idea.md",
        "/memories/wiki/%2e%2e/project.json",
        "/memories/wiki/idea.md\x00.txt",
        "/memories/wiki/ idea.md",
        "/memories/wiki /idea.md",
        "//memories/wiki/idea.md",
        "/memoriesx/wiki/idea.md",
    ],
)
def test_paths_cannot_leave_the_memory_directory(project, theorist, path):
    before = sorted(p.as_posix() for p in project.root.rglob("*") if p.is_file())
    for call in (
        lambda: theorist.view(path),
        lambda: theorist.create(path, HYPOTHESIS),
        lambda: theorist.str_replace(path, "a", "b"),
        lambda: theorist.insert(path, 0, "x"),
        lambda: theorist.delete(path),
        lambda: theorist.rename(path, "/memories/wiki/hypotheses/x.md"),
        lambda: theorist.rename("/memories/wiki/idea.md", path),
    ):
        with pytest.raises(ToolError):
            call()
    assert sorted(p.as_posix() for p in project.root.rglob("*") if p.is_file()) == before


def test_a_link_in_the_wiki_or_the_inputs_is_not_followed(project, theorist):
    secret = project.root / "transcripts" / "secret.md"
    secret.write_text("---\ntitle: Secret\n---\nwhat another session said\n")
    os.symlink(secret, project.root / "wiki" / "topics-link.md")
    (project.root / "wiki" / "topics").mkdir()
    os.symlink(secret, project.root / "wiki" / "topics" / "link.md")
    os.symlink(secret, project.inputs_dir / "link.md")
    os.symlink(project.root / "transcripts", project.inputs_dir / "more")
    for path in (
        "/memories/wiki/topics-link.md",
        "/memories/wiki/topics/link.md",
        "/memories/inputs/link.md",
        "/memories/inputs/more/secret.md",
    ):
        with pytest.raises(ToolError, match="does not exist"):
            theorist.view(path)
    assert not any(
        "link" in entry or "more" in entry for entry in listed(theorist.view("/memories"))
    )
    assert not any("link" in entry for entry in listed(theorist.view("/memories/wiki")))


def test_a_page_is_shown_with_line_numbers_and_ranges(project, theorist):
    text = theorist.view("/memories/wiki/idea.md")
    assert text.startswith("Here's the content of /memories/wiki/idea.md with line numbers:")
    assert "     1\t---" in text and f"\t{BRIEF}\n" in text
    number = next(
        int(line.split("\t")[0]) for line in text.splitlines() if line.endswith("kind: idea")
    )
    assert theorist.view("/memories/wiki/idea.md", (number, number)).splitlines()[1:] == [
        f"{number:>6}\tkind: idea"
    ]
    with pytest.raises(ToolError, match="Invalid view_range"):
        theorist.view("/memories/wiki/idea.md", (999, 1000))
    with pytest.raises(ToolError, match="two integers"):
        theorist.view("/memories/wiki/idea.md", ("a", None))
    project.update_page(
        "wiki/idea.md", author="human", body="\n".join(f"row {i} " + "x" * 90 for i in range(600))
    )
    long = theorist.view("/memories/wiki/idea.md")
    assert "[Truncated at line" in long and "Use view_range" in long and len(long) < 17_000


def test_a_guessed_path_is_answered_with_the_real_one(project, theorist):
    project.review(
        project.create_page("hypothesis", "Warmup", "x", author="agent:analyst").path, "accept"
    )
    with pytest.raises(ToolError, match=r"mean is at /memories/wiki/hypotheses/H1-warmup\.md"):
        theorist.view("/memories/wiki/hypotheses/H1.md")
    with pytest.raises(ToolError, match=r"does not exist. Please provide a valid path"):
        theorist.view("/memories/wiki/hypotheses/H2.md")


def test_what_each_session_sees_of_proposals_and_rejections(project):
    scout_page = project.create_page(
        "topic", "The scout's draft", "Unreviewed synthesis.", author="agent:scout"
    )
    turned_down = project.review(
        project.create_page(
            "hypothesis",
            "A vague hunch",
            "The secret reasoning nobody should build on.",
            author="agent:theorist",
            summary="It is all connected",
        ).path,
        "reject",
        reason="Not falsifiable",
    )
    earlier = project.create_page(
        "hypothesis", "From last round", "Pending.", author="agent:theorist"
    )
    theorist, scout = backend(project, "theorist"), backend(project, "scout")
    theorist.create("/memories/wiki/hypotheses/mine.md", "# Written in this session\n\nNew.")

    # An unreviewed proposal exists only for the session that wrote it: for anyone else
    # it is not there by path, not in a listing, not in the index.
    for session, page in ((theorist, scout_page), (scout, earlier), (theorist, earlier)):
        with pytest.raises(ToolError, match="does not exist"):
            session.view(f"/memories/{page.path}")
        assert page.title not in session.view("/memories/MEMORY.md")
        assert f"/memories/{page.path}" not in listed(session.view("/memories/wiki"))
        assert f"/memories/{page.path}" not in listed(
            session.view(f"/memories/{page.path.rsplit('/', 1)[0]}")
        )
    assert "/memories/wiki/topics" not in listed(theorist.view("/memories"))
    # What it wrote itself it reads in full and finds in its index.
    assert "New." in theorist.view("/memories/wiki/hypotheses/H3-mine.md")
    index = shown(theorist.view("/memories/MEMORY.md"))
    assert index == project.index_text(for_role="theorist", session="s-theorist")
    assert "[H3, proposed in this session, open] Written in this session" in index
    assert "you can create pages under: hypotheses/, questions/" in index
    # A session that is resumed under the same id finds its proposals again.
    resumed = backend(project, "theorist")
    assert "New." in resumed.view("/memories/wiki/hypotheses/H3-mine.md")

    # What the person turned down: title and reason, for everyone, and nothing else.
    for session in (theorist, scout):
        stub = shown(session.view(f"/memories/{turned_down.path}"))
        assert "title: A vague hunch" in stub and "reason: Not falsifiable" in stub
        assert "id: H1" in stub and "rejected by the person" in stub
        assert "secret reasoning" not in stub and "all connected" not in stub
    assert "secret reasoning" not in theorist.view("/memories/MEMORY.md")

    project.review(scout_page.path, "accept")
    assert "Unreviewed synthesis." in theorist.view(f"/memories/{scout_page.path}")


def test_attached_files_are_read_as_text_and_only_read(project, theorist):
    text = theorist.view("/memories/inputs/notes.txt")
    assert shown(text) == "line one\nline two\nline three\n"
    assert theorist.view("/memories/inputs/notes.txt", (2, 2)).splitlines()[1:] == [
        "     2\tline two"
    ]
    assert BRIEF in theorist.view("/memories/inputs/brief.md")
    assert '"notes.txt"' in theorist.view("/memories/inputs/manifest.json")
    with pytest.raises(ToolError, match="not a text file"):
        theorist.view("/memories/inputs/figure.png")
    (project.inputs_dir / "big.csv").write_text("x" * 2_000_001)
    with pytest.raises(ToolError, match="too large to read here"):
        theorist.view("/memories/inputs/big.csv")

    before = (project.inputs_dir / "notes.txt").read_text()
    for call in (
        lambda: theorist.create("/memories/inputs/notes.txt", "mine"),
        lambda: theorist.create("/memories/inputs/new.md", "mine"),
        lambda: theorist.str_replace("/memories/inputs/notes.txt", "line one", "mine"),
        lambda: theorist.insert("/memories/inputs/notes.txt", 0, "mine"),
        lambda: theorist.delete("/memories/inputs/notes.txt"),
    ):
        with pytest.raises(ToolError, match="person's own files and is read-only"):
            call()
    assert (project.inputs_dir / "notes.txt").read_text() == before
    assert not (project.inputs_dir / "new.md").exists()
    with pytest.raises(ToolError, match="index, generated from the wiki"):
        theorist.create("/memories/MEMORY.md", "mine")
    with pytest.raises(ToolError, match="index, generated from the wiki"):
        theorist.str_replace("/memories/MEMORY.md", "Research", "My")


# -- creating -------------------------------------------------------------------


def test_a_page_is_created_as_a_proposal_with_the_front_matter_completed(project, theorist):
    project.create_page("topic", "Background", "What is known.", author="human")
    text = HYPOTHESIS.replace(
        "---\n## Statement", "motivated_by:\n- topics/background\n---\n## Statement"
    )
    result = theorist.create("/memories/wiki/hypotheses/warmup-variance.md", text)
    assert result.startswith(
        "File created successfully at: /memories/wiki/hypotheses/H1-warmup-variance.md\n"
    )
    assert "It is H1" in result and "proposal" in result and "revise it in this session" in result
    page = project.page("H1")
    assert page.path == "wiki/hypotheses/H1-warmup-variance.md"
    assert (page.kind, page.review, page.state, page.author, page.round) == (
        "hypothesis",
        "proposed",
        "open",
        "agent:theorist",
        1,
    )
    assert page.title == "Warmup reduces variance"
    assert page.summary == "Linear warmup lowers the variance of early updates"
    assert page.relations == {"motivated_by": ["topics/background"]}
    assert page.session == "s-theorist-1"
    assert page.body.startswith("## Statement")
    assert theorist.created == theorist.changed == [page.path]
    event = project.events()[-1]
    assert (event["type"], event["agent"], event["round"]) == ("page.created", "theorist", 1)
    # It reads its own proposal back, as written to disk.
    assert "review: proposed" in theorist.view(f"/memories/{page.path}")


def test_what_the_text_says_about_review_and_state_does_not_count(project):
    designer = backend(project, "designer")
    result = designer.create(
        "/memories/wiki/experiments/E7-sweep.md",
        "---\n"
        "title: Sweep warmup length\n"
        "kind: finding\n"
        "id: E7\n"
        "review: accepted\n"
        "state: approved\n"
        "author: human\n"
        "round: 99\n"
        "decided_by: human\n"
        "session: someone-else\n"
        "mode: script\n"
        "timeout_seconds: 600\n"
        "packages:\n- numpy\n"
        "---\n"
        "## Objective\n\nMeasure it.\n",
    )
    assert "/memories/wiki/experiments/E1-sweep.md" in result
    assert "were left as they are" in result
    for name in ("author", "decided_by", "id", "kind", "review", "round", "session", "state"):
        assert name in result.split("were left as they are:")[1]
    page = project.page("E1")
    assert (page.id, page.kind, page.review, page.state, page.author, page.round) == (
        "E1",
        "experiment",
        "proposed",
        "planned",
        "agent:designer",
        1,
    )
    assert "decided_by" not in page.meta and page.session == "s-designer"
    assert page.meta["mode"] == "script" and page.meta["timeout_seconds"] == 600
    assert page.meta["packages"] == ["numpy"]
    assert project.pending() == [page]

    # Nor does a later edit of its own page get it approved or accepted.
    path = f"/memories/{page.path}"
    with pytest.raises(ToolError, match=r"No change was made.*review.*only by the person"):
        designer.str_replace(path, "review: proposed", "review: accepted")
    with pytest.raises(ToolError, match=r"No change was made.*state"):
        designer.str_replace(path, "state: planned", "state: approved")
    with pytest.raises(ToolError, match="No change was made"):
        designer.insert(path, 1, "decided_by: human")
    mixed = designer.create(
        path, "---\ntitle: Sweep warmup\nstate: approved\nreview: accepted\n---\nNew plan.\n"
    )
    assert "were left as they are: review, state" in mixed
    page = project.page("E1")
    assert (page.review, page.state, page.title) == ("proposed", "planned", "Sweep warmup")
    assert page.body == "New plan.\n" and "mode" not in page.meta
    assert project.counts()["experiments"]["approved"] == 0


def test_a_title_is_found_when_the_front_matter_has_none(project, theorist):
    theorist.create(
        "/memories/wiki/hypotheses/h3_cosine-decay.md", "# Cosine decay helps\n\nIt does."
    )
    theorist.create("/memories/wiki/questions/why-seeds-matter.md", "No heading, just text.")
    theorist.create(
        "/memories/wiki/hypotheses/H9.md", "---\ntitle: Named by number only\n---\nText."
    )
    assert [(p.id, p.title, p.path) for p in project.pages(kind=["hypothesis", "question"])] == [
        ("Q1", "Why seeds matter", "wiki/questions/Q1-why-seeds-matter.md"),
        ("H1", "Cosine decay helps", "wiki/hypotheses/H1-cosine-decay.md"),
        ("H2", "Named by number only", "wiki/hypotheses/H2-named-by-number-only.md"),
    ]
    with pytest.raises(ToolError, match="has no text"):
        theorist.create("/memories/wiki/hypotheses/empty.md", "---\ntitle: Empty\n---\n\n")
    with pytest.raises(ToolError, match="file_text must be a string"):
        theorist.create("/memories/wiki/hypotheses/x.md", None)


ROLE_SECTIONS = {
    "intake": {"topics", "questions"},
    "scout": {"topics", "questions"},
    "theorist": {"hypotheses", "questions"},
    "designer": {"experiments"},
    "analyst": {"findings", "questions", "hypotheses", "experiments", "journal"},
    "runner": set(),
}


@pytest.mark.parametrize("role", sorted(ROLE_SECTIONS))
def test_each_role_creates_only_its_kinds(project, role):
    session = backend(project, role)
    sections = (
        "questions",
        "topics",
        "hypotheses",
        "experiments",
        "findings",
        "journal",
        "lessons",
    )
    for section in sections:
        path = f"/memories/wiki/{section}/a-page.md"
        if section in ROLE_SECTIONS[role]:
            assert "File created successfully" in session.create(path, "# A page\n\nText.")
        else:
            with pytest.raises(ToolError) as caught:
                session.create(path, "# A page\n\nText.")
            message = str(caught.value)
            assert f"The {role} does not create" in message and "propose_edit" in message
            for allowed in ROLE_SECTIONS[role]:
                assert f"/memories/wiki/{allowed}/" in message
    created = {page.path.split("/")[1] for page in project.pages() if page.kind != "idea"}
    assert created == ROLE_SECTIONS[role]
    assert all(page.author == f"agent:{role}" for page in project.pages() if page.kind != "idea")


def test_places_no_role_writes(project, theorist):
    cases = {
        "/memories/wiki/sources/Smith2020Deep.md": "records that the pipeline writes.*verify_paper",
        "/memories/wiki/results/E1-run1.md": "records that the pipeline writes",
        "/memories/wiki/idea.md": "written by the person; propose an edit with propose_edit",
        "/memories/wiki/loose.md": "not a place for a page.*hypotheses/, /memories/wiki/questions/",
        "/memories/wiki/essays/x.md": "not a place for a page",
        "/memories/wiki/hypotheses/deep/x.md": "not a place for a page",
        "/memories/wiki/hypotheses/x.txt": r"not a place for a page.*\(\.md\)",
        "/memories/notes/x.md": "not a place for a page",
        "/memories/transcripts/x.md": "not a place for a page",
        "/memories/proposals/ep-00000000.json": "not a place for a page",
        "/memories/wiki/hypotheses": "not a place for a page",
        "/memories": "is a directory",
    }
    for path, message in cases.items():
        with pytest.raises(ToolError, match=message):
            theorist.create(path, HYPOTHESIS)
    assert [page.path for page in project.pages()] == ["wiki/idea.md"]
    assert not (project.root / "proposals" / "ep-00000000.json").exists()


def test_relations_must_point_at_pages_the_session_can_build_on(project, theorist):
    project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    project.review(
        project.write_record("source", "Bad2019", "Bad", "x").path, "reject", reason="Retracted"
    )
    project.create_page("topic", "The scout's draft", "x", author="agent:scout")
    accepted = project.review(
        project.create_page("topic", "Warmup", "x", author="agent:scout").path, "accept"
    )

    def page(front: str) -> str:
        return f"---\ntitle: T\n{front}\n---\nText.\n"

    cases = {
        "cites: Ghost1999": "cites: 'Ghost1999' is not a page in this wiki.*verify_paper",
        "motivated_by: topics/the-scout-s-draft": "is not a page in this wiki",
        "motivated_by: H9": "motivated_by: 'H9' is not a page in this wiki. Use an id",
        "cites: Bad2019": r"cites: Bad2019 was rejected by the person \(Retracted\)",
        "cites: topics/warmup": "cites: topics/warmup is a topic; cites points at: source",
        "tests: H1": "'tests' is not a field of a hypothesis page",
        "motivated_by: idea": "motivated_by: idea is an idea; motivated_by points at: source",
    }
    for front, message in cases.items():
        with pytest.raises(ToolError, match=message):
            theorist.create("/memories/wiki/hypotheses/x.md", page(front))
    assert project.pages(kind="hypothesis") == []

    theorist.create(
        "/memories/wiki/hypotheses/x.md",
        page("cites: [smith2020deep]\nmotivated_by:\n- [[topics/warmup]]\n- Smith2020Deep"),
    )
    assert project.page("H1").relations == {
        "cites": ["Smith2020Deep"],
        "motivated_by": [accepted.link, "Smith2020Deep"],
    }
    # A source the person has yet to look at is still a fact that can be cited.
    project.settings.review["source"] = "review"
    project.write_record("source", "Lee2021Wide", "Wide", "A.")
    theorist.create("/memories/wiki/hypotheses/y.md", page("cites: Lee2021Wide"))
    assert project.page("H2").relations == {"cites": ["Lee2021Wide"]}
    with pytest.raises(ToolError, match="does not exist"):
        theorist.view("/memories/wiki/sources/Lee2021Wide.md")


def test_a_tool_creates_a_page_for_the_session_under_the_same_rules(project):
    designer = backend(project, "designer")
    project.review(
        project.create_page("hypothesis", "Warmup", "x", author="agent:theorist").path, "accept"
    )
    page = designer.create_page(
        "experiment",
        "Sweep warmup length",
        "## Objective\n\nMeasure it.",
        summary="A sweep over five lengths",
        fields={"mode": "script", "state": "approved", "review": "accepted"},
        relations={"tests": ["h1"]},
    )
    assert (page.id, page.review, page.state, page.author) == (
        "E1",
        "proposed",
        "planned",
        "agent:designer",
    )
    assert page.session == "s-designer" and page.relations == {"tests": ["H1"]}
    assert page.meta["mode"] == "script" and page.summary == "A sweep over five lengths"
    assert designer.created == [page.path]
    # It is the session's own page: readable and revisable through the memory tool.
    assert designer.str_replace(f"/memories/{page.path}", "Measure it.", "Measure it twice.")
    with pytest.raises(ToolError, match="The designer does not create finding pages"):
        designer.create_page("finding", "Gap", "x")
    with pytest.raises(ToolError, match="tests: 'H9' is not a page in this wiki"):
        designer.create_page("experiment", "Other", "x", relations={"tests": ["H9"]})
    with pytest.raises(ToolError, match="has no text"):
        designer.create_page("experiment", "Empty", "  ")


# -- revising -------------------------------------------------------------------


def test_a_session_revises_the_pages_it_created(project, theorist):
    theorist.create("/memories/wiki/hypotheses/warmup.md", HYPOTHESIS)
    path = "/memories/wiki/hypotheses/H1-warmup.md"
    assert theorist.str_replace(path, "first thousand", "first hundred") == (
        "The memory file has been edited."
    )
    assert theorist.str_replace(
        path, "title: Warmup reduces variance", "title: Warmup halves variance"
    )
    lines = theorist.view(path).splitlines()
    end = next(i for i, line in enumerate(lines) if line.endswith("\t## Statement"))
    assert "has been edited" in theorist.insert(path, end + 1, "## Risks\n\nSmall effect.")
    page = project.page("H1")
    assert page.title == "Warmup halves variance" and "first hundred" in page.body
    assert "## Risks\n\nSmall effect." in page.body
    assert (page.review, page.state, page.author, page.session) == (
        "proposed",
        "open",
        "agent:theorist",
        "s-theorist-1",
    )
    assert len(project.history("H1")) == 3 and theorist.changed == [page.path]

    # Writing the whole file again replaces the text; a field left out is removed.
    theorist.create(path, "---\ntitle: Warmup halves variance\ncites: []\n---\nShorter.\n")
    page = project.page("H1")
    assert page.body == "Shorter.\n" and page.relations == {} and "summary" not in page.meta
    assert page.id == "H1" and page.review == "proposed"

    with pytest.raises(ToolError, match="did not appear verbatim"):
        theorist.str_replace(path, "not there", "x")
    theorist.create(path, "---\ntitle: T\n---\nsame same\n")
    with pytest.raises(ToolError, match="Multiple occurrences"):
        theorist.str_replace(path, "same", "x")
    with pytest.raises(ToolError, match="inside the page's front matter"):
        theorist.insert(path, 2, "free text that is not a field")
    with pytest.raises(ToolError, match="range of lines"):
        theorist.insert(path, 999, "x")
    with pytest.raises(ToolError, match="would have no text left"):
        theorist.str_replace(path, "same same", "")
    with pytest.raises(ToolError, match="'tests' is not a field of a hypothesis"):
        theorist.insert(path, 1, "tests: H1")


def test_what_a_session_may_not_change(project, theorist):
    theorist.create("/memories/wiki/hypotheses/accepted.md", HYPOTHESIS)
    theorist.create("/memories/wiki/hypotheses/rejected.md", HYPOTHESIS)
    project.review("H1", "accept")
    project.review("H2", "reject", reason="Too vague")
    persons = project.create_page("hypothesis", "The person's", "Mine.", author="human")
    others = project.create_page("question", "The analyst's", "Why?", author="agent:analyst")
    project.review(others.path, "accept")
    record = project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    earlier = backend(project, "theorist", session="s-theorist-0")
    earlier.create("/memories/wiki/hypotheses/earlier.md", HYPOTHESIS)
    journal = project.create_page("journal", "Round 1", "Auto-accepted.", author="agent:analyst")
    project.settings.review["question"] = "auto"
    earlier.create("/memories/wiki/questions/from-before.md", "# From before\n\nWhy?")

    cases = {
        "wiki/hypotheses/H1-accepted.md": "H1 was accepted by the person; propose an edit with propose_edit",
        "wiki/hypotheses/H2-rejected.md": r"H2 was rejected by the person \(Too vague\).*write a new page",
        persons.path: "H3 was written by the person; propose an edit with propose_edit",
        "wiki/idea.md": "was written by the person; propose an edit with propose_edit",
        others.path: "Q1 was accepted by the person; propose an edit with propose_edit",
        journal.path: "was written by agent:analyst; propose an edit with propose_edit",
        record.path: "Smith2020Deep is a record the pipeline wrote",
        "wiki/questions/Q2-from-before.md": "Q2 is your page from an earlier session.*propose_edit",
    }
    for rel, message in cases.items():
        path = f"/memories/{rel}"
        before = (project.root / rel).read_text()
        for call in (
            lambda path=path: theorist.create(path, "---\ntitle: Mine\n---\nOverwritten.\n"),
            lambda path=path: theorist.str_replace(path, "title", "caption"),
            lambda path=path: theorist.insert(path, 0, "x"),
            lambda path=path: theorist.delete(path),
        ):
            with pytest.raises(ToolError, match=message):
                call()
        assert (project.root / rel).read_text() == before
    # Another session's pending proposal is not there to be changed, or to be written over.
    pending = "/memories/wiki/hypotheses/H4-earlier.md"
    for call in (
        lambda: theorist.str_replace(pending, "title", "caption"),
        lambda: theorist.insert(pending, 0, "x"),
        lambda: theorist.delete(pending),
    ):
        with pytest.raises(ToolError, match="does not exist"):
            call()
    assert "H5-earlier.md" in theorist.create(pending, "---\ntitle: Mine\n---\nA page of my own.\n")
    assert "first thousand" in project.page("H4").body
    # The session that wrote it can still revise it.
    assert earlier.str_replace(pending, "first thousand", "first ten")
    # And a resumed session, opened again under the same id, picks its pages up.
    resumed = backend(project, "theorist", session="s-theorist-0")
    assert resumed.str_replace("/memories/wiki/hypotheses/H4-earlier.md", "first ten", "first five")


def test_a_page_accepted_on_creation_stays_the_sessions_until_the_person_decides(project):
    analyst = backend(project, "analyst")
    result = analyst.create("/memories/wiki/journal/round-1.md", "# Round 1\n\nWhat happend.")
    assert "accepted on creation" in result
    path = "/memories/wiki/journal/round-1.md"
    analyst.str_replace(path, "happend", "happened")
    assert project.page("wiki/journal/round-1.md").body.endswith("What happened.\n")
    project.update_page("wiki/journal/round-1.md", author="human", body="The person's account.")
    with pytest.raises(ToolError, match="accepted by the person; propose an edit"):
        analyst.str_replace(path, "account", "story")


def test_a_session_cannot_rewrite_what_the_person_approved_without_review(tmp_path):
    # Where experiments are accepted without review, approving the run is the only decision
    # the person makes: the plan they approved must not change under them afterwards.
    project = Project.create(
        tmp_path / "auto",
        title="T",
        brief=BRIEF,
        settings={"review": {"experiment": "auto", "hypothesis": "auto"}},
    )
    designer = backend(project, "designer")
    project.create_page("hypothesis", "Warmup", "It does.", author="human")
    plan = "---\ntitle: Sweep\nmode: script\nnetwork: false\ntests: H1\n---\n## Objective\n\nx\n"
    designer.create("/memories/wiki/experiments/sweep.md", plan)
    path = "/memories/wiki/experiments/E1-sweep.md"
    assert project.page("E1").review == "accepted" and project.page("E1").decided_by is None
    # Until the person acts on it, the session still shapes its plan.
    designer.str_replace(path, "## Objective\n\nx", "## Objective\n\nMeasure it.")

    approved = project.set_state("E1", "approved", by="human")
    attempts = [
        lambda: designer.str_replace(path, "network: false", "network: true"),
        lambda: designer.create(path, plan.replace("script", "gpu")),
        lambda: designer.insert(path, 20, "More."),
        lambda: designer.delete(path),
    ]
    for attempt in attempts:
        with pytest.raises(ToolError, match=r"E1 is approved.*propose an edit with propose_edit"):
            attempt()
    assert project.page("E1") == approved and project.page("E1").meta["network"] is False
    with pytest.raises(NotAllowed, match="E1 is approved"):
        project.update_page("E1", author="agent:designer", fields={"network": True})
    with pytest.raises(NotAllowed, match="E1 is approved"):
        project.delete_page("E1", author="agent:designer")

    # The same holds for a hypothesis once it is under test, and for a run that is over.
    theorist = backend(project, "theorist")
    theorist.create("/memories/wiki/hypotheses/noise.md", HYPOTHESIS)
    project.set_state("H2", "testing", by="system")
    with pytest.raises(ToolError, match="H2 is testing"):
        theorist.str_replace("/memories/wiki/hypotheses/H2-noise.md", "lowers", "raises")
    for state in ("queued", "running", "succeeded"):
        project.set_state("E1", state, by="system")
    with pytest.raises(ToolError, match="E1 is succeeded"):
        designer.str_replace(path, "Measure it.", "Measured something else.")
    # What it can still do is suggest, and the person decides.
    tool = propose_edit_tool(project, "designer", 1)
    assert "Proposed as ep-" in tool.handler(
        {"target": "E1", "summary": "Needs the network", "fields": {"network": True}}
    )
    assert project.page("E1").meta["network"] is False


def test_deleting_and_renaming(project, theorist):
    theorist.create("/memories/wiki/hypotheses/warmup.md", HYPOTHESIS)
    path = "/memories/wiki/hypotheses/H1-warmup.md"
    with pytest.raises(ToolError, match=r"Pages are not renamed.*title"):
        theorist.rename(path, "/memories/wiki/hypotheses/H1-better.md")
    assert theorist.delete(path) == f"Successfully deleted {path}"
    assert (
        project.resolve("H1") is None and len(project.history("wiki/hypotheses/H1-warmup.md")) == 1
    )
    assert theorist.created == []
    with pytest.raises(ToolError, match="does not exist"):
        theorist.delete(path)
    with pytest.raises(ToolError, match="Cannot delete the /memories directory"):
        theorist.delete("/memories")
    with pytest.raises(ToolError, match="does not exist"):
        theorist.delete("/memories/wiki/hypotheses")
    with pytest.raises(ToolError, match="written by the person"):
        theorist.delete("/memories/wiki/idea.md")


# -- tools ----------------------------------------------------------------------


def test_propose_edit_suggests_a_change_for_the_person_to_decide(project):
    project.review(
        project.create_page("hypothesis", "Warmup", "It does.", author="agent:theorist").path,
        "accept",
    )
    project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    tool = propose_edit_tool(project, "analyst", 2)
    assert tool.name == "propose_edit"
    assert tool.input_schema["required"] == ["target", "summary"]
    assert set(tool.input_schema["properties"]) == {"target", "summary", "body", "fields"}

    result = tool.handler(
        {
            "target": "h1",
            "summary": "Mark H1 as supported by F1",
            "body": "---\ntitle: Warmup\nreview: accepted\n---\nIt does, by half.",
            "fields": {"state": "supported", "cites": ["Smith2020Deep"]},
        }
    )
    (proposal,) = project.proposals()
    assert result.startswith(f"Proposed as {proposal.id}: Mark H1 as supported by F1")
    assert "unchanged until the person accepts" in result and "front matter" in result
    assert (proposal.author, proposal.round, proposal.target) == (
        "agent:analyst",
        2,
        "wiki/hypotheses/H1-warmup.md",
    )
    assert proposal.body == "It does, by half.\n"
    assert proposal.fields == {"state": "supported"}
    assert proposal.relations_add == {"cites": ["Smith2020Deep"]}
    assert project.page("H1").state == "open"
    # Asking twice does not fill the person's inbox.
    again = tool.handler(
        {
            "target": "/memories/wiki/hypotheses/H1-warmup.md",
            "summary": "Mark H1 as supported",
            "body": "It does, by half.",
            "fields": {"state": "supported", "cites": ["Smith2020Deep"]},
        }
    )
    assert (
        again == f"This edit is already proposed as {proposal.id} and awaits the person's decision."
    )
    assert len(project.proposals()) == 1

    project.review(proposal.id, "accept")
    assert project.page("H1").state == "supported"


def test_propose_edit_says_what_to_do_when_it_cannot(project):
    exp = project.review(
        project.create_page("experiment", "Sweep", "x", author="agent:designer").path, "accept"
    )
    source = project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    hidden = project.create_page("topic", "The scout's draft", "x", author="agent:scout")
    rejected = project.review(
        project.create_page("topic", "Bad", "x", author="agent:scout").path,
        "reject",
        reason="Off topic",
    )
    tool = propose_edit_tool(project, "agent:analyst", 1)
    cases = [
        (
            {"target": exp.id, "summary": "Run it", "fields": {"state": "approved"}},
            "only when the person approves",
        ),
        (
            {"target": exp.id, "summary": "Accept it", "fields": {"review": "accepted"}},
            "'review' is set by the project",
        ),
        ({"target": source.path, "summary": "Fix", "body": "Better."}, "is a record"),
        ({"target": hidden.path, "summary": "Fix", "body": "Better."}, "No page matches"),
        ({"target": "H9", "summary": "Fix", "body": "Better."}, "No page matches 'H9'. Use an id"),
        (
            {"target": rejected.path, "summary": "Fix", "body": "Better."},
            r"rejected by the person \(Off topic\)",
        ),
        ({"target": exp.id, "summary": "Nothing"}, "changes something"),
        ({"target": exp.id, "summary": " ", "body": "x"}, "summary must say"),
        ({"target": "", "summary": "s", "body": "x"}, "target must name a page"),
        ({"target": exp.id, "summary": "s", "body": "  "}, "full new text"),
        (
            {"target": exp.id, "summary": "s", "fields": ["state"]},
            "fields, when given, is an object",
        ),
    ]
    for tool_input, message in cases:
        with pytest.raises(ToolError, match=message):
            tool.handler(tool_input)
    assert project.proposals() == [] and project.page(exp.id) == exp
    # Its own pending proposal waits for the person as it is.
    own = project.create_page("finding", "Gap", "x", author="agent:analyst")
    with pytest.raises(ToolError, match="F1 is your own proposal and still awaits"):
        tool.handler({"target": own.id, "summary": "Sharpen", "body": "y"})
    assert project.proposals() == []


def test_a_proposed_relation_must_point_at_a_page_the_agent_can_build_on(project):
    target = project.create_page("hypothesis", "Warmup", "It does.", author="human")
    project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    project.review(
        project.create_page("topic", "Bad", "x", author="agent:scout").path,
        "reject",
        reason="Off topic",
    )
    project.create_page("topic", "The scout's draft", "x", author="agent:scout")
    own = project.create_page("finding", "Gap", "x", author="agent:analyst")
    tool = propose_edit_tool(project, "analyst", 1)
    cases = [
        # A source nobody verified would be an invented citation once the person accepts.
        ({"cites": ["Fake2020Nope"]}, "cites: 'Fake2020Nope' is not a page in this wiki"),
        ({"cites": ["Smith2020Deep", "Other1999"]}, "'Other1999' is not a page"),
        ({"motivated_by": ["idea"]}, "motivated_by: idea is an idea; motivated_by points at"),
        ({"motivated_by": ["topics/bad"]}, r"rejected by the person \(Off topic\)"),
        # Another agent's unreviewed proposal does not exist for this one.
        ({"motivated_by": ["topics/the-scout-s-draft"]}, "is not a page in this wiki"),
        ({"tests": ["H1"]}, "'tests' is not a field of a hypothesis page"),
    ]
    for fields, message in cases:
        with pytest.raises(ToolError, match=message):
            tool.handler({"target": target.id, "summary": "Relate", "fields": fields})
    assert project.proposals() == []

    # A verified source and the agent's own proposal are things it can point at.
    tool.handler(
        {
            "target": target.id,
            "summary": "Say where it comes from",
            "fields": {"cites": ["smith2020deep"], "motivated_by": [own.id]},
        }
    )
    (proposal,) = project.proposals()
    assert proposal.relations_add == {"cites": ["Smith2020Deep"], "motivated_by": ["F1"]}


def test_search_finds_accepted_pages_only(project):
    project.review(
        project.create_page(
            "hypothesis",
            "Warmup reduces gradient noise",
            "Accuracy rose to 87.3 percent.",
            author="agent:theorist",
        ).path,
        "accept",
    )
    project.create_page(
        "hypothesis",
        "Cosine decay reduces gradient noise",
        "Unreviewed: 87.3 percent.",
        author="agent:theorist",
    )
    project.review(
        project.create_page(
            "hypothesis",
            "Dropout reduces gradient noise",
            "Rejected: 87.3 percent.",
            author="agent:theorist",
        ).path,
        "reject",
    )
    project.wiki.add_note("gradient noise note: 87.3 percent", stage="research")
    tool = search_tool(project)
    assert tool.name == "wiki_search" and tool.input_schema["required"] == ["query"]

    result = tool.handler({"query": "gradient noise"})
    assert result.startswith(
        "1. /memories/wiki/hypotheses/H1-warmup-reduces-gradient-noise.md: Warmup"
    )
    assert "Cosine" not in result and "Dropout" not in result and "notes/" not in result
    assert tool.handler({"query": "gradient noise", "kind": "topic"}) == "No matching pages."
    exact = tool.handler({"query": "87.3 percent", "exact": True})
    assert exact.startswith("/memories/wiki/hypotheses/H1-warmup-reduces-gradient-noise.md:")
    assert exact.count("\n") == 0 and "Accuracy rose to 87.3 percent." in exact
    assert tool.handler({"query": "99.9 percent", "exact": True}) == (
        "No line contains that text verbatim."
    )
    for bad, message in (
        ({"query": " "}, "non-empty"),
        ({"query": "x", "kind": "essay"}, "kind must be one of"),
        ({"query": "x", "k": "many"}, "k must be an integer"),
    ):
        with pytest.raises(ToolError, match=message):
            tool.handler(bad)


# -- through the harness, as an agent session runs ------------------------------


def test_an_agent_session_fills_the_wiki_with_proposals(project):
    project.review(
        project.create_page("hypothesis", "Warmup", "It does.", author="agent:theorist").path,
        "accept",
    )
    seen: dict = {}

    def analyst(request: FakeRequest) -> str:
        seen["index"] = request.memory.view("/memories/MEMORY.md")
        seen["tools"] = sorted(request.tools)
        request.memory.create(
            "/memories/wiki/findings/variance-halved.md",
            "---\ntitle: Variance halved\nsupports: H1\nreview: accepted\n---\n"
            "## What was observed\n\nVariance fell by half.\n",
        )
        seen["denied"] = request.call_tool(
            "propose_edit", {"target": "H1", "summary": "Approve", "fields": {"review": "accepted"}}
        )
        seen["proposed"] = request.call_tool(
            "propose_edit",
            {"target": "H1", "summary": "Supported by F1", "fields": {"state": "supported"}},
        )
        try:
            request.memory.str_replace(
                "/memories/wiki/hypotheses/H1-warmup.md", "It does.", "It certainly does."
            )
        except ToolError as exc:
            seen["refused"] = str(exc)
        seen["search"] = request.call_tool("wiki_search", {"query": "warmup"})
        return "done"

    harness = ContextHarness(
        project=project.wiki,
        transcripts=TranscriptLog(project.root / "transcripts", session_id="s1"),
        client_for=lambda model: FakeLLM(analyst),
        dream_mode="off",
    )
    session = ProjectMemoryBackend(project, "analyst", 1, "s1")
    conversation = harness.open(
        agent="analyst",
        stage="analyze",
        model="fake",
        system="You analyse results.",
        memory=MemoryPolicy.none(),
        memory_backend=session,
        tools=[propose_edit_tool(project, "analyst", 1), search_tool(project)],
    )
    assert conversation.send("go").text == "done"

    assert "hypotheses/H1-warmup.md [H1, accepted, open] Warmup" in seen["index"]
    assert seen["tools"] == ["propose_edit", "wiki_search"]
    assert "'review' is set by the project" in seen["denied"]
    assert seen["proposed"].startswith("Proposed as ep-")
    assert seen["refused"] == "H1 was accepted by the person; propose an edit with propose_edit."
    assert "/memories/wiki/hypotheses/H1-warmup.md" in seen["search"]

    finding = project.page("F1")
    assert (finding.review, finding.author, finding.relations) == (
        "proposed",
        "agent:analyst",
        {"supports": ["H1"]},
    )
    assert session.created == [finding.path]
    waiting = project.pending()
    assert [item.id for item in waiting] == ["F1", project.proposals()[0].id]
    assert project.page("H1").state == "open" and project.page("H1").body == "It does.\n"
    types = [event["type"] for event in project.events()]
    assert types[-2:] == ["page.created", "proposal.created"]
    # The session is on record, and the record is not something a session can read.
    assert any(event["type"] == "tool_call" for event in harness.transcripts.events())
    with pytest.raises(ToolError, match="does not exist"):
        session.view("/memories/transcripts")
    assert json.loads((project.root / "project.json").read_text())["id"] == project.id


# -- the record of an approval (audit: containment 3; execution 2) ---------------------


def test_an_agents_page_cannot_say_where_its_run_goes_or_that_it_was_approved(project):
    """``backend`` and the digests an approval leaves are the person's to decide: what
    the designer writes into its page for them is dropped, on creation and on revision."""
    designer = backend(project, "designer")
    result = designer.create(
        "/memories/wiki/experiments/sweep.md",
        "---\n"
        "title: Sweep warmup length\n"
        "mode: script\n"
        "backend: local\n"
        f"approved_digest: {'0' * 64}\n"
        "approved_at: 2026-11-15T12:00:00Z\n"
        f"repaired_digest: {'1' * 64}\n"
        "---\n"
        "## Objective\n\nMeasure it.\n",
    )
    listed_as_ignored = result.split("were left as they are:")[1]
    for name in ("approved_at", "approved_digest", "backend", "repaired_digest"):
        assert name in listed_as_ignored
    page = project.page("E1")
    assert page.meta["mode"] == "script"
    assert not {"backend", "approved_digest", "approved_at", "repaired_digest"} & set(page.meta)

    path = f"/memories/{page.path}"
    with pytest.raises(ToolError, match=r"No change was made.*backend"):
        designer.insert(path, 1, "backend: local")
    revised = designer.create(
        path, "---\ntitle: Sweep warmup length\nmode: script\nbackend: local\n---\nNew plan.\n"
    )
    assert "were left as they are: backend" in revised
    assert "backend" not in project.page("E1").meta and project.page("E1").body == "New plan.\n"


def test_an_agent_cannot_propose_where_a_run_goes(project):
    exp = project.review(
        project.create_page("experiment", "Sweep", "x", author="agent:designer").path, "accept"
    )
    tool = propose_edit_tool(project, "agent:analyst", 1)
    for field in ("backend", "approved_digest", "approved_at", "repaired_digest"):
        with pytest.raises(ToolError, match=f"'{field}' is part of the record of an approval"):
            tool.handler({"target": exp.id, "summary": "Tidy", "fields": {field: "local"}})
    assert project.proposals() == [] and project.page(exp.id) == exp
