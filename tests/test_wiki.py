from __future__ import annotations

import threading

import pytest

from paper_orchestra.memory.wiki import (
    CATALOG_START,
    INDEX_FILE,
    Wiki,
    WikiError,
    join_front_matter,
    link_for,
    split_front_matter,
)


@pytest.fixture
def wiki(tmp_path, fixed_clock) -> Wiki:
    return Wiki(tmp_path / "memory", name="demo", clock=fixed_clock)


def test_front_matter_round_trip():
    meta = {
        "title": "A: tricky title",
        "year": 2017,
        "authors": ["Ada", "Bob"],
        "numeric_string": "2017",
        "flag": True,
        "bool_string": "true",
        "leading_dash": "-x",
    }
    parsed, body = split_front_matter(join_front_matter(meta, "Body\n"))
    assert parsed == meta
    assert body == "Body\n"


def test_front_matter_block_lists_and_garbage():
    meta, body = split_front_matter("---\ntags:\n  - a\n  - b\n???\nx: 1\n---\nhello")
    assert meta == {"tags": ["a", "b"], "x": 1}
    assert body == "hello"
    assert split_front_matter("no front matter") == ({}, "no front matter")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "/etc/passwd",
        "../outside.md",
        "wiki/../../outside.md",
        "wiki/.hidden.md",
        "wiki\\x.md",
        "transcripts/session.jsonl",
        "wiki/page.txt",
        "wiki",
        "other/page.md",
        "wiki/a\x00b.md",
    ],
)
def test_normalize_rejects_unsafe_paths(wiki, bad):
    with pytest.raises(WikiError):
        wiki.normalize(bad)


def test_normalize_rejects_symlink_escape(wiki, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (wiki.root / "wiki" / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WikiError):
        wiki.normalize("wiki/link/secret.md")


def test_write_read_stamps_metadata(wiki):
    wiki.write("wiki/method/overview.md", "# Overview\nText", {"summary": "s"}, author="ingest")
    page = wiki.read("wiki/method/overview.md")
    assert page.meta["summary"] == "s"
    assert page.meta["author"] == "ingest"
    assert page.meta["updated"].endswith("Z")
    assert page.title == "Overview"
    assert page.body.endswith("\n")


def test_sources_are_read_only_to_agents(wiki):
    wiki.write("wiki/sources/idea.md", "ground truth", force=True)
    for operation in (
        lambda: wiki.write("wiki/sources/idea.md", "tampered"),
        lambda: wiki.write_text("wiki/sources/idea.md", "tampered"),
        lambda: wiki.append("wiki/sources/idea.md", "more"),
        lambda: wiki.delete("wiki/sources/idea.md"),
        lambda: wiki.rename("wiki/sources/idea.md", "wiki/x.md"),
        lambda: wiki.rename("wiki/x.md", "wiki/sources/y.md"),
    ):
        with pytest.raises(WikiError):
            operation()
    assert wiki.read("wiki/sources/idea.md").body.strip() == "ground truth"


def test_page_size_limit(tmp_path):
    wiki = Wiki(tmp_path, max_page_bytes=100)
    with pytest.raises(WikiError, match="Split it"):
        wiki.write("wiki/big.md", "x" * 500)
    assert not wiki.exists("wiki/big.md")
    assert not list((tmp_path / "wiki").glob(".*tmp"))


def test_links_resolve_and_repair(wiki):
    wiki.write("wiki/literature/Vaswani2017Attention.md", "x", {"title": "Attention"})
    wiki.write(
        "wiki/synthesis/themes.md",
        "See [[literature/Vaswani2017Attention]], [[vaswani2017attention|the paper]], "
        "[[literature/DoesNotExist]] and [[Attention#abstract]].",
    )
    assert wiki.resolve_link("literature/Vaswani2017Attention") == (
        "wiki/literature/Vaswani2017Attention.md"
    )
    assert wiki.resolve_link("MEMORY") == INDEX_FILE
    assert wiki.resolve_link("../../etc/passwd") is None

    broken = {target for _, target in wiki.lint().broken_links}
    assert broken == {"vaswani2017attention", "literature/DoesNotExist", "Attention"}

    repairs = wiki.repair_links()
    assert {old for _, old, _ in repairs} == {"vaswani2017attention", "Attention"}
    body = wiki.read("wiki/synthesis/themes.md").body
    assert "[[literature/Vaswani2017Attention|the paper]]" in body
    assert "[[literature/Vaswani2017Attention#abstract]]" in body
    assert [t for _, t in wiki.lint().broken_links] == ["literature/DoesNotExist"]


def test_repair_leaves_ambiguous_links_alone(wiki):
    wiki.write("wiki/a/topic.md", "x")
    wiki.write("wiki/b/topic.md", "x")
    wiki.write("wiki/c/page.md", "[[topic]]")
    assert wiki.repair_links() == []


def test_search_ranks_title_matches_first(wiki):
    wiki.write("wiki/literature/a.md", "mentions diffusion once", {"title": "Other"})
    wiki.write("wiki/literature/b.md", "unrelated body", {"title": "Diffusion models"})
    wiki.write("wiki/venue/c.md", "nothing relevant")
    hits = wiki.search("diffusion", k=5)
    assert [h.path for h in hits] == ["wiki/literature/b.md", "wiki/literature/a.md"]
    assert wiki.search("diffusion", prefix="wiki/venue") == []
    assert wiki.search("the of and") == []


def test_notes_are_unique_and_archivable(wiki):
    paths = set()

    def writer(i: int) -> None:
        paths.add(wiki.add_note(f"observation {i}", stage="refinement", agent="a", kind="lesson"))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(paths) == 20
    assert len(wiki.pending_notes()) == 20
    moved = wiki.archive_notes(sorted(paths)[:5], "dream 1")
    assert len(moved) == 5 and all(p.startswith("notes/consolidated/dream-1/") for p in moved)
    assert len(wiki.pending_notes()) == 15
    # Archived notes are not searched.
    assert all("consolidated" not in h.path for h in wiki.search("observation", k=50))


def test_index_keeps_standing_notes_and_is_idempotent(wiki):
    wiki.set_standing_notes("- Numbers come from [[sources/experimental_log]] only.")
    wiki.write("wiki/venue/rules.md", "x", {"summary": "page limit is 8"})
    first = wiki.rebuild_index()
    assert first == wiki.rebuild_index()
    assert first.count("## Standing notes") == 1
    assert "- Numbers come from [[sources/experimental_log]] only." in first
    assert "- [[venue/rules]]: page limit is 8" in first
    assert wiki.standing_notes() == "- Numbers come from [[sources/experimental_log]] only."


def test_index_survives_model_removing_markers(wiki):
    wiki.write("wiki/a/x.md", "x", {"summary": "s"})
    (wiki.root / INDEX_FILE).write_text("# Memory index: demo\n\n- keep this rule\n")
    text = wiki.rebuild_index()
    assert "- keep this rule" in text and CATALOG_START in text and "[[a/x]]" in text


def test_index_collapses_large_sections_to_stay_in_budget(tmp_path):
    wiki = Wiki(tmp_path, index_max_lines=40)
    for i in range(80):
        wiki.write(f"wiki/literature/p{i:02d}.md", "x", {"summary": f"paper {i}"})
    wiki.write("wiki/venue/rules.md", "x", {"summary": "rules"})
    text = wiki.rebuild_index()
    assert len(text.splitlines()) <= 40
    assert "**literature/**: 80 pages" in text and "[[venue/rules]]" in text
    index_page = wiki.read("wiki/literature/index.md")
    assert index_page.meta["auto"] is True
    assert len(index_page.links()) == 80
    report = wiki.lint()
    assert not report.index_over_budget and not report.broken_links
    # Shrinking the section removes the generated index again.
    for i in range(75):
        wiki.delete(f"wiki/literature/p{i:02d}.md")
    assert "### literature (5)" in wiki.rebuild_index()
    assert not wiki.exists("wiki/literature/index.md")


def test_injected_context_is_hard_capped(tmp_path):
    wiki = Wiki(tmp_path, index_max_lines=30, index_max_bytes=2_000)
    wiki.set_standing_notes("\n".join(f"- rule {i} " + "x" * 80 for i in range(100)))
    context = wiki.injected_context()
    assert len(context.encode()) <= 2_000 + 200
    assert "[index truncated" in context
    assert wiki.lint().index_over_budget


def test_link_for():
    assert link_for("wiki/literature/x.md") == "literature/x"
    assert link_for("notes/n.md") == "notes/n"


def test_generated_section_index_stays_within_the_page_limit(tmp_path):
    wiki = Wiki(tmp_path, max_page_bytes=6_000, collapse_over=10)
    for i in range(300):
        wiki.write(f"wiki/literature/Paper{i:03d}.md", "x", {"summary": "s" * 150})
    text = wiki.rebuild_index()  # must not raise even though the listing is long
    assert "**literature/**: 300 pages" in text
    listing = wiki.read_text("wiki/literature/index.md")
    assert len(listing.encode()) <= 6_000 and "more: search the wiki" in listing
    assert wiki.lint().broken_links == []


def test_open_existing_memory_without_creating_anything(tmp_path):
    viewer = Wiki(tmp_path / "absent", create=False)
    assert not (tmp_path / "absent").exists()
    assert viewer.list() == [] and viewer.standing_notes() == ""
    real = Wiki(tmp_path / "real")
    real.write("wiki/a/x.md", "body", {"summary": "s"})
    assert [i.path for i in Wiki(tmp_path / "real", create=False).list()] == ["wiki/a/x.md"]
