"""Markdown rendering for the web app: what a hostile page can and cannot produce."""

from __future__ import annotations

import difflib
import random
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest

pytest.importorskip("markdown_it")

from paper_orchestra.research import Project
from paper_orchestra.web.render import (
    diff_html,
    file_url,
    page_href,
    render_markdown,
    render_marked,
    unified_diff,
)

PID = "p-0123456789ab"

# ---------------------------------------------------------------------------
# What the renderer is allowed to write, checked on every output below
# ---------------------------------------------------------------------------

ALLOWED_ATTRIBUTES = {
    "a": {"href", "class", "rel", "target", "title", "data-page"},
    "span": {"class", "title", "data-target"},
    "img": {"src", "alt", "title", "loading"},
    "code": {"class"},
    "th": {"class"},
    "td": {"class"},
    "ol": {"start"},
    "ins": {"class"},
    "del": {"class"},
    "div": {"class"},
    "pre": {"class"},
    **{f"h{level}": {"id"} for level in range(1, 7)},
}
ALLOWED_TAGS = set(ALLOWED_ATTRIBUTES) | {
    "p", "ul", "li", "em", "strong", "s", "blockquote", "br", "hr",
    "table", "thead", "tbody", "tr",
}  # fmt: skip
SAFE_HREF = re.compile(
    r"^(https?://[^\s]+|mailto:[^\s]*|#[^\s]*|/api/projects/[^/\s]+/files/[^\s]+)$"
)
SAFE_SRC = re.compile(r"^/api/projects/[^/\s]+/files/(inputs|experiments|paper)/[^\s]+$")


class _Audit(HTMLParser):
    """Collects every tag and attribute of a rendered text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def audit(rendered: str) -> _Audit:
    """Parse ``rendered`` and fail on anything outside the renderer's allowance."""
    parser = _Audit()
    parser.feed(rendered)
    parser.close()
    for tag, attrs in parser.tags:
        assert tag in ALLOWED_TAGS, f"unexpected <{tag}> in {rendered!r}"
        extra = set(attrs) - ALLOWED_ATTRIBUTES.get(tag, set())
        assert not extra, f"unexpected attributes {extra} on <{tag}> in {rendered!r}"
        if "href" in attrs:
            assert SAFE_HREF.match(attrs["href"] or ""), f"unsafe href {attrs['href']!r}"
        if "src" in attrs:
            assert SAFE_SRC.match(attrs["src"] or ""), f"unsafe src {attrs['src']!r}"
    return parser


def tags_of(rendered: str, name: str) -> list[dict[str, str | None]]:
    return [attrs for tag, attrs in audit(rendered).tags if tag == name]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "proj", title="Warmup", brief="Does warmup matter?")
    project.create_page(
        "hypothesis", "Warmup reduces variance", "## Statement\n\nIt does.\n", author="human"
    )
    project.create_page("topic", "Adaptive optimisers", "About Adam.\n", author="human")
    return project


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

ATTACKS = [
    "<script>alert(1)</script>",
    "<SCRIPT SRC=//evil.example/x.js></SCRIPT>",
    "text <img src=x onerror=alert(1)> more",
    '<a href="javascript:alert(1)" onclick="alert(2)">x</a>',
    "<svg onload=alert(1)>",
    "<iframe src=javascript:alert(1)></iframe>",
    "<style>body{background:url(javascript:alert(1))}</style>",
    "<!-- comment --><b>bold</b>",
    "[x](javascript:alert(1))",
    "[x](JaVaScRiPt:alert(1))",
    "[x](  javascript:alert(1))",
    "[x](<javascript:alert(1)>)",
    "[x](java\tscript:alert(1))",
    "[x](&#106;avascript:alert(1))",
    "[x](&#x6A;avascript&colon;alert(1))",
    "[x](javascript&colon;alert(1))",
    "[x](vbscript:msgbox(1))",
    "[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
    "[x](file:///etc/passwd)",
    "[x](//evil.example/path)",
    "[x](/\\evil.example)",
    "[x](\\\\evil.example)",
    "[x](http:evil)",
    "[x](https:/evil)",
    '[x](http://a.example/" onmouseover="alert(1))',
    '[x](http://a.example "title\\" onmouseover=\\"alert(1)")',
    "[x][ref]\n\n[ref]: javascript:alert(1)",
    "[x][ref]\n\n[ref]: data:text/html,<script>alert(1)</script>",
    "<javascript:alert(1)>",
    "<data:text/html,hi>",
    "![x](javascript:alert(1))",
    "![x](data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9YWxlcnQoMSk+)",
    "![x](data:image/png;base64,iVBORw0KGgo=)",
    "![x](https://tracker.example/pixel.png)",
    "![x](//tracker.example/pixel.png)",
    "![x](inputs/../../../etc/passwd)",
    "![x](/api/projects/other-project/files/inputs/a.png)",
    '![x" onerror="alert(1)](inputs/a.png)',
    '![x](inputs/a.png "t\\" onload=\\"alert(1)")',
    "[[<script>alert(1)</script>]]",
    "[[target|<img src=x onerror=alert(1)>]]",
    '[[target" onclick="alert(1)]]',
    "[[a|[[b]]]]",
    "[[[[a]]]]",
    "[[a]] [[b|c]] [[d#e|f]] [[ ]] [[|]]",
    "[link [[inner]] text](https://example.org)",
    "&lt;script&gt;alert(1)&lt;/script&gt;",
    "&#60;script&#62;alert(1)&#60;/script&#62;",
    "&amp;lt;script&amp;gt;",
    "```\n<script>alert(1)</script>\n```",
    '```"><script>alert(1)</script>\ncode\n```',
    "```py onclick=alert(1)\ncode\n```",
    "    <script>indented()</script>",
    "`<script>inline()</script>`",
    "| a | b |\n|---|---|\n| <script>x</script> | [y](javascript:z) |",
    "# heading <script>x</script>\n\n## heading <script>x</script>",
    "> quote <img src=x onerror=alert(1)>",
    "1. item <b onmouseover=alert(1)>x</b>",
    'https://example.org/a?b=<script>&c="d"',
    "http://a.example/'onmouseover='alert(1)",
    "\x00null byte <script>",
    "[x](http://a.example/\x01\x02)",
]


@pytest.mark.parametrize("source", ATTACKS)
def test_hostile_markdown_yields_only_allowed_markup(source):
    # ``audit`` parses the result as a browser would: an event handler, a script or an
    # address outside the rules would show up as a tag or an attribute it does not allow.
    rendered = render_markdown(source, pid=PID)
    parser = audit(rendered)
    assert not any(name.startswith("on") for _, attrs in parser.tags for name in attrs)
    assert "<script" not in rendered.lower()


@pytest.mark.parametrize("source", ATTACKS)
def test_hostile_markdown_in_a_suggested_edit_is_just_as_contained(source):
    for old, new in ((source, "plain text"), ("plain text", source), (source, source + " more")):
        rendered = render_marked(old, new, pid=PID).html
        audit(rendered)
        assert "<script" not in rendered.lower()


def test_raw_html_is_shown_as_text():
    rendered = render_markdown("before <b>bold</b> <script>alert(1)</script> after")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&lt;b&gt;bold&lt;/b&gt;" in rendered
    assert [tag for tag, _ in audit(rendered).tags] == ["p"]


def test_a_script_link_never_becomes_an_address():
    for source in ("[x](javascript:alert(1))", "[x](&#106;avascript:alert(1))", "<javascript:x>"):
        assert tags_of(render_markdown(source), "a") == []


def test_entities_are_text_not_markup():
    rendered = render_markdown("&lt;script&gt; &amp; &#60;b&#62; &copy;")
    assert [tag for tag, _ in audit(rendered).tags] == ["p"]
    assert "".join(audit(rendered).text).strip() == "<script> & <b> ©"


def test_web_links_open_outside_the_app_without_a_referrer():
    (link,) = tags_of(render_markdown('[paper](https://arxiv.org/abs/1412.6980 "Adam")'), "a")
    assert link == {
        "href": "https://arxiv.org/abs/1412.6980",
        "class": "external",
        "rel": "noopener noreferrer nofollow",
        "target": "_blank",
        "title": "Adam",
    }
    (mail,) = tags_of(render_markdown("[write](mailto:someone@example.org)"), "a")
    assert mail == {"href": "mailto:someone@example.org"}
    (inside,) = tags_of(render_markdown(f"[inbox](#/p/{PID}/inbox)"), "a")
    assert inside == {"href": f"#/p/{PID}/inbox"}


def test_addresses_written_out_in_the_text_become_links():
    rendered = render_markdown("See https://example.org/a_b?c=1. And (http://x.example/y).")
    assert [link["href"] for link in tags_of(rendered, "a")] == [
        "https://example.org/a_b?c=1",
        "http://x.example/y",
    ]
    rendered = render_markdown("(see https://en.wikipedia.org/wiki/Adam_(optimizer)).")
    assert [link["href"] for link in tags_of(rendered, "a")] == [
        "https://en.wikipedia.org/wiki/Adam_(optimizer)"
    ]
    # Not inside a link that is already there, and not in code.
    rendered = render_markdown("[https://a.example](https://b.example) `https://c.example`")
    assert [link["href"] for link in tags_of(rendered, "a")] == ["https://b.example"]


def test_other_addresses_are_shown_as_text_with_the_address_in_reach():
    rendered = render_markdown("[ftp](ftp://example.org/file) [rel](some/where.html)", pid=PID)
    assert tags_of(rendered, "a") == []
    assert [span["class"] for span in tags_of(rendered, "span")] == ["link-blocked"] * 2
    assert tags_of(rendered, "span")[0]["title"] == "ftp://example.org/file"
    assert "ftp" in "".join(audit(rendered).text)


def test_fenced_code_is_kept_as_text():
    rendered = render_markdown("```python\nprint('<b>') # [[H1]] https://x.example\n```\n")
    assert rendered == (
        '<pre><code class="language-python">'
        "print('&lt;b&gt;') # [[H1]] https://x.example\n</code></pre>\n"
    )
    assert tags_of(render_markdown("```\nplain\n```"), "code") == [{}]


def test_tables_are_rendered_and_alignment_is_a_class():
    rendered = render_markdown("| metric | value |\n|:--|--:|\n| acc | 0.91 |\n")
    assert [tag for tag, _ in audit(rendered).tags][:3] == ["table", "thead", "tr"]
    assert tags_of(rendered, "th") == [{"class": "align-left"}, {"class": "align-right"}]
    assert "style" not in rendered


def test_headings_get_stable_unique_ids():
    source = "# Results\n\n## Limits\n\n## Limits\n\n### `code` & <b>\n"
    rendered = render_markdown(source)
    ids = [attrs["id"] for tag, attrs in audit(rendered).tags if tag.startswith("h")]
    assert ids == ["h-results", "h-limits", "h-limits-2", "h-code-b"]
    assert render_markdown(source) == rendered
    assert "id=" not in render_markdown(source, heading_ids=False)


# ---------------------------------------------------------------------------
# Wiki links
# ---------------------------------------------------------------------------


def test_wiki_links_resolve_through_the_project(project):
    rendered = render_markdown(
        "See [[H1]], [[hypotheses/H1-warmup-reduces-variance|the hypothesis]], "
        "[[topics/adaptive-optimisers#bias|Adam]] and [[nowhere|a gap]].",
        project=project,
    )
    path = "wiki/hypotheses/H1-warmup-reduces-variance.md"
    links = tags_of(rendered, "a")
    assert links[0] == {"href": page_href(project.id, path), "class": "wikilink", "data-page": path}
    assert (
        links[0]["href"]
        == f"#/p/{project.id}/wiki/wiki%2Fhypotheses%2FH1-warmup-reduces-variance.md"
    )
    assert links[1]["data-page"] == path
    assert links[2]["data-page"] == "wiki/topics/adaptive-optimisers.md"
    assert tags_of(rendered, "span") == [
        {
            "class": "wikilink wikilink-missing",
            "data-target": "nowhere",
            "title": "No page matches nowhere",
        }
    ]
    assert "the hypothesis" in rendered and "a gap" in rendered


def test_wiki_link_text_cannot_carry_markup(project):
    rendered = render_markdown(
        '[[H1|<img src=x onerror=alert(1)>]] [[x" onclick="y]]', project=project
    )
    parser = audit(rendered)
    assert [tag for tag, _ in parser.tags] == ["p", "a", "span"]
    assert "<img src=x onerror=alert(1)>" in "".join(parser.text)
    assert tags_of(rendered, "span")[0]["data-target"] == 'x" onclick="y'


def test_nested_wiki_link_syntax_links_the_inner_target_only(project):
    rendered = render_markdown("[[a|[[H1]]]] and [[[[H1]]]]", project=project)
    assert [link["data-page"] for link in tags_of(rendered, "a")] == [
        "wiki/hypotheses/H1-warmup-reduces-variance.md"
    ] * 2
    assert "[[a|" in "".join(audit(rendered).text)


def test_a_wiki_link_inside_a_link_does_not_nest(project):
    # As with Markdown links: the inner one is the link, the brackets around it are text.
    rendered = render_markdown("[see [[H1]] here](https://example.org)", project=project)
    assert [link.get("data-page") for link in tags_of(rendered, "a")] == [
        "wiki/hypotheses/H1-warmup-reduces-variance.md",
        None,
    ]
    assert not re.search(r"<a\b[^>]*>(?:(?!</a>).)*<a\b", rendered, re.DOTALL)


def test_an_address_in_a_wiki_links_label_is_not_a_second_link(project):
    rendered = render_markdown(
        "[[H1|see https://example.org/x]] [[nowhere|https://example.org]]", project=project
    )
    assert len(tags_of(rendered, "a")) == 1 and len(tags_of(rendered, "span")) == 1
    assert not re.search(r"<a\b[^>]*>(?:(?!</a>).)*<a\b", rendered, re.DOTALL)


def test_without_a_project_wiki_links_are_marked_unresolved():
    rendered = render_markdown("[[H1]]")
    assert tags_of(rendered, "a") == []
    assert tags_of(rendered, "span")[0]["class"] == "wikilink wikilink-missing"


def test_markdown_links_to_pages_and_files_stay_inside_the_app(project):
    rendered = render_markdown(
        "[hyp](../hypotheses/H1-warmup-reduces-variance.md) [by id](H1) "
        "[data](inputs/table.csv) [state](state.json)",
        project=project,
        page_path="wiki/topics/adaptive-optimisers.md",
    )
    links = tags_of(rendered, "a")
    assert [link.get("data-page") for link in links[:2]] == [
        "wiki/hypotheses/H1-warmup-reduces-variance.md"
    ] * 2
    assert links[2]["href"] == file_url(project.id, "inputs/table.csv")
    assert links[2]["class"] == "file-link"
    assert len(links) == 3  # state.json is not something a page links to


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def test_images_come_only_from_the_projects_files(project):
    artifact = "experiments/E1/runs/1/artifacts/loss curve.png"
    rendered = render_markdown(
        f"![curve](<{artifact}>) ![same](/files/{artifact.replace(' ', '%20')}) "
        f'![up](../../{artifact.replace(" ", "%20")}) ![in](inputs/photo.jpg "A photo")',
        project=project,
        page_path="wiki/results/E1-run1.md",
    )
    images = tags_of(rendered, "img")
    expected = f"/api/projects/{project.id}/files/experiments/E1/runs/1/artifacts/loss%20curve.png"
    assert [image["src"] for image in images[:3]] == [expected] * 3
    assert images[0]["alt"] == "curve" and images[0]["loading"] == "lazy"
    assert images[3] == {
        "src": f"/api/projects/{project.id}/files/inputs/photo.jpg",
        "alt": "in",
        "loading": "lazy",
        "title": "A photo",
    }


def test_a_remote_image_is_a_link_and_nothing_is_fetched():
    rendered = render_markdown("![chart](https://example.org/chart.png)", pid=PID)
    assert tags_of(rendered, "img") == []
    (link,) = tags_of(rendered, "a")
    assert link["href"] == "https://example.org/chart.png"
    assert link["class"] == "external image-link"
    assert "chart" in "".join(audit(rendered).text)


@pytest.mark.parametrize(
    "address",
    [
        "data:image/png;base64,iVBORw0KGgo=",
        "state.json",
        "wiki/idea.md",
        "../outside.png",
        "inputs/../../outside.png",
        "inputs/%2e%2e/%2e%2e/outside.png",
        "inputs%5c..%5coutside.png",
        "%2e%2e/inputs/a.png",
        "/api/projects/p-other/files/inputs/a.png",
        "//example.org/a.png",
    ],
)
def test_other_images_are_dropped_to_their_text(address):
    rendered = render_markdown(f"![the label]({address})", pid=PID)
    assert tags_of(rendered, "img") == []
    assert tags_of(rendered, "a") == []
    assert "the label" in rendered


def test_no_image_is_loaded_without_a_project():
    assert tags_of(render_markdown("![x](inputs/a.png)"), "img") == []


# ---------------------------------------------------------------------------
# Suggested edits read in place
# ---------------------------------------------------------------------------


def test_changed_words_are_marked_inside_the_block():
    marked = render_marked(
        "# Title\n\nWarmup helps at small batch sizes.\n\nUnchanged paragraph.\n",
        "# Title\n\nWarmup helps most at large batch sizes.\n\nUnchanged paragraph.\n",
    )
    assert marked.html == (
        "<h1>Title</h1>\n"
        '<div class="diff-changed"><p>Warmup helps <ins>most</ins> at '
        "<del>small</del><ins>large</ins> batch sizes.</p>\n</div>\n"
        "<p>Unchanged paragraph.</p>\n"
    )
    assert (marked.words_added, marked.words_removed) == (2, 1)


def test_added_and_removed_blocks_are_marked_whole():
    marked = render_marked("Keep.\n\nDrop this one.\n", "Keep.\n\n- a new list\n")
    assert marked.html == (
        "<p>Keep.</p>\n"
        '<del class="diff-block"><p>Drop this one.</p>\n</del>\n'
        '<ins class="diff-block"><ul>\n<li>a new list</li>\n</ul>\n</ins>\n'
    )
    assert (marked.words_added, marked.words_removed) == (3, 3)


def test_an_edit_behind_an_inserted_paragraph_is_still_read_word_by_word():
    marked = render_marked(
        "Alpha beta gamma delta.\n\nThe second paragraph stays mostly as it was.\n",
        "A new opening.\n\nAlpha beta gamma epsilon.\n\nThe second paragraph stays as it was.\n",
    )
    assert marked.html.count('<ins class="diff-block">') == 1
    assert "<del>delta</del><ins>epsilon</ins>" in marked.html
    assert "stays <del>mostly</del> as it was" in marked.html


def test_marks_never_cross_a_tag_and_formatting_survives(project):
    marked = render_marked(
        "The effect is **large** in [[H1]] and small elsewhere.\n",
        "The effect is **very large** in [[H1]] and absent elsewhere.\n",
        project=project,
    )
    audit(marked.html)
    assert "<strong><ins>very</ins> large</strong>" in marked.html
    assert 'class="wikilink"' in marked.html
    assert "<del>small</del><ins>absent</ins>" in marked.html


def test_deleted_text_is_escaped_text(project):
    marked = render_marked(
        "Old claim <script>alert(1)</script> with [x](javascript:alert(2)) here.\n",
        "New claim here.\n",
    )
    audit(marked.html)
    assert "<script" not in marked.html
    assert "&lt;script&gt;" in marked.html


def test_list_items_and_table_cells_are_marked_in_place():
    marked = render_marked(
        "- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
        "- one\n- three\n\n| a | b |\n|---|---|\n| 1 | 5 |\n",
    )
    audit(marked.html)
    assert "<li><del>two</del><ins>three</ins></li>" in marked.html
    assert "<td><del>2</del><ins>5</ins></td>" in marked.html
    # Nothing was put where a list or a table row allows no text.
    assert not re.search(r"</li>\s*<(ins|del)", marked.html)
    assert not re.search(r"</td>\s*<(ins|del)", marked.html)


def _prose(words: int, seed: int) -> list[str]:
    """Words drawn so that many repeat, as in real text."""
    rng = random.Random(seed)
    return [f"w{rng.randrange(400)}" for _ in range(words)]


@pytest.fixture
def spent(monkeypatch):
    """What every comparison of a render took from its budget."""
    from paper_orchestra.web import render

    taken: list[int] = []
    take = render._Budget.take

    def recording(self, pairs):
        granted = take(self, pairs)
        if granted:
            taken.append(pairs)
        return granted

    monkeypatch.setattr(render._Budget, "take", recording)
    return taken


def test_one_edit_in_a_very_long_block_is_cheap_and_still_marked_by_word(spent):
    # A table or a paragraph of tens of thousands of words with one word changed: what
    # the two versions share before and after the edit is not compared word against word.
    words = _prose(30_000, seed=1)
    changed = [*words[:17_000], "CHANGED", *words[17_001:]]
    marked = render_marked(" ".join(words) + "\n", " ".join(changed) + "\n")
    assert f"<del>{words[17_000]}</del><ins>CHANGED</ins>" in marked.html
    assert marked.html.count("<ins>") == 1 and marked.html.count("<del>") == 1
    assert (marked.words_added, marked.words_removed) == (1, 1)
    assert sum(spent) <= 10


def test_a_very_long_block_edited_throughout_is_shown_replaced_not_compared_for_minutes(spent):
    # Comparing two such blocks word by word takes difflib tens of seconds, and every look
    # at the page would pay it again.  Past the budget the block reads as removed and added.
    from paper_orchestra.web.render import _MAX_COMPARED_PAIRS

    words = _prose(30_000, seed=2)
    edited = [f"EDIT{n}" if n % 40 == 0 else word for n, word in enumerate(words)]
    marked = render_marked(" ".join(words) + "\n", " ".join(edited) + "\n")
    audit(marked.html)
    assert marked.html.count('<del class="diff-block">') == 1
    assert marked.html.count('<ins class="diff-block">') == 1
    assert "diff-changed" not in marked.html
    assert (marked.words_added, marked.words_removed) == (30_000, 30_000)
    assert sum(spent) <= _MAX_COMPARED_PAIRS


def test_many_long_changed_blocks_are_each_read_against_their_earlier_version(spent):
    # 40 long paragraphs, each lightly edited, behind a new opening one.  Comparing every
    # old paragraph with every new one to pair them would take seconds; they are paired
    # by the words they hold, and each is then marked word by word.
    from paper_orchestra.web.render import _MAX_COMPARED_PAIRS

    paragraphs = [_prose(700, seed=10 + n) for n in range(40)]
    edited = [
        [f"EDIT{n}" if n % 25 == 0 else word for n, word in enumerate(paragraph)]
        for paragraph in paragraphs
    ]
    old = "\n\n".join(" ".join(paragraph) for paragraph in paragraphs) + "\n"
    new = "A new opening.\n\n" + "\n\n".join(" ".join(paragraph) for paragraph in edited) + "\n"
    marked = render_marked(old, new)
    audit(marked.html)
    assert marked.html.count('<div class="diff-changed">') == 40
    assert marked.html.count('<ins class="diff-block">') == 1
    assert '<del class="diff-block">' not in marked.html
    assert marked.words_removed == 40 * 28 and marked.words_added == 40 * 28 + 3
    assert sum(spent) <= _MAX_COMPARED_PAIRS / 2


def test_blocks_that_only_share_their_words_are_not_read_as_versions_of_each_other():
    # Paired by the words they hold, two paragraphs of the same words in another order
    # look alike.  They must not come out as one paragraph with every other word struck.
    paragraphs = [_prose(700, seed=100 + n) for n in range(40)]
    shuffled = []
    for n, paragraph in enumerate(paragraphs):
        other = list(paragraph)
        random.Random(n).shuffle(other)
        shuffled.append(other)
    old = "\n\n".join(" ".join(paragraph) for paragraph in paragraphs) + "\n"
    new = "\n\n".join(" ".join(paragraph) for paragraph in shuffled) + "\n"
    marked = render_marked(old, new)
    audit(marked.html)
    assert "diff-changed" not in marked.html
    assert marked.html.count('<del class="diff-block">') == 40
    assert marked.html.count('<ins class="diff-block">') == 40


def test_a_text_of_thousands_of_blocks_that_differ_throughout_is_still_rendered(spent):
    from paper_orchestra.web.render import _MAX_COMPARED_PAIRS

    rng = random.Random(9)
    old = "\n\n".join(f"p{rng.randrange(60)}" for _ in range(6000)) + "\n"
    new = "\n\n".join(f"p{rng.randrange(60)}" for _ in range(6000)) + "\n"
    marked = render_marked(old, new)
    audit(marked.html)
    assert marked.words_added > 0 and marked.words_removed > 0
    assert sum(spent) <= 2 * _MAX_COMPARED_PAIRS


def test_an_unchanged_text_is_rendered_unmarked():
    text = "# A\n\nSame [link][r].\n\n[r]: https://example.org\n"
    marked = render_marked(text, text)
    assert "<ins" not in marked.html and "<del" not in marked.html
    assert 'href="https://example.org"' in marked.html
    assert (marked.words_added, marked.words_removed) == (0, 0)


def test_link_definitions_of_the_whole_text_reach_every_block():
    marked = render_marked(
        "Read [the paper][p].\n\n[p]: https://example.org/old\n",
        "Read [the paper][p] twice.\n\n[p]: https://example.org/new\n",
    )
    assert 'href="https://example.org/new"' in marked.html


# ---------------------------------------------------------------------------
# Unified diffs
# ---------------------------------------------------------------------------


def test_unified_diff_by_line_and_its_html():
    diff = unified_diff("a\nb\nc\n", "a\n<b>\nc\n")
    assert diff.splitlines()[:2] == ["--- current", "+++ proposed"]
    assert "-b" in diff.splitlines() and "+<b>" in diff.splitlines()
    rendered = diff_html(diff)
    assert '<span class="diff-line diff-add">+&lt;b&gt;</span>' in rendered
    assert '<span class="diff-line diff-del">-b</span>' in rendered
    assert rendered.startswith('<pre class="diff">') and "<b>" not in rendered
    assert unified_diff("same\n", "same\n") == ""
    assert diff_html("") == '<pre class="diff"></pre>'


def test_the_head_of_a_diff_is_said_in_words_where_a_person_reads_it():
    old = "\n".join(f"line {number}" for number in range(1, 41)) + "\n"
    new = old.replace("line 30\n", "line 30, changed\nand one more\n")
    diff = unified_diff(
        old, new, old_label="the page as it is", new_label="as the analyst suggests it"
    )
    # The text stays a unified diff, for whoever applies or copies it.
    assert diff.splitlines()[:3] == [
        "--- the page as it is",
        "+++ as the analyst suggests it",
        "@@ -27,7 +27,8 @@",
    ]

    rendered = diff_html(diff)

    # What the person is shown says which side is which and where the change is, without
    # the marks of a patch file.
    assert '<span class="diff-line diff-file">− the page as it is</span>' in rendered
    assert '<span class="diff-line diff-file">+ as the analyst suggests it</span>' in rendered
    assert '<span class="diff-line diff-hunk">From line 27</span>' in rendered
    assert "@@" not in rendered and "---" not in rendered and "+++" not in rendered
    assert '<span class="diff-line diff-del">-line 30</span>' in rendered
    # Text put before the first line has no line to start from.
    top = diff_html(unified_diff("", "new\n"))
    assert '<span class="diff-line diff-hunk">At the top</span>' in top
    # A line of the text that merely looks like a mark of the format stays the text.
    inner = diff_html(unified_diff("a\n", "a\n--- not a head\n@@ not a hunk @@\n"))
    assert '<span class="diff-line diff-add">+--- not a head</span>' in inner
    assert '<span class="diff-line diff-add">+@@ not a hunk @@</span>' in inner


def _patched(old: str, diff: str) -> str:
    """``old`` with a unified diff applied, checking every line the diff says is there."""
    lines = old.splitlines()
    out: list[str] = []
    position = 0
    for line in diff.split("\n")[2:]:
        if line.startswith("@@"):
            start = int(line.split()[1].lstrip("-").split(",")[0])
            count = line.split()[1].split(",")[1] if "," in line.split()[1] else "1"
            start = start if int(count) else start + 1  # "-N,0" names the line before
            out += lines[position : start - 1]
            position = start - 1
        elif line.startswith("+"):
            out.append(line[1:])
        else:
            assert lines[position] == line[1:], (position, line)
            position += 1
            if line.startswith(" "):
                out.append(line[1:])
    return "\n".join(out + lines[position:])


def test_a_diff_of_very_many_lines_is_one_hunk_instead_of_minutes_of_comparing():
    # 20000 short lines, every eleventh changed: difflib takes a quarter of a minute for
    # these, and the inbox asks for the diff of every proposal.
    rng = random.Random(4)
    old_lines = [f"{rng.randrange(150):x}" for _ in range(20_000)]
    new_lines = [
        f"{rng.randrange(150):x}" if n % 11 == 5 else line for n, line in enumerate(old_lines)
    ]
    old, new = "\n".join(old_lines), "\n".join(new_lines)
    diff = unified_diff(old, new)
    assert diff.count("\n@@ ") == 1 and diff.startswith("--- current\n+++ proposed\n@@ -")
    assert _patched(old, diff) == new
    # What the two versions share before and after the change is not part of the hunk.
    assert diff.count("\n") < 2 * 20_000 + 12

    # One line changed in such a text is still one line in the diff.
    one = list(old_lines)
    one[12_345] = "changed"
    small = unified_diff(old, "\n".join(one))
    assert small.count("\n") == 2 + 1 + 3 + 2 + 3 - 1 and _patched(old, small) == "\n".join(one)


def test_changes_far_apart_in_a_long_text_are_still_read_line_by_line():
    old_lines = [f"Line {n} of a long page, with words of its own." for n in range(20_000)]
    new_lines = list(old_lines)
    new_lines[100], new_lines[15_000] = "changed", "changed too"
    del new_lines[19_000]
    old, new = "\n".join(old_lines), "\n".join(new_lines)
    diff = unified_diff(old, new)
    assert diff.count("\n@@ ") == 3 and _patched(old, diff) == new
    assert diff.count("\n") < 40


def test_the_diff_is_the_one_difflib_writes():
    rng = random.Random(8)
    for _ in range(300):
        old_lines = [rng.choice("abcdef") for _ in range(rng.randrange(0, 30))]
        new_lines = [rng.choice("abcdef") for _ in range(rng.randrange(0, 30))]
        expected = "\n".join(
            difflib.unified_diff(
                old_lines, new_lines, fromfile="current", tofile="proposed", n=2, lineterm=""
            )
        )
        got = unified_diff("\n".join(old_lines), "\n".join(new_lines), context=2)
        assert got == expected, (old_lines, new_lines)
        assert _patched("\n".join(old_lines), got) == "\n".join(new_lines)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("a\nb\nc\nd\ne\nf\ng\nh\ni", "a\nb\nc\nd\nX\nY\ng\nh\ni"),
        ("a\nb", "X\nY\nZ"),
        ("a\nb\nc", "a\nb\nc\nd\ne"),
        ("a\nb\nc\nd", "c\nd"),
        ("a\nb\nc", "a\nc"),
        ("", "a\nb"),
        ("a\nb", ""),
        ("a", "b"),
    ],
)
def test_the_one_hunk_diff_is_a_diff_that_applies(old, new, monkeypatch):
    from paper_orchestra.web import render

    monkeypatch.setattr(render, "_MAX_COMPARED_PAIRS", 0)  # every comparison is given up
    diff = unified_diff(old, new)
    assert diff.count("@@ -") == 1
    assert _patched(old, diff) == new
    monkeypatch.undo()
    assert _patched(old, unified_diff(old, new)) == new  # and so does difflib's


def test_without_a_budget_a_suggested_edit_reads_as_removed_and_added(monkeypatch):
    from paper_orchestra.web import render

    monkeypatch.setattr(render, "_MAX_COMPARED_PAIRS", 0)
    marked = render_marked("Same.\n\nOne two three.\n\nEnd.\n", "Same.\n\nOne 2 three.\n\nEnd.\n")
    assert marked.html == (
        "<p>Same.</p>\n"
        '<del class="diff-block"><p>One two three.</p>\n</del>\n'
        '<ins class="diff-block"><p>One 2 three.</p>\n</ins>\n'
        "<p>End.</p>\n"
    )


# ---------------------------------------------------------------------------
# Rendering time
# ---------------------------------------------------------------------------


def _trimmed_one_by_one(address: str) -> str:
    """What "without the sentence's punctuation" means, spelled out one character at a
    time: the definition the fast version is compared with."""
    closing = {")": "(", "]": "[", "}": "{"}
    while address:
        last = address[-1]
        if last in ".,;:!?'\"" or (
            last in closing and address.count(last) > address.count(closing[last])
        ):
            address = address[:-1]
        else:
            break
    return address


def test_the_end_of_an_address_is_found_as_before():
    from paper_orchestra.web.render import _without_trailing_punctuation

    for address, expected in [
        ("http://x.example/y).", "http://x.example/y"),
        (
            "https://en.wikipedia.org/wiki/Adam_(optimizer)).",
            "https://en.wikipedia.org/wiki/Adam_(optimizer)",
        ),
        ("http://x.example/[a]]", "http://x.example/[a]"),
        ("http://x.example/a.)!", "http://x.example/a"),
        ("http://x.example/a(", "http://x.example/a("),
        ("http://x.example/{a}}};", "http://x.example/{a}"),
        ("...", ""),
        ("", ""),
    ]:
        assert _without_trailing_punctuation(address) == expected, address
    rng = random.Random(7)
    for _ in range(2000):
        address = "http://a/" + "".join(
            rng.choice("ab()[]{}.,;:!?'\"") for _ in range(rng.randrange(30))
        )
        assert _without_trailing_punctuation(address) == _trimmed_one_by_one(address), address


@pytest.mark.parametrize("tail", [")", ")]}", ".)"])
def test_an_address_followed_by_a_long_run_of_punctuation_renders_in_linear_time(tail):
    # A page is a model's text or copied from the web, and is rendered at every look.
    # One character at a time, with a count and a copy for each, this text took minutes;
    # the bound is far above what it takes now, on a machine that is busy with other work.
    text = "See http://a.example/x" + tail * (600_000 // len(tail))
    before = time.perf_counter()
    rendered = render_markdown(text)
    assert time.perf_counter() - before < 20.0
    assert [link["href"] for link in tags_of(rendered, "a")] == ["http://a.example/x"]


def test_a_long_run_of_full_stops_after_an_address_is_not_copied_once_per_stop():
    from paper_orchestra.web.render import _without_trailing_punctuation

    address = "http://a.example/x" + "." * 2_000_000
    before = time.perf_counter()
    assert _without_trailing_punctuation(address) == "http://a.example/x"
    assert time.perf_counter() - before < 20.0
