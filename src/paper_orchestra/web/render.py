"""Markdown to HTML for the web app, on the server.

Everything a wiki page contains is untrusted: models wrote it, and some of it
was copied from the web.  The HTML produced here is the only text from a
project that the frontend inserts as markup, so the guarantee is made here and
nowhere else:

* raw HTML in the source is shown as text, never passed through;
* a link leads to ``http``, ``https`` or ``mailto``, into the app (``#/...``), or
  to a page or file of the same project; anything else is shown as plain text;
* an image comes from the project's own files endpoint or is not loaded at all
  (a remote image becomes a link: the browser fetches nothing by itself);
* every attribute value is written by this module or escaped by the renderer.

Wiki links ``[[target|label]]`` become in-app links to the page
``Project.resolve`` finds, and are marked when it finds none.

:func:`render_marked` shows a suggested edit in place: the page as it would
read, with inserted and deleted words marked.  The Markdown is compared block
by block, each block is rendered by the same renderer, and only then are
``<ins>`` and ``<del>`` put around escaped text, so page text cannot become
markup on that path either.
"""

from __future__ import annotations

import difflib
import html
import posixpath
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token

# The same pattern decides what the graph counts as a link (``Page.links``); a copy
# here could drift from it and show links the graph does not have.
from paper_orchestra.memory.wiki import WIKILINK, slugify
from paper_orchestra.web.security import FILE_AREAS

if TYPE_CHECKING:
    from paper_orchestra.research.project import Project

__all__ = [
    "Marked",
    "diff_html",
    "file_url",
    "page_href",
    "render_markdown",
    "render_marked",
    "unified_diff",
]

HEADING_ID_PREFIX = "h-"
_ENV_KEY = "paper_orchestra"

_SCHEME = re.compile(r"^([a-z][a-z0-9+.\-]*):")
_WEB_URL = re.compile(r"^https?://[^\s/?#\\]+", re.IGNORECASE)
_BARE_URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_FENCE_LANGUAGE = re.compile(r"[A-Za-z0-9_+#.\-]+")
_TEXT_ALIGN = re.compile(r"^text-align:(left|right|center)$")
_API_FILES = re.compile(r"^/?api/projects/([^/]+)/files/(.+)$")


def page_href(pid: str, path: str) -> str:
    """The in-app address of a wiki page: ``#/p/<pid>/wiki/<path as one segment>``."""
    return f"#/p/{quote(pid, safe='')}/wiki/{quote(path, safe='')}"


def file_url(pid: str, path: str) -> str:
    """The address of a project file on the files endpoint."""
    return f"/api/projects/{quote(pid, safe='')}/files/{quote(path, safe='/')}"


# ---------------------------------------------------------------------------
# What a render knows about its project
# ---------------------------------------------------------------------------


class _Context:
    """The project a text belongs to, for resolving links; one per render."""

    def __init__(
        self,
        project: Project | None,
        pid: str | None,
        page_path: str | None,
        heading_ids: bool,
    ) -> None:
        self.project = project
        self.pid = pid or (project.id if project is not None else None)
        self.folder = posixpath.dirname(page_path) if page_path else ""
        self.heading_ids = heading_ids
        self.used_ids: dict[str, int] = {}
        self._pages: dict[str, str | None] = {}

    def page(self, target: str) -> str | None:
        """Path of the page ``target`` names, or ``None``."""
        if self.project is None:
            return None
        if target not in self._pages:
            self._pages[target] = self.project.resolve(target)
        return self._pages[target]

    def project_path(self, reference: str) -> str | None:
        """A relative reference as a project-relative path, or ``None`` when it leaves
        the project.  ``./`` and ``../`` are read from the page's own directory, as a
        plain Markdown viewer would; everything else from the project root."""
        # The parser has percent-encoded the address; the checks below are made on the
        # path it stands for (``%2e%2e`` climbs like ``..``).
        reference = unquote(reference)
        match = _API_FILES.match(reference)
        if match:
            if self.pid is None or match.group(1) != self.pid:
                return None
            reference = match.group(2)
        elif reference.startswith(("./", "../")):
            reference = posixpath.join(self.folder, reference)
        else:
            reference = reference.lstrip("/").removeprefix("files/")
        path = posixpath.normpath(reference)
        if path.startswith(("../", "/")) or path in ("..", ".", ""):
            return None
        if "\\" in path or any(ord(character) < 32 for character in path):
            return None
        return path

    def heading_id(self, text: str) -> str:
        slug = slugify(text, 60)
        count = self.used_ids.get(slug, 0) + 1
        self.used_ids[slug] = count
        return f"{HEADING_ID_PREFIX}{slug}" + (f"-{count}" if count > 1 else "")


def _context(env: dict[str, Any]) -> _Context:
    return env.get(_ENV_KEY) or _Context(None, None, None, False)


# ---------------------------------------------------------------------------
# Links and images: one decision for every address in a page
# ---------------------------------------------------------------------------


def _visible(address: str) -> str:
    """The address as a browser reads it: without the blanks and control characters it
    drops before deciding what kind of address this is."""
    return "".join(ch for ch in address if ch > " " and ch != "\x7f")


def _link_target(address: str, context: _Context) -> dict[str, str] | None:
    """Attributes of the ``<a>`` for ``address``, or ``None`` when it may not be a link."""
    seen = _visible(address)
    lowered = seen.lower()
    if lowered.startswith("#"):
        return {"href": seen}
    scheme = _SCHEME.match(lowered)
    if scheme:
        if scheme.group(1) in ("http", "https") and _WEB_URL.match(seen):
            return {
                "href": seen,
                "class": "external",
                "rel": "noopener noreferrer nofollow",
                "target": "_blank",
            }
        if scheme.group(1) == "mailto":
            return {"href": seen}
        return None
    if lowered.startswith("//") or "\\" in seen or context.pid is None:
        return None  # another site without saying so, or nowhere to look the path up
    reference = seen.split("#", 1)[0].split("?", 1)[0]
    relative = context.project_path(reference)
    if relative is None:
        return None
    page = context.page(relative) or context.page(reference)
    if page is not None:
        return {"href": page_href(context.pid, page), "class": "wikilink", "data-page": page}
    if relative.split("/", 1)[0] in FILE_AREAS:
        return {
            "href": file_url(context.pid, relative),
            "class": "file-link",
            "rel": "noopener",
            "target": "_blank",
        }
    return None


def _image_source(address: str, context: _Context) -> str | None:
    """The files-endpoint address of an image of this project, or ``None``."""
    seen = _visible(address)
    if context.pid is None or _SCHEME.match(seen.lower()) or seen.startswith("//"):
        return None
    relative = context.project_path(seen.split("#", 1)[0].split("?", 1)[0])
    if relative is None or relative.split("/", 1)[0] not in FILE_AREAS:
        return None
    return file_url(context.pid, relative)


def _text_token(content: str) -> Token:
    token = Token("text", "", 0)
    token.content = content
    return token


def _anchor(attrs: dict[str, str], label: str) -> list[Token]:
    opening = Token("link_open", "a", 1)
    opening.attrs = dict(attrs)
    opening.meta = {"checked": True}
    return [opening, _text_token(label), Token("link_close", "a", -1)]


def _checked_image(token: Token, context: _Context) -> list[Token]:
    """The image when it is the project's own; else a link to it, or just its text."""
    address = str(token.attrGet("src") or "")
    label = "".join(child.content for child in token.children or [] if child.type == "text")
    source = _image_source(address, context)
    if source is not None:
        title = token.attrGet("title")
        token.attrs = {"src": source, "alt": "", "loading": "lazy"}
        if title:
            token.attrSet("title", str(title))
        return [token]
    target = _link_target(address, context)
    if target is not None and target.get("class") == "external":
        target["class"] = "external image-link"
        return _anchor(target, label or address)
    return [_text_token(label or address)]


_CLOSING = {")": "(", "]": "[", "}": "{"}
_SENTENCE_PUNCTUATION = frozenset(".,;:!?'\"")


def _without_trailing_punctuation(address: str) -> str:
    """An address found in running text, without the sentence's punctuation after it.
    A closing bracket stays when the address itself opened it (``.../Adam_(optimizer)``).

    The text is a model's or copied from the web, and a page is rendered at every look:
    the brackets are counted once and the end is found by walking back with those
    counts, so the time grows with the length of the address and not with its square.
    """
    counts = {bracket: address.count(bracket) for pair in _CLOSING.items() for bracket in pair}
    end = len(address)
    while end:
        last = address[end - 1]
        if last in _CLOSING and counts[last] > counts[_CLOSING[last]]:
            counts[last] -= 1  # what is left of the address has one fewer of them
        elif last not in _SENTENCE_PUNCTUATION:
            break
        end -= 1
    return address[:end]


def _linked_urls(text: str) -> list[Token]:
    """``text`` with the web addresses written out in it turned into links."""
    tokens: list[Token] = []
    position = 0
    for match in _BARE_URL.finditer(text):
        address = _without_trailing_punctuation(match.group(0))
        target = _link_target(address, _Context(None, None, None, False))
        if target is None:
            continue
        if match.start() > position:
            tokens.append(_text_token(text[position : match.start()]))
        tokens += _anchor(target, address)
        position = match.start() + len(address)
    if not tokens:
        return [_text_token(text)]
    if position < len(text):
        tokens.append(_text_token(text[position:]))
    return tokens


def _checked_children(children: Sequence[Token], context: _Context) -> list[Token]:
    """The inline tokens of one block with every link and image put through the rules."""
    checked: list[Token] = []
    blocked = 0  # > 0 inside a link that was turned into plain text
    inside_link = 0
    for token in children:
        if token.type == "link_open":
            inside_link += 1
            if token.meta.get("checked"):
                checked.append(token)
                continue
            address = str(token.attrGet("href") or "")
            target = _link_target(address, context)
            if target is None:
                blocked += 1
                token.tag = "span"
                token.attrs = {"class": "link-blocked", "title": address}
            else:
                title = token.attrGet("title")
                token.attrs = {**target, **({"title": str(title)} if title else {})}
            checked.append(token)
        elif token.type == "link_close":
            inside_link -= 1
            if blocked and token.tag == "a":
                blocked -= 1
                token.tag = "span"
            checked.append(token)
        elif token.type in ("wikilink_open", "wikilink_close"):
            inside_link += token.nesting
            checked.append(token)
        elif token.type == "image":
            checked += _checked_image(token, context)
        elif token.type == "text" and not inside_link:
            checked += _linked_urls(token.content)
        else:
            checked.append(token)
    return checked


def _finish(state: StateCore) -> None:
    """Core rule, run last: apply the link and image rules, name the headings, and
    replace the ``style`` attribute of aligned table cells by a class (the page's
    Content-Security-Policy does not allow inline styles)."""
    context = _context(state.env)
    tokens = state.tokens
    for index, token in enumerate(tokens):
        if token.type == "inline" and token.children is not None:
            token.children = _checked_children(token.children, context)
        elif token.type == "heading_open" and context.heading_ids:
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            text = "".join(
                child.content
                for child in (inline.children if inline is not None else None) or []
                if child.type in ("text", "code_inline")
            )
            token.attrSet("id", context.heading_id(text))
        elif token.type in ("th_open", "td_open"):
            align = _TEXT_ALIGN.match(str(token.attrs.pop("style", "")))
            if align:
                token.attrSet("class", f"align-{align.group(1)}")


# ---------------------------------------------------------------------------
# Wiki links
# ---------------------------------------------------------------------------


def _wikilink(state: StateInline, silent: bool) -> bool:
    """Inline rule: ``[[target]]``, ``[[target|label]]``, ``[[target#anchor|label]]``."""
    if not state.src.startswith("[[", state.pos) or state.linkLevel > 0:
        return False
    match = WIKILINK.match(state.src, state.pos)
    if match is None or match.end() > state.posMax:
        return False
    if not silent:
        target = match.group(1).strip()
        label = (match.group(3) or "")[1:].strip() or target
        context = _context(state.env)
        path = context.page(target)
        if path is not None and context.pid is not None:
            opening = state.push("wikilink_open", "a", 1)
            opening.attrs = {
                "href": page_href(context.pid, path),
                "class": "wikilink",
                "data-page": path,
            }
            closing_tag = "a"
        else:
            opening = state.push("wikilink_open", "span", 1)
            opening.attrs = {
                "class": "wikilink wikilink-missing",
                "data-target": target,
                "title": f"No page matches {target}",
            }
            closing_tag = "span"
        state.push("text", "", 0).content = label
        state.push("wikilink_close", closing_tag, -1)
    state.pos = match.end()
    return True


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------


def _fence(renderer: Any, tokens: Sequence[Token], idx: int, options: Any, env: Any) -> str:
    """A fenced block as text.  Only a plain language name becomes a class."""
    token = tokens[idx]
    language = _FENCE_LANGUAGE.match(token.info.strip()) if token.info else None
    attribute = f' class="language-{language.group(0)}"' if language else ""
    return f"<pre><code{attribute}>{escapeHtml(token.content)}</code></pre>\n"


def _build() -> MarkdownIt:
    markdown = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
    markdown.enable(["table", "strikethrough"])
    markdown.inline.ruler.before("link", "wikilink", _wikilink)
    markdown.core.ruler.push("paper_orchestra_finish", _finish)
    markdown.add_render_rule("fence", _fence)
    return markdown


_MARKDOWN = _build()


def _normalise(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def render_markdown(
    text: str,
    *,
    project: Project | None = None,
    pid: str | None = None,
    page_path: str | None = None,
    heading_ids: bool = True,
) -> str:
    """``text`` as HTML that is safe to insert into the app's page.

    ``project`` resolves wiki links and relative links; ``pid`` is the project's
    id in addresses (default: ``project.id``); ``page_path`` is where the text
    lives, for ``../`` references.  Without a project, wiki links are shown as
    unresolved and no image is loaded.  Headings get ids ``h-<slug>``, unique
    within the text, unless ``heading_ids`` is false.
    """
    env = {_ENV_KEY: _Context(project, pid, page_path, heading_ids)}
    return _MARKDOWN.render(_normalise(text), env)


# ---------------------------------------------------------------------------
# A suggested edit, read in place
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Marked:
    """A text as it would read after an edit, with the changes marked."""

    html: str
    words_added: int
    words_removed: int


_TAG = re.compile(r"<[^>]*>")
_TAG_NAME = re.compile(r"^<\s*(/?)\s*([A-Za-z0-9]+)")
#: The blanks before a word, and the word: letters and digits, or one other character.
_WORD = re.compile(r"(\s*)(\w+|[^\w\s])")
_VOID_TAGS = frozenset({"img", "br", "hr"})
#: Elements whose direct text is prose, so a ``<del>`` may stand there.  In the others
#: (lists, tables, quotes) the only direct text is the line breaks between children.
_TEXT_PARENTS = frozenset(
    {"p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}
    | {"a", "em", "strong", "code", "s", "span"}
)
#: Two versions of a block are compared word by word only when they share this much.
_SAME_BLOCK_RATIO = 0.35
#: Above this many pairs of blocks in one rewritten stretch, they are paired by position.
_MAX_ALIGNED_CELLS = 2500
#: Above this many pairs of words, frequent words are left out of the comparison
#: (``difflib``'s own shortcut), which keeps a rewritten long block from taking seconds.
_MAX_EXACT_CELLS = 250_000
#: Blocks paired by the words they hold rather than by comparing them must share this
#: much: the count ignores order, so it overstates what two blocks have in common.
_SAME_WORDS_RATIO = 0.6
#: How much comparing one suggested edit may cost in all, in pairs of equal items that
#: ``difflib`` walks over (a second or so of work).  A page is rendered again at every
#: look and its text is a model's: without a limit, one long block edited throughout,
#: or a text of thousands of short lines, keeps a request busy for minutes.  Past the
#: limit, what is left of the changed blocks is shown as removed and added whole.
_MAX_COMPARED_PAIRS = 4_000_000

_Opcode = tuple[str, int, int, int, int]


class _TooLong(Exception):
    """A comparison was given up: it would cost more than its budget allows."""


class _Budget:
    """What is left of the work the comparisons of one render may do."""

    def __init__(self, pairs: int | None = None) -> None:
        self.left = _MAX_COMPARED_PAIRS if pairs is None else pairs

    def take(self, pairs: int) -> bool:
        """Set ``pairs`` aside for a comparison; ``False`` when that many are not left."""
        if pairs > self.left:
            return False
        self.left -= pairs
        return True


class _BoundedMatcher(difflib.SequenceMatcher):
    """``difflib``'s matcher, giving up (``_TooLong``) when its budget is used up.

    Its time goes into ``find_longest_match``: for every item of ``a`` in the range it
    is asked about, it walks the places where that item occurs in ``b``, and it is
    asked once for every stretch the two sequences share.  Sequences with few
    distinct items that differ in many places make that thousands of long walks.
    The length of each walk is known beforehand, so the cost is counted before it
    is paid.
    """

    def __init__(
        self, a: Sequence[str], b: Sequence[str], budget: _Budget, *, autojunk: bool = True
    ) -> None:
        super().__init__(None, a, b, autojunk)
        self._budget = budget
        places = self.b2j
        walked = 0
        #: _walks[n]: the places in ``b`` of the first ``n`` items of ``a``, plus ``n``.
        self._walks = [0]
        for item in a:
            walked += len(places.get(item, ())) + 1
            self._walks.append(walked)

    def find_longest_match(
        self, alo: int = 0, ahi: int | None = None, blo: int = 0, bhi: int | None = None
    ) -> Any:
        ahi = len(self.a) if ahi is None else ahi
        if not self._budget.take(self._walks[ahi] - self._walks[alo]):
            raise _TooLong
        return super().find_longest_match(alo, ahi, blo, bhi)


def _blocks(text: str) -> tuple[list[str], dict[str, Any]]:
    """The top-level blocks of a Markdown text as source, and its link definitions
    (which belong to the whole text, so each block is rendered with them)."""
    env: dict[str, Any] = {}
    lines = text.split("\n")
    blocks = [
        "\n".join(lines[token.map[0] : token.map[1]])
        for token in _MARKDOWN.parse(text, env)
        if token.level == 0 and token.nesting >= 0 and token.map
    ]
    return blocks, dict(env.get("references") or {})


def _words(text: str) -> tuple[list[tuple[str, str]], str]:
    """``text`` as ``(blanks before, word)`` pairs, and the blanks after the last word."""
    words = []
    end = 0
    for match in _WORD.finditer(text):
        words.append((match.group(1), match.group(2)))
        end = match.end()
    return words, text[end:]


def _plain_words(text: str) -> list[str]:
    return [word for _, word in _words(text)[0]]


def _count(words: Sequence[str]) -> int:
    """How many of ``words`` are words rather than punctuation."""
    return sum(1 for word in words if word[:1].isalnum() or word[:1] == "_")


def _matcher(old: Sequence[str], new: Sequence[str]) -> difflib.SequenceMatcher:
    return difflib.SequenceMatcher(None, old, new, autojunk=len(old) * len(new) > _MAX_EXACT_CELLS)


def _shared_ends(old: Sequence[str], new: Sequence[str]) -> tuple[int, int]:
    """How many items two sequences share at their start, and then at their end."""
    shorter = min(len(old), len(new))
    head = 0
    while head < shorter and old[head] == new[head]:
        head += 1
    tail = 0
    while tail < shorter - head and old[len(old) - 1 - tail] == new[len(new) - 1 - tail]:
        tail += 1
    return head, tail


def _opcodes(old: Sequence[str], new: Sequence[str], budget: _Budget) -> list[_Opcode] | None:
    """How ``old`` becomes ``new`` item by item (``difflib``'s opcodes), or ``None`` when
    the budget does not cover the comparison.

    What the two share at their start and at their end is set aside first and costs
    nothing: an edit in one place of a long block is cheap however long the block.
    """
    head, tail = _shared_ends(old, new)
    old_end, new_end = len(old) - tail, len(new) - tail
    codes: list[_Opcode] = [("equal", 0, head, 0, head)] if head else []
    if old_end > head or new_end > head:
        exact = (old_end - head) * (new_end - head) <= _MAX_EXACT_CELLS
        try:
            middle = _BoundedMatcher(
                old[head:old_end], new[head:new_end], budget, autojunk=not exact
            ).get_opcodes()
        except _TooLong:
            return None
        codes += [
            (operation, old_from + head, old_to + head, new_from + head, new_to + head)
            for operation, old_from, old_to, new_from, new_to in middle
        ]
    if tail:
        codes.append(("equal", old_end, len(old), new_end, len(new)))
    return codes


def _text_nodes(rendered: str) -> list[tuple[str, str, bool]] | None:
    """``rendered`` as ``(kind, raw, prose)`` pieces: tags, and the text between them
    with whether a ``<del>`` may stand there.  ``None`` when the markup is not what
    this module's renderer writes."""
    pieces: list[tuple[str, str, bool]] = []
    open_tags: list[str] = []
    position = 0
    for match in _TAG.finditer(rendered):
        if match.start() > position:
            raw = rendered[position : match.start()]
            prose = bool(open_tags) and open_tags[-1] in _TEXT_PARENTS
            if not prose and raw.strip():
                return None
            pieces.append(("text", raw, prose))
        name = _TAG_NAME.match(match.group(0))
        if name is None:
            return None
        tag = name.group(2).lower()
        if name.group(1):
            if not open_tags or open_tags.pop() != tag:
                return None
        elif tag not in _VOID_TAGS:
            open_tags.append(tag)
        pieces.append(("tag", match.group(0), False))
        position = match.end()
    if rendered[position:].strip() or open_tags:
        return None
    pieces.append(("text", rendered[position:], False))
    return pieces


def _run(words: Sequence[tuple[str, str]]) -> str:
    """Words as they stood, without the blanks before the first."""
    return words[0][1] + "".join(blanks + word for blanks, word in words[1:])


def _mark_words(
    old_html: str, new_html: str, budget: _Budget
) -> tuple[str, int, int, float] | None:
    """``new_html`` with the words that differ from ``old_html`` marked; with it, how
    many words were added and removed, and how much the two share (0 to 1).

    The new block keeps its structure; inserted words are wrapped where they
    stand, deleted words are put back as escaped text inside ``<del>``.  ``None``
    when the block has no prose to mark (an image, a rule) or the budget does not
    cover comparing the two.
    """
    old_pieces, new_pieces = _text_nodes(old_html), _text_nodes(new_html)
    if old_pieces is None or new_pieces is None:
        return None
    old_words = [
        pair
        for kind, raw, prose in old_pieces
        if kind == "text" and prose
        for pair in _words(html.unescape(raw))[0]
    ]
    new_words: list[tuple[int, str, str]] = []  # (text node, blanks before, word)
    trailing: dict[int, str] = {}
    for index, (kind, raw, prose) in enumerate(new_pieces):
        if kind == "text" and prose:
            words, trailing[index] = _words(html.unescape(raw))
            new_words += [(index, blanks, word) for blanks, word in words]
    if not new_words:
        return None
    output: dict[int, list[str]] = {}
    added = removed = 0
    last_node = new_words[0][0]

    def emit_deleted(first: int, last: int, node: int) -> None:
        nonlocal removed
        gone = old_words[first:last]
        text = html.escape(_run(gone))
        output.setdefault(node, []).append(f"{html.escape(gone[0][0])}<del>{text}</del>")
        removed += _count([word for _, word in gone])

    def emit_inserted(first: int, last: int, *, after_deleted: bool) -> None:
        nonlocal added, last_node
        index = first
        while index < last:
            # One <ins> per text node, so that no tag of the block is ever crossed.
            node = new_words[index][0]
            run = []
            while index < last and new_words[index][0] == node:
                run.append(new_words[index][1:])
                index += 1
            # Right behind what it replaces, the blanks before it are already there.
            blanks = "" if after_deleted and run[0][0] == " " else html.escape(run[0][0])
            output.setdefault(node, []).append(f"{blanks}<ins>{html.escape(_run(run))}</ins>")
            added += _count([word for _, word in run])
            last_node, after_deleted = node, False

    codes = _opcodes([word for _, word in old_words], [word for _, _, word in new_words], budget)
    if codes is None:
        return None
    shared = 0
    for operation, old_from, old_to, new_from, new_to in codes:
        if operation == "equal":
            shared += new_to - new_from
            for node, blanks, word in new_words[new_from:new_to]:
                output.setdefault(node, []).append(html.escape(blanks + word))
                last_node = node
        elif operation == "delete":
            # Where the words stood: at the end of what precedes them, so that words cut
            # from the end of a paragraph stay in that paragraph.
            emit_deleted(old_from, old_to, last_node)
        else:
            if operation == "replace":
                emit_deleted(old_from, old_to, new_words[new_from][0])
            emit_inserted(new_from, new_to, after_deleted=operation == "replace")
    marked = "".join(
        "".join(output.get(index, [])) + html.escape(trailing[index])
        if kind == "text" and prose
        else raw
        for index, (kind, raw, prose) in enumerate(new_pieces)
    )
    return marked, added, removed, 2 * shared / (len(old_words) + len(new_words))


def _likeness(was: Sequence[str], now: Sequence[str]) -> float:
    """How much two blocks share, 0 to 1; 0 as soon as it is clear they are not versions
    of each other (the quick estimates are upper bounds and cost next to nothing)."""
    matcher = _matcher(was, now)
    if min(matcher.real_quick_ratio(), matcher.quick_ratio()) < _SAME_BLOCK_RATIO:
        return 0.0
    return matcher.ratio()


def _shared_words(was: Counter[str], now: Counter[str]) -> float:
    """The most two blocks can share, 0 to 1, going by the words they hold and not by
    their order: what ``difflib`` calls the quick ratio, from counts made once a block."""
    total = sum(was.values()) + sum(now.values())
    return 2 * sum((was & now).values()) / total if total else 0.0


def _aligned(
    before: Sequence[str], after: Sequence[str]
) -> list[tuple[str | None, str | None, bool]]:
    """Pair the blocks of a rewritten stretch: each old block with the new block that
    is a version of it, in order; a block without a counterpart is paired with ``None``.

    Pairing by position would show an edit behind an inserted paragraph as everything
    deleted and written anew.  The third item says whether the pair is known to be two
    versions of one block.  It is when every old block was compared with every new
    one word by word, which a stretch of ordinary size allows.  A long stretch is
    paired by the words its blocks hold (which overstates what they share, hence the
    higher bar), a very long one by position, and whoever marks the words then finds
    out how alike two blocks really are.
    """
    if len(before) * len(after) > _MAX_ALIGNED_CELLS:
        pairs = [(was, now, False) for was, now in zip(before, after, strict=False)]
        pairs += [(was, None, False) for was in before[len(after) :]]
        return pairs + [(None, now, False) for now in after[len(before) :]]
    old_words = [_plain_words(block) for block in before]
    new_words = [_plain_words(block) for block in after]
    sure = sum(map(len, old_words)) * sum(map(len, new_words)) <= _MAX_EXACT_CELLS
    if sure:
        bar = _SAME_BLOCK_RATIO
        alike = [[_likeness(was, now) for now in new_words] for was in old_words]
    else:
        bar = _SAME_WORDS_RATIO
        old_counts = [Counter(words) for words in old_words]
        new_counts = [Counter(words) for words in new_words]
        alike = [[_shared_words(was, now) for now in new_counts] for was in old_counts]
    # best[i][j]: the most likeness that pairing before[i:] with after[j:] can keep.
    best = [[0.0] * (len(after) + 1) for _ in range(len(before) + 1)]
    for i in range(len(before) - 1, -1, -1):
        for j in range(len(after) - 1, -1, -1):
            paired = alike[i][j] + best[i + 1][j + 1] if alike[i][j] >= bar else 0
            best[i][j] = max(paired, best[i + 1][j], best[i][j + 1])
    pairs: list[tuple[str | None, str | None, bool]] = []
    i = j = 0
    while i < len(before) and j < len(after):
        if alike[i][j] >= bar and best[i][j] == alike[i][j] + best[i + 1][j + 1]:
            pairs.append((before[i], after[j], sure))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            pairs.append((before[i], None, sure))
            i += 1
        else:
            pairs.append((None, after[j], sure))
            j += 1
    pairs += [(was, None, sure) for was in before[i:]]
    return pairs + [(None, now, sure) for now in after[j:]]


def render_marked(
    old: str,
    new: str,
    *,
    project: Project | None = None,
    pid: str | None = None,
    page_path: str | None = None,
) -> Marked:
    """``new`` as HTML with what differs from ``old`` marked by ``<ins>`` and ``<del>``.

    Unchanged blocks are rendered as they are.  A block that changed is wrapped
    in ``<div class="diff-changed">`` with the changed words marked inside it; a
    block that was added or removed (or rewritten beyond comparison) is wrapped
    whole in ``<ins class="diff-block">`` or ``<del class="diff-block">``.
    Headings carry no ids here: the page itself is usually on screen too.

    The time this takes is bounded (``_MAX_COMPARED_PAIRS``): the text is a
    model's, and the page is rendered at every look.  A very long block edited
    throughout is shown as removed and added whole rather than word by word.
    """
    old_text, new_text = _normalise(old), _normalise(new)
    old_blocks, old_references = _blocks(old_text)
    new_blocks, new_references = _blocks(new_text)

    def render(block: str, references: dict[str, Any]) -> str:
        env = {_ENV_KEY: _Context(project, pid, page_path, False), "references": references}
        return _MARKDOWN.render(block, env)

    parts: list[str] = []
    added = removed = 0
    budget = _Budget()

    def whole(tag: str, block: str, references: dict[str, Any]) -> int:
        rendered = render(block, references)
        parts.append(f'<{tag} class="diff-block">{rendered}</{tag}>\n')
        return _count(_plain_words(html.unescape(_TAG.sub(" ", rendered))))

    # Block against block first.  A text of thousands of blocks that differ throughout
    # gets a share of the budget for that; past it, everything between what the two
    # versions share at their ends is one stretch.
    stripped_old = [block.strip() for block in old_blocks]
    stripped_new = [block.strip() for block in new_blocks]
    by_block = _opcodes(stripped_old, stripped_new, _Budget(_MAX_COMPARED_PAIRS // 4))
    if by_block is None:
        head, tail = _shared_ends(stripped_old, stripped_new)
        old_end, new_end = len(old_blocks) - tail, len(new_blocks) - tail
        by_block = [
            ("equal", 0, head, 0, head),
            ("replace", head, old_end, head, new_end),
            ("equal", old_end, len(old_blocks), new_end, len(new_blocks)),
        ]
    for operation, old_from, old_to, new_from, new_to in by_block:
        if operation == "equal":
            parts += [render(block, new_references) for block in new_blocks[new_from:new_to]]
            continue
        stretch = _aligned(old_blocks[old_from:old_to], new_blocks[new_from:new_to])
        for was, now, sure in stretch:
            marked = None
            if was is not None and now is not None:
                marked = _mark_words(
                    render(was, old_references), render(now, new_references), budget
                )
            if marked is not None and not sure and marked[3] < _SAME_BLOCK_RATIO:
                # Paired by position or by the words they hold, and not alike after
                # all: one was removed and the other written.
                marked = None
            if marked is not None:
                parts.append(f'<div class="diff-changed">{marked[0]}</div>\n')
                added += marked[1]
                removed += marked[2]
                continue
            if was is not None:
                removed += whole("del", was, old_references)
            if now is not None:
                added += whole("ins", now, new_references)
    return Marked(html="".join(parts), words_added=added, words_removed=removed)


# ---------------------------------------------------------------------------
# Unified diffs
# ---------------------------------------------------------------------------


def _hunk_range(start: int, stop: int) -> str:
    """Lines ``start`` to ``stop`` (counted from 0) as the header of a hunk names them."""
    length = stop - start
    if length == 1:
        return str(start + 1)
    return f"{start + 1 if length else start},{length}"


def _hunks(old: Sequence[str], new: Sequence[str], groups: Any) -> list[str]:
    """The hunks of a unified diff for ``difflib``'s grouped opcodes, as ``diff -u``
    writes them."""
    lines: list[str] = []
    for group in groups:
        first, last = group[0], group[-1]
        lines.append(f"@@ -{_hunk_range(first[1], last[2])} +{_hunk_range(first[3], last[4])} @@")
        for operation, old_from, old_to, new_from, new_to in group:
            if operation == "equal":
                lines += [f" {line}" for line in old[old_from:old_to]]
                continue
            if operation in ("replace", "delete"):
                lines += [f"-{line}" for line in old[old_from:old_to]]
            if operation in ("replace", "insert"):
                lines += [f"+{line}" for line in new[new_from:new_to]]
    return lines


def _one_hunk(old: Sequence[str], new: Sequence[str], context: int) -> list[str]:
    """The diff that replaces everything between what the two texts share at their
    start and at their end: correct and applicable, though not the shortest."""
    head, tail = _shared_ends(old, new)
    first = max(0, head - context)
    old_end, new_end = len(old) - tail, len(new) - tail
    after = min(tail, context)
    return [
        f"@@ -{_hunk_range(first, old_end + after)} +{_hunk_range(first, new_end + after)} @@",
        *(f" {line}" for line in old[first:head]),
        *(f"-{line}" for line in old[head:old_end]),
        *(f"+{line}" for line in new[head:new_end]),
        *(f" {line}" for line in old[old_end : old_end + after]),
    ]


def unified_diff(
    old: str, new: str, *, old_label: str = "current", new_label: str = "proposed", context: int = 3
) -> str:
    """The line-by-line unified diff from ``old`` to ``new``; empty when they are equal.

    The diff ``difflib`` finds, where finding it is affordable.  Texts of thousands
    of short lines that differ throughout would take it minutes (and the inbox
    asks for the diff of every pending edit): for those, one hunk from the first
    difference to the last.
    """
    old_lines, new_lines = _normalise(old).splitlines(), _normalise(new).splitlines()
    if old_lines == new_lines:
        return ""
    try:
        matcher = _BoundedMatcher(old_lines, new_lines, _Budget())
        hunks = _hunks(old_lines, new_lines, matcher.get_grouped_opcodes(context))
    except _TooLong:
        hunks = _one_hunk(old_lines, new_lines, context)
    return "\n".join([f"--- {old_label}", f"+++ {new_label}", *hunks])


def diff_html(diff: str) -> str:
    """A unified diff as HTML: one escaped ``<span>`` per line, classed by what it is."""
    spans = []
    for line in diff.split("\n") if diff else []:
        if line.startswith(("+++", "---")):
            kind = "diff-file"
        elif line.startswith("@@"):
            kind = "diff-hunk"
        elif line.startswith("+"):
            kind = "diff-add"
        elif line.startswith("-"):
            kind = "diff-del"
        else:
            kind = "diff-context"
        spans.append(f'<span class="diff-line {kind}">{html.escape(line)}</span>')
    return '<pre class="diff">' + "\n".join(spans) + "</pre>"
