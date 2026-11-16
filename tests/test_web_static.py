"""The web app's frontend as files: complete, self-contained, and allowed by the page's
Content-Security-Policy.

The frontend is static files with no build step, so these checks read them as text: every
file that is referenced exists, nothing is fetched from another site, nothing is inline
(the policy allows scripts and styles from the app's own origin only), every module parses,
and the server hands the files out as what they are.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from paper_orchestra.web.app import STATIC_DIR, create_app
from paper_orchestra.web.security import CONTENT_SECURITY_POLICY

BASE_URL = "http://127.0.0.1:8600"

INDEX = STATIC_DIR / "index.html"
SCRIPT_SRC = re.compile(r"<script\b[^>]*\bsrc=\"([^\"]+)\"", re.IGNORECASE)
LINK_HREF = re.compile(r"<link\b[^>]*\bhref=\"([^\"]+)\"", re.IGNORECASE)
IMPORT = re.compile(r"""^\s*(?:import|export)\b[^'"\n]*?\bfrom\s+['"]([^'"]+)['"]""", re.MULTILINE)
BARE_IMPORT = re.compile(r"""^\s*import\s+['"]([^'"]+)['"]""", re.MULTILINE)
DYNAMIC_IMPORT = re.compile(r"""\bimport\(\s*['"]([^'"]+)['"]""")
URL = re.compile(r"""(?:https?:)?//[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,}[^\s"'`)<>]*""")
#: XML namespaces are names, not addresses anything is fetched from.
NAMESPACES = ("http://www.w3.org/",)


def _files(*suffixes: str) -> list[Path]:
    return sorted(p for p in STATIC_DIR.rglob("*") if p.is_file() and p.suffix in suffixes)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _modules() -> dict[Path, str]:
    """Every ES module reachable from ``index.html``, with its source."""
    entries = [
        (STATIC_DIR / src).resolve()
        for src in re.findall(
            r"<script\b[^>]*type=\"module\"[^>]*\bsrc=\"([^\"]+)\"", _text(INDEX), re.IGNORECASE
        )
    ]
    found: dict[Path, str] = {}
    queue = list(entries)
    while queue:
        path = queue.pop()
        if path in found:
            continue
        assert path.is_file(), f"{path.relative_to(STATIC_DIR)} is imported but does not exist"
        source = _text(path)
        found[path] = source
        for pattern in (IMPORT, BARE_IMPORT, DYNAMIC_IMPORT):
            for target in pattern.findall(source):
                assert target.startswith(("./", "../")), (
                    f"{path.name} imports {target!r}: only relative imports work without a build"
                )
                queue.append((path.parent / target).resolve())
    return found


def test_the_frontend_is_installed_next_to_the_server():
    assert INDEX.is_file(), "paper_orchestra/web/static/index.html is missing"
    assert "<title>PaperOrchestra</title>" in _text(INDEX)


def test_everything_index_html_refers_to_exists():
    html = _text(INDEX)
    referred = SCRIPT_SRC.findall(html) + LINK_HREF.findall(html)
    assert referred, "index.html loads no script and no stylesheet"
    for target in referred:
        assert not target.startswith(("http:", "https:", "//", "/")), (
            f"index.html refers to {target!r}: files are addressed relative to the page"
        )
        assert (STATIC_DIR / target).is_file(), f"index.html refers to {target}, which is missing"


def test_every_import_resolves_and_no_module_is_left_over():
    modules = _modules()
    assert len(modules) >= 10
    on_disk = {path.resolve() for path in _files(".js")}
    classic = {
        (STATIC_DIR / src).resolve()
        for src in SCRIPT_SRC.findall(_text(INDEX))
        if (STATIC_DIR / src).resolve() not in modules
    }
    unused = sorted(str(p.relative_to(STATIC_DIR)) for p in on_disk - set(modules) - classic)
    assert not unused, f"script files nothing loads: {unused}"


def test_no_file_reaches_for_another_site():
    """No CDN, no font service, no analytics: the policy's ``default-src 'none'`` would
    block them, and the app has to work with the network unplugged."""
    for path in _files(".html", ".css", ".js", ".svg"):
        text = _text(path)
        for found in URL.findall(text):
            assert found.startswith(NAMESPACES), (
                f"{path.relative_to(STATIC_DIR)} names an outside address: {found}"
            )
        assert "@import" not in text or path.suffix != ".css", f"{path.name} imports a stylesheet"
    for path in _files(".css"):
        for target in re.findall(r"url\(\s*['\"]?([^'\")]+)", _text(path)):
            if target.startswith("#"):
                continue  # a pattern of the same document (the map's hatch)
            assert (
                not target.startswith(("http", "//", "data:")) and (path.parent / target).exists()
            ), f"{path.name} refers to {target}"


def test_index_html_has_nothing_inline():
    html = _text(INDEX)
    for tag in re.findall(r"<script\b[^>]*>", html, re.IGNORECASE):
        assert " src=" in tag, f"an inline script would be blocked: {tag}"
    assert not re.search(r"<style\b", html, re.IGNORECASE), "an inline stylesheet would be blocked"
    assert not re.search(r"\sstyle\s*=", html, re.IGNORECASE), "a style attribute is blocked"
    assert not re.search(r"\son[a-z]+\s*=", html, re.IGNORECASE), "an inline handler is blocked"
    assert "javascript:" not in html.lower()
    first_script = SCRIPT_SRC.findall(html)[0]
    assert first_script.endswith("theme.js"), "the theme is set by the first script, before paint"


def test_the_policy_is_the_one_the_frontend_was_written_for():
    policy = {
        part.split()[0]: part.split()[1:] for part in CONTENT_SECURITY_POLICY.split("; ") if part
    }
    assert policy["default-src"] == ["'none'"]
    assert policy["script-src"] == ["'self'"]
    assert policy["style-src"] == ["'self'"]
    assert policy["connect-src"] == ["'self'"]


def test_scripts_insert_no_markup_the_policy_would_block():
    """Markup built in scripts goes into the page as HTML, where the policy applies to it
    as to the page itself: no ``style`` attribute, no handler attribute, no script."""
    for path in _files(".js"):
        source = _text(path)
        name = path.relative_to(STATIC_DIR)
        assert not re.search(r"""[\s"'`]style\s*=\s*\\?["'$]""", source), (
            f"{name} writes a style attribute into markup; set it through the CSSOM instead"
        )
        assert not re.search(
            r"""[\s"'`]on(click|load|error|input|change|submit|key\w+|mouse\w+)\s*=\s*\\?["']""",
            source,
        ), f"{name} writes an inline event handler into markup"
        assert "<script" not in source.lower().replace("<script src", ""), f"{name} writes a script"
        assert "javascript:" not in source.lower(), f"{name} builds a javascript: address"
        assert not re.search(r"\beval\s*\(|\bnew Function\s*\(|document\.write\s*\(", source), (
            f"{name} evaluates text as code"
        )
        assert "setAttribute('style'" not in source and 'setAttribute("style"' not in source
        assert ".cssText" not in source


def test_text_from_the_api_reaches_the_page_through_one_escaping_door():
    """``innerHTML`` is assigned in three reviewed places only: the helper that takes
    escaped markup, the editor preview (the server's rendering), and the map's SVG (numbers,
    class names and escaped text).  A new assignment has to be reviewed and listed here."""
    assigning = {
        str(path.relative_to(STATIC_DIR)): len(re.findall(r"\.innerHTML\s*=", _text(path)))
        for path in _files(".js")
        if re.search(r"\.innerHTML\s*=", _text(path))
    }
    assert assigning == {"js/util.js": 1, "js/page.js": 1, "js/map.js": 1}, assigning
    assert not any(
        re.search(r"insertAdjacentHTML|outerHTML\s*=|DOMParser|createContextualFragment", _text(p))
        for p in _files(".js")
    )
    util = _text(STATIC_DIR / "js" / "util.js")
    assert "throw new TypeError('setHTML takes the result of html`...`')" in util
    # The mockup's client-side Markdown renderer and its mock data are gone.
    assert not (STATIC_DIR / "js" / "data.js").exists()
    for path in _files(".js"):
        assert "Atlas" not in _text(path), f"{path.name} still names the concept, not the product"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_script_parses(tmp_path):
    modules = _modules()
    for path in _files(".js"):
        # A module is checked as one whatever node would guess from the file name.
        suffix = ".mjs" if path.resolve() in modules else ".cjs"
        copy = tmp_path / (path.stem + suffix)
        copy.write_text(_text(path), encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(copy)], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"{path.name} does not parse:\n{result.stderr}"


def test_the_server_serves_the_frontend_as_what_it_is(tmp_path):
    client = TestClient(create_app(tmp_path), base_url=BASE_URL)
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert page.headers["cache-control"] == "no-store"
    assert page.text == _text(INDEX)
    expected = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
    served = 0
    for path in _files(".js", ".css", ".svg"):
        response = client.get("/" + path.relative_to(STATIC_DIR).as_posix())
        assert response.status_code == 200, path.name
        assert response.headers["content-type"].startswith(expected[path.suffix]), (
            f"{path.name} is served as {response.headers['content-type']}: a browser runs "
            "a module only when it is served as JavaScript"
        )
        assert response.text == _text(path)
        served += 1
    assert served >= 12
    # A client-side route is answered with the page; a missing file is not.
    assert client.get("/p/p-123/wiki").text == _text(INDEX)
    assert client.get("/js/no-such-module.js").status_code == 404


def test_the_stylesheet_covers_what_the_server_renders():
    """The classes of the server's HTML (wiki links, marked edits, diffs, aligned cells)
    are styled: unstyled, a suggested edit would read like the page itself."""
    css = _text(STATIC_DIR / "css" / "app.css")
    for selector in (
        ".wikilink",
        ".wikilink-missing",
        "a.external",
        ".diff-changed",
        "ins.diff-block",
        "del.diff-block",
        "pre.diff",
        ".diff-add",
        ".diff-del",
        ".diff-hunk",
        ".diff-file",
        ".align-right",
        ".align-center",
        ".file-link",
    ):
        assert selector in css, f"the stylesheet has no rule for {selector}"
    assert "prefers-reduced-motion" in css and ":focus-visible" in css
    assert '[data-theme="dark"]' in css and "prefers-color-scheme: dark" in css


def test_no_default_of_the_server_is_written_into_the_client():
    """What a new project starts with (its limits, what needs a review, whom a request can
    be for) is asked of the server, ``GET /api/defaults``: a number or a list copied into the
    frontend would go on saying the old value after the server's changed."""
    api = _text(STATIC_DIR / "js" / "api.js")
    assert "'/api/defaults'" in api
    home = _text(STATIC_DIR / "js" / "home.js")
    assert "defaults()" in home, "the composer does not read the server's defaults"
    for path in _files(".js"):
        source = _text(path)
        name = path.relative_to(STATIC_DIR)
        for copied in ("25.00", "5.00 dollars", "15 minutes", "|| 900", "|| 25)"):
            assert copied not in source, f"{name} carries a default of its own: {copied!r}"
    overview = _text(STATIC_DIR / "js" / "overview.js")
    assert "request_for" in overview, "the addressees of an instruction are not the server's"


def test_the_frontend_speaks_the_api_as_it_is_now():
    """The requests the repairs of the server added or changed: forgetting a worker that
    cannot be verified, a phase the person names, an instruction with an addressee, and a
    save that accepts only when it says so."""
    api = _text(STATIC_DIR / "js" / "api.js")
    assert "/worker/forget" in api
    assert "for: role" in api, "an instruction is sent without its addressee"
    actions = _text(STATIC_DIR / "js" / "actions.js")
    # One place sets it, and only for "Save and accept": editing is not deciding.
    assert actions.count("body.accept = true") == 1
    assert re.search(r"if \(accept\) body\.accept = true", actions)
    assert "api.advance(s.pid, phase ? { phase } : {})" in actions
    page = _text(STATIC_DIR / "js" / "page.js")
    assert "number_report" in page and "edited_record" in page
    # The findings a paper would leave out are the server's list, with its reasons.
    assert "left_out" in _text(STATIC_DIR / "js" / "paper.js")
    for gone in ("meta.unverified", "(r.handled_in || {})"):
        assert not any(gone in _text(path) for path in _files(".js")), (
            f"the frontend still reads {gone}, which the server no longer writes that way"
        )


def test_the_stylesheet_covers_what_the_repairs_brought():
    css = _text(STATIC_DIR / "css" / "app.css")
    for selector in (
        ".go-menu",
        ".strip-sub",
        ".strip.s-unverified",
        ".top-prob",
        ".ov-problems",
        ".rq-for",
        ".num-far",
        ".numcheck-far",
        ".review-num",
        ".note-edited",
        ".rested",
        ".callout-interrupted",
        ".lib-unreadable",
        ".paper-numbers",
        ".paper-left",
    ):
        assert selector in css, f"the stylesheet has no rule for {selector}"
