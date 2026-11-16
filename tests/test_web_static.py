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


def _js(name: str) -> str:
    return _text(STATIC_DIR / "js" / name)


def test_the_client_shows_what_the_server_says_and_keeps_no_copy_of_its_rules():
    """What the server says about a project, the client shows; what it kept only because
    the server did not say it is gone: the copy of the engine's rule for what a press of
    Continue does, the pattern that found the analyst's recommendation in a sentence, the
    guess that a state is lost, what rests on a rejected experiment kept on the device, and
    the lists of efforts and backends."""
    shell, store = _js("shell.js"), _js("store.js")
    # What a press of Continue does is the server's `next_step`, not worked out from the
    # list of experiments and the results nobody has analysed.
    assert "next_step" in shell, "the main button is not labelled from the server's next_step"
    for mirrored in (".analysed", "x.state === 'approved' || x.state === 'queued'"):
        assert mirrored not in shell, f"shell.js still works out what Continue does: {mirrored}"
    # The analyst's recommendation is a field of the checkpoint, not a sentence to find.
    assert "recommends_page" in shell and ".recommends" in shell
    for source in (shell, _js("overview.js")):
        assert "journal page of this round recommends[" not in source, (
            "the recommendation is still looked for in the checkpoint's sentence"
        )
    # Where a project stands is the engine's reading, with what is wrong said beside it.
    assert "state_problem" in shell and "state_source" in shell
    assert "path === 'state.json'" not in shell, "a lost state is still guessed from the problems"
    assert "state_problem" in store, "the state that arrives with the stream loses its problem"
    home = _js("home.js")
    assert "state_source" in home, "the library row does not say where a project's state is from"
    # What rests on a rejected experiment comes with the experiment, from the server.
    core = _js("core.js")
    for kept in ("keepRested", "restedOf", "restedKey"):
        assert not any(kept in _text(path) for path in _files(".js")), (
            f"what rests on a rejected experiment is still kept on the device ({kept})"
        )
    # (What an earlier version kept there is removed when a project opens, and never read.)
    assert (
        "ui.rested" not in core
        and "rested: {}" not in core
        and "storage.get('po-rested" not in core
    )
    assert ".dependents" in _js("page.js")
    # The efforts and the compute backends a setting accepts are the server's lists.
    assert "efforts" in home and "compute_backends" in home
    for copied in ("['xhigh', 'extra high']", "['modal', 'Modal sandboxes']", "['low', 'low']"):
        assert copied not in home, f"the composer carries a list of its own: {copied}"


def test_the_client_says_what_the_server_said_happened():
    """Taking an approval back, how a write-up ended, what ends with the server and how
    many problems a project has are said in the server's words or from its fields."""
    api, actions = _js("api.js"), _js("actions.js")
    # Cancel says what the person was told it does, and shows the server's sentence.
    assert "{ expect }" in api, "a cancel does not say what the person was told it does"
    assert "will not run: its approval is taken back" not in actions
    assert "answer.message" in actions
    experiments = _js("experiments.js")
    assert "cancel_does" in experiments and "Take the approval back" in experiments
    assert "data-does" in experiments
    # How the last write-up went is `last` of GET /paper, not the server's memory of a
    # worker and the tail of its log.
    paper = _js("paper.js")
    assert "paper.last" in paper
    for gone in ("last_exit", "workerLog", "x.type === 'paper.started'"):
        assert gone not in paper, f"the paper view still reads {gone}"
    # A write-up whose worker was killed left a state that still reads as running: how it
    # ended is the server's account (it carries the worker's exit code), and is shown.
    assert "last.exit_code" in paper, "the server's account of a write-up's end is not shown"
    shell = _js("shell.js")
    assert "stops_workers_on_exit" in shell
    assert "unless it was started to stop them" not in shell
    assert "problems_total" in shell and "50 or more" not in shell
    # A withdrawn request is an event: the list of requests is read again when it arrives.
    assert re.search(r"request\\\.\(created\|withdrawn\)", _js("store.js")), (
        "request.withdrawn is not among the events that make the lists be read again"
    )
    # The quick accept of a suggested edit to a finding knows what the number check found.
    assert "number_report" in _js("inbox.js")
    # An interrupted run is drawn as that wherever the page has a mark.
    vocab = _js("vocab.js")
    assert "interrupted: 'halt'" in vocab and "shownState" in vocab


def test_the_stylesheet_covers_what_the_seams_brought():
    css = _text(STATIC_DIR / "css" / "app.css")
    for selector in (
        ".gl.t-halt",
        ".tag-state.t-halt",
        ".go-warn",
        ".go-pick",
        ".in-flag",
        ".paper-last",
    ):
        assert selector in css, f"the stylesheet has no rule for {selector}"


# ---------------------------------------------------------------------------
# A server that was restarted has a new access token: what the open app says and does
# ---------------------------------------------------------------------------

RESTARTED = "This server was restarted; open the new link printed in its terminal."

# Drives the client's API module with a server that answers as given; no network, no browser.
CONNECTION_SCRIPT = r"""
import { api, connection, ApiError, Unreachable } from './js/api.js';

const asked = [];
let answer = { status: 200, body: { ok: true } };
globalThis.fetch = async (url, init) => {
  asked.push((init && init.method) + ' ' + url);
  if (answer === null) throw new TypeError('fetch failed');
  return { ok: answer.status >= 200 && answer.status < 300, status: answer.status, text: async () => JSON.stringify(answer.body) };
};
const flips = [];
connection.on((ok) => flips.push(ok ? 'ok' : connection.denied ? 'denied' : 'gone'));
const outcome = async (call) => {
  try { await call(); return 'answered'; }
  catch (error) { return error instanceof Unreachable ? 'unreachable' : error instanceof ApiError ? error.status + ' ' + error.code + ' ' + error.message : String(error); }
};
const out = {};
const refused = { status: 401, body: { error: { code: 'token_required', message: 'This app is opened with the link printed in the terminal.' } } };

out.before = [connection.ok, connection.denied];
answer = refused;
out.read = await outcome(() => api.project('p-1'));
out.denied = [connection.ok, connection.denied];
/* reading is not tried again by itself: a poll asks, and nothing goes to the server */
let sent = asked.length;
out.polled = [await outcome(() => api.projects()), await outcome(() => api.runLog('p-1', 'E1', 1, 0)), await outcome(() => api.paper('p-1'))];
out.pollsSent = asked.length - sent;
/* what the person asks to be done is tried, and so is the question whether it is over */
sent = asked.length;
out.write = await outcome(() => api.advance('p-1', {}));
out.probe = await outcome(() => api.health());
out.triedSent = asked.length - sent;
out.stillDenied = connection.denied;
/* the new link was opened in this browser: the server answers again */
answer = { status: 200, body: { ok: true } };
out.probeAfter = await outcome(() => api.health());
out.after = [connection.ok, connection.denied];
out.readAfter = await outcome(() => api.project('p-1'));
/* a server that does not answer at all is another matter, and so is any other refusal */
answer = null;
out.gone = await outcome(() => api.project('p-1'));
out.whenGone = [connection.ok, connection.denied];
answer = { status: 404, body: { error: { code: 'not_found', message: 'There is no such project.' } } };
out.other = await outcome(() => api.project('p-1'));
out.afterOther = [connection.ok, connection.denied];
/* a write that is refused the same way says the same, and marks the connection */
answer = refused;
out.writeRefused = await outcome(() => api.stop('p-1'));
out.afterWrite = [connection.ok, connection.denied];
out.flips = flips;
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_an_answer_of_401_is_told_apart_from_a_server_that_is_gone(tmp_path):
    """While the app is open, 401 means that the server was restarted and has a new access
    token.  The client says that, in those words, and stops asking by itself: no poll
    reaches the server until the person has opened the new link."""
    (tmp_path / "js").mkdir()
    for name in ("api.js", "util.js"):
        (tmp_path / "js" / name).write_text(_js(name), encoding="utf-8")
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    (tmp_path / "check.js").write_text(CONNECTION_SCRIPT, encoding="utf-8")
    done = subprocess.run(
        ["node", str(tmp_path / "check.js")], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    out = __import__("json").loads(done.stdout)
    restarted = f"401 token_required {RESTARTED}"
    assert out["before"] == [True, False]
    assert out["read"] == restarted and out["denied"] == [False, True]
    assert out["polled"] == [restarted] * 3 and out["pollsSent"] == 0
    assert out["write"] == restarted and out["probe"] == restarted and out["triedSent"] == 2
    assert out["stillDenied"] is True
    assert out["probeAfter"] == "answered" and out["after"] == [True, False]
    assert out["readAfter"] == "answered"
    assert out["gone"] == "unreachable" and out["whenGone"] == [False, False]
    assert out["other"] == "404 not_found There is no such project."
    assert out["afterOther"] == [True, False]
    assert out["writeRefused"] == restarted and out["afterWrite"] == [False, True]
    assert out["flips"] == ["denied", "ok", "gone", "ok", "denied"]


def test_the_app_says_that_the_server_was_restarted_and_does_not_retry_for_ever():
    api, shell, main, store = _js("api.js"), _js("shell.js"), _js("main.js"), _js("store.js")
    assert RESTARTED in api
    # The banner says it in those words and does not promise that anything is tried again.
    banner = shell[shell.index("export function banner()") :]
    denied = banner[banner.index("connection.denied") : banner.index("stops_workers_on_exit")]
    assert "RESTARTED" in denied and "tried again every few seconds" not in denied
    assert "not answering" not in denied and "data-denied" in denied
    # The quiet word at the foot of a project is not "offline" either.
    assert "new link needed" in shell
    # Nothing asks at intervals while the server refuses this page ...
    assert "if (!connection.ok && !connection.denied) askHealth()" in main
    assert "if (!connection.ok) askHealth()" not in main
    # ... the event stream is not opened again and again ...
    connect = store[store.index("  connect() {") : store.index("  takeState(")]
    assert "connection.denied" in connect.split("new EventSource")[0]
    # ... and the app asks once when the person comes back to it, and opens the stream
    # again when the server lets it in.
    assert re.search(r"connection\.denied\) askHealth\(\)", main)
    assert "app.store.connect()" in main


# ---------------------------------------------------------------------------
# What a first-time user met (round 4): the client's own modules, given a project as
# the server describes it.  No browser and no network: the views are functions that
# return markup, and Node runs them.
# ---------------------------------------------------------------------------

#: A real ``Store`` filled as a fetch of the lists would fill it, and the route of a project.
HARNESS = r"""
import { app } from './js/core.js';
import { Store } from './js/store.js';

export const NOW = Date.now();
/** A stamp `seconds` ago, as the server writes them. */
export const ago = (seconds) => new Date(NOW - seconds * 1000).toISOString().replace(/\.\d+Z$/, 'Z');
export function projectStore(lists) {
  const store = new Store('p-1', () => {});
  store.server = Object.assign({ project: {}, pages: [], edges: [], proposals: [], requests: [], experiments: [] }, lists);
  store.server.project = Object.assign({ id: 'p-1', title: 'A project', settings: { limits: { model_usd: 25, experiment_usd: 5 } }, worker: { state: 'idle', last_exit: null } }, lists.project || {});
  store.loaded = true;
  store.derive();
  app.store = store;
  app.route = { name: 'project', pid: 'p-1', view: 'map', ref: '', sel: '', sub: '', q: {} };
  return store;
}
export const page = (kind, id, more) => Object.assign({ path: 'wiki/' + kind + 's/' + id + '.md', kind, id, title: 'The page ' + id, summary: '', review: 'accepted', state: null, author: 'agent:designer', round: 1, created: ago(3600), updated: ago(3600), meta: {}, relations: {} }, more || {});
export const checkpoint = (phase, next, more) => Object.assign({ status: 'checkpoint', round: 2, phase, next_phase: next, phases: [{ round: 2, phase, status: 'completed' }], spent: {} }, more || {});
export const said = (more) => Object.assign({ will_run: [], then: null, experiments: [], not_running: [], results: 0, checkpoint_of: null, needs_phase: false, blocked: null }, more || {});
/** How often `part` occurs in `text`. */
export const times = (text, part) => String(text).split(part).length - 1;
"""


def _client(tmp_path: Path, script: str) -> dict:
    """Run ``script`` beside a copy of the client's modules and the harness; what it
    prints is one JSON object."""
    (tmp_path / "js").mkdir()
    for path in _files(".js"):
        if path.parent.name == "js":
            (tmp_path / "js" / path.name).write_text(_text(path), encoding="utf-8")
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    (tmp_path / "harness.js").write_text(HARNESS, encoding="utf-8")
    (tmp_path / "check.js").write_text(script, encoding="utf-8")
    done = subprocess.run(
        ["node", str(tmp_path / "check.js")], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return __import__("json").loads(done.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

#: The engine's one sentence about a model that cannot be used, as the server sends it.
NEEDS = (
    "The analysis phase needs a model, and the model set for research, reasoning and coding "
    "(claude-fable-5-2) cannot be used because no Anthropic credentials were found: set "
    "ANTHROPIC_API_KEY or the Workload Identity Federation variables (Setup in the app and "
    "`paper-orchestra doctor` show what is configured)."
)

NO_MODEL_SCRIPT = r"""
import { projectStore, page, checkpoint, said, times, ago } from './harness.js';
import { app } from './js/core.js';
import { engineView, nextStep, phaseChoices, strip, statusbar, continueNotes } from './js/shell.js';
import { overviewView } from './js/overview.js';

const NEEDS = process.env.NEEDS;
const blocked = { phase: 'analyze', needs: 'model', settings: ['research', 'reasoning', 'coding'], models: ['claude-fable-5-2'], message: NEEDS };
const e4 = page('experiment', 'E4', { state: 'approved' });
const look = (store) => {
  const e = engineView(store), step = nextStep(store, e), bar = String(strip(store)), foot = String(statusbar(store));
  return {
    crashed: !!e.crashed, label: step.label, href: step.href, need: step.need ? step.need.message : null,
    choices: phaseChoices(store, e, step).map((c) => c.phase),
    saysIt: times(bar, NEEDS.slice(0, 60)), setup: bar.includes('#/setup'),
    accident: /stopped unexpectedly|Show log|Try again|ended with exit code/.test(bar + foot),
    promises: /analyse/i.test(step.label + ' ' + step.says + ' ' + step.tip),
    restart: /start (the app|it) again/i.test(bar),
    again: /run the command again/i.test(bar),
  };
};
const out = {};
/* an approved experiment runs without a model; its analysis cannot */
out.run = look(projectStore({ pages: [e4], experiments: [Object.assign({ runs: 0 }, e4)],
  project: { state: checkpoint('design', 'run'), next_step: said({ will_run: ['run'], then: 'analyze', experiments: ['E4'], blocked }) } }));
/* a run that comes before an analysis that was next already (an earlier result waits for it):
   the fine print must not say that the analysis goes ahead when the run cannot */
out.before = (() => {
  const store = projectStore({ pages: [e4], experiments: [Object.assign({ runs: 1 }, e4)],
    project: { state: checkpoint('run', 'analyze'), next_step: said({ will_run: ['run'], then: 'analyze', experiments: ['E4'], results: 1, blocked }) } });
  const step = nextStep(store, engineView(store));
  return { label: step.label, says: step.says, tip: step.tip, notes: String(continueNotes(store, engineView(store))) };
})();
/* nothing can be done without a model, and a press that was made all the same ended for that reason */
out.none = look(projectStore({ pages: [], experiments: [],
  project: { state: checkpoint('run', 'analyze'), next_step: said({ then: 'analyze', results: 1, blocked }),
    worker: { state: 'idle', last_exit: { kind: 'advance', code: 3, at: ago(5), stopped: false, interrupted_phase: null, reason: 'needs_model' } } } }));
/* a project that has not started */
out.fresh = look(projectStore({ pages: [], experiments: [],
  project: { state: { status: 'idle', round: 0, phase: null, next_phase: 'intake', phases: [], spent: {} }, state_source: 'not_started', next_step: said({ then: 'intake', blocked: Object.assign({}, blocked, { phase: 'intake' }) }) } }));
/* a worker that ended with the same code and no such reason did die */
out.died = look(projectStore({ pages: [], experiments: [],
  project: { state: checkpoint('run', 'analyze'), next_step: said({ will_run: ['analyze'], then: 'hypothesize', results: 1 }),
    worker: { state: 'idle', last_exit: { kind: 'advance', code: 3, at: ago(5), stopped: false, interrupted_phase: null, reason: null } } } }));
/* the overview: what the runs did, and the sentence about the model once (beside the button) or,
   when a model can be used again, not at all: the checkpoint's own account still holds it */
const paused = (next) => ({ state: checkpoint('run', 'analyze'), next_step: next,
  checkpoint: { round: 2, phase: 'run', at: ago(60), summary: '1 experiment ran: E4 succeeded. ' + NEEDS + ' Next: analysing the new results.', did: '1 experiment ran: E4 succeeded. ' + NEEDS } });
const head = (store) => {
  /* (the view writes its markup into the element it is given, and then looks for nothing in it) */
  const root = { innerHTML: '', querySelectorAll: () => [], querySelector: () => null };
  app.route = { name: 'project', pid: 'p-1', view: 'overview', ref: '', sel: '', sub: '', q: {} };
  overviewView.render(root);
  const text = root.innerHTML.slice(root.innerHTML.indexOf('class="ov-head"'), root.innerHTML.indexOf('class="ov-map"'));
  return [times(text, 'needs a model'), text.includes('E4 succeeded')];
};
out.overview = {
  blocked: head(projectStore({ pages: [], experiments: [], project: paused(said({ then: 'analyze', results: 1, blocked })) })),
  usable: head(projectStore({ pages: [], experiments: [], project: paused(said({ will_run: ['analyze'], then: 'hypothesize', results: 1 })) })),
};
/* a server that says nothing about models (an older one) changes nothing */
out.older = look(projectStore({ pages: [e4], experiments: [Object.assign({ runs: 0 }, e4)],
  project: { state: checkpoint('design', 'run'), next_step: { will_run: ['run', 'analyze'], then: 'hypothesize', experiments: ['E4'], results: 0, checkpoint_of: null, needs_phase: false } } }));
console.log(JSON.stringify(out));
"""


@needs_node
def test_a_project_without_a_usable_model_is_something_to_set_up_not_an_accident(
    tmp_path, monkeypatch
):
    """Where no model can be used, the server's forecast says so (``next_step.blocked``)
    and a worker that started nothing says why by a field (``last_exit.reason``).  The
    main button promises what happens without a model and nothing else; the engine's
    sentence stands beside it once; nothing is called an accident."""
    monkeypatch.setenv("NEEDS", NEEDS)
    out = _client(tmp_path, NO_MODEL_SCRIPT)
    run = out["run"]
    assert run["label"] == "Run the approved experiment" and not run["promises"], run
    assert run["need"] == NEEDS and run["saysIt"] == 1 and run["setup"] and run["restart"], run
    assert not run["accident"] and not run["again"] and run["choices"] == [], run
    # A run before an analysis that was next already: the analysis is still next, and
    # nothing says that it goes ahead in the run's place, which it cannot without a model.
    before = out["before"]
    assert before["label"] == "Run the approved experiment", before
    assert "still analysis" in before["says"] and "goes ahead" not in before["tip"], before
    assert "credentials are read when it starts" in before["notes"], before
    none = out["none"]
    assert none["crashed"] is False and not none["accident"], none
    assert none["label"] == "Set up a model" and none["href"] == "#/setup", none
    assert none["saysIt"] == 1 and none["restart"] and none["choices"] == [], none
    fresh = out["fresh"]
    assert fresh["label"] == "Set up a model" and fresh["href"] == "#/setup", fresh
    assert fresh["saysIt"] == 1 and not fresh["accident"], fresh
    # The overview says the sentence once, and not at all once a model can be used again.
    assert out["overview"] == {"blocked": [1, True], "usable": [0, True]}, out["overview"]
    # Only the server's field tells the two ends apart: without it, it is a crash as before.
    assert out["died"]["crashed"] is True and out["died"]["accident"], out["died"]
    older = out["older"]
    assert older["label"] == "Run the approved experiment, then analyse it", older
    assert older["need"] is None and older["saysIt"] == 0, older
    store = _js("store.js")
    assert "needs_model" in store, "the log of a worker that only needed a model is still read"


CHANGED = (
    "it is not what was approved at 2026-11-16T17:07:01Z (main.py: changed after the approval "
    "(written by human at 2026-11-16T17:07:01Z))"
)

CHANGED_SCRIPT = r"""
import { projectStore, page, checkpoint, said, times, ago } from './harness.js';
import { engineView, nextStep, continueNotes } from './js/shell.js';
import { shownState } from './js/vocab.js';
import { statusParts, statusLine } from './js/page.js';
import * as experiments from './js/experiments.js';

const CHANGED = process.env.CHANGED;
const row = (p, more) => Object.assign({ runs: 1, last_run: null, interrupted: false, cancel_does: null, will_not_run: null, last_success: null }, p, more || {});
const e1 = page('experiment', 'E1', { state: 'approved', will_not_run: CHANGED });
const e2 = page('experiment', 'E2', { state: 'approved' });
const h1 = page('hypothesis', 'H1', { state: 'open' });
const store = projectStore({ pages: [h1, e1, e2], experiments: [row(e1, { cancel_does: 'withdraw' }), row(e2, { cancel_does: 'withdraw' })],
  project: { state: checkpoint('analyze', 'hypothesize'), next_step: said({ will_run: ['run', 'hypothesize'], then: 'design', not_running: [{ id: 'E1', reason: CHANGED }] }) } });
const e = engineView(store), step = nextStep(store, e), notes = String(continueNotes(store, e));
const p1 = store.byPath.get(e1.path), p2 = store.byPath.get(e2.path);
const group = typeof experiments.groupName === 'function' ? experiments.groupName : () => null;
const out = {
  label: step.label, held: (step.held || []).map((h) => h.id), notes,
  shown: [shownState(p1), shownState(p2)],
  parts: [statusParts(store, p1), statusParts(store, p2)],
  line: statusLine(store, p1).full,
  groups: [group(store.expById.get('E1')), group(store.expById.get('E2'))],
};
/* one of two still runs: the button counts that one only, and the other is named */
const both = projectStore({ pages: [h1, e1, e2], experiments: [row(e1), row(e2)],
  project: { state: checkpoint('analyze', 'hypothesize'), next_step: said({ will_run: ['run', 'analyze'], then: 'hypothesize', experiments: ['E2'], not_running: [{ id: 'E1', reason: CHANGED }] }) } });
const step2 = nextStep(both, engineView(both));
out.both = { label: step2.label, notes: String(continueNotes(both, engineView(both))) };
/* a row that carries the reason where the page of the other list does not is still marked */
const bare = Object.assign({}, e1); delete bare.will_not_run;
const lists = projectStore({ pages: [bare], experiments: [row(e1)], project: { state: checkpoint('analyze', 'hypothesize'), next_step: said({ will_run: ['hypothesize'], then: 'design' }) } });
out.fromRow = shownState(lists.byPath.get(e1.path));
console.log(JSON.stringify(out));
"""


@needs_node
def test_an_approved_experiment_that_changed_is_shown_as_needing_approval_again(
    tmp_path, monkeypatch
):
    """An experiment that reads approved and is no longer what was approved will not run
    (``will_not_run`` of its rows, ``not_running`` of the forecast): every list shows it as
    needing approval again, and the button's sentence says that it will not run, and why."""
    monkeypatch.setenv("CHANGED", CHANGED)
    out = _client(tmp_path, CHANGED_SCRIPT)
    assert out["shown"] == ["needs approval again", "approved"], out["shown"]
    assert out["parts"][0][0] == "needs approval again" and out["parts"][1][0] == "approved"
    assert out["line"].startswith("needs approval again")
    assert out["groups"] == ["Needs your approval", "Approved, queued and running"], out["groups"]
    assert out["fromRow"] == "needs approval again"
    assert out["label"] == "Continue to hypotheses" and out["held"] == ["E1"], out
    notes = out["notes"]
    assert "E1 will not run" in notes and "main.py: changed after the approval" in notes, notes
    # The server's reason, as the app says things: no raw stamp, and the person is "you".
    assert "2026-11-16T" not in notes and "human" not in notes and "written by you" in notes
    assert "pprove it again" in notes and "/experiments/E1" in notes, notes
    assert "No experiment is approved to run" not in notes, notes
    assert "goes straight on to hypotheses" in notes, notes
    assert out["both"]["label"] == "Run the approved experiment, then analyse it", out["both"]
    assert "E1 will not run" in out["both"]["notes"], out["both"]


CANCELLED_SCRIPT = r"""
import { projectStore, page, checkpoint, said } from './harness.js';
import { shownState } from './js/vocab.js';
import { statusParts, statusLine } from './js/page.js';
import * as experiments from './js/experiments.js';

const won = { number: 1, result_page: 'wiki/results/E4-run1.md' };
const row = (p, more) => Object.assign({ runs: 2, last_run: { number: 2, status: 'cancelled', finished: null }, interrupted: false, cancel_does: null, will_not_run: null, last_success: null }, p, more || {});
const e4 = page('experiment', 'E4', { state: 'cancelled', last_success: won });
const e5 = page('experiment', 'E5', { state: 'cancelled' });
const e6 = page('experiment', 'E6', { state: 'failed', last_success: { number: 2, result_page: 'wiki/results/E6-run2.md' } });
const result = page('result', '', { path: 'wiki/results/E4-run1.md', id: null, title: 'E4 run 1', state: 'succeeded', author: 'system' });
const store = projectStore({ pages: [e4, e5, e6, result], experiments: [row(e4), row(e5), row(e6)], project: { state: checkpoint('design', 'run'), next_step: said({ will_run: ['hypothesize'], then: 'design' }) } });
const get = (p) => store.byPath.get(p.path);
const sub = typeof experiments.standingWords === 'function' ? experiments.standingWords : () => '';
console.log(JSON.stringify({
  parts: [statusParts(store, get(e4)), statusParts(store, get(e5)), statusParts(store, get(e6))],
  line: statusLine(store, get(e4)).full,
  shown: shownState(get(e4)),
  words: [sub(store.expById.get('E4')), sub(store.expById.get('E5')), sub(store.expById.get('E6'))],
}));
"""


@needs_node
def test_a_cancelled_experiment_whose_earlier_run_succeeded_says_so(tmp_path):
    """The state of an experiment is how its last approved run ended.  Where that reads
    cancelled (or failed) and an earlier run succeeded, the server names that run
    (``last_success``), and wherever the state is said the run is said with it."""
    out = _client(tmp_path, CANCELLED_SCRIPT)
    assert out["parts"][0] == ["cancelled", "run 1 succeeded"], out["parts"]
    assert out["parts"][1] == ["cancelled"], out["parts"]
    assert out["parts"][2] == ["failed", "run 2 succeeded"], out["parts"]
    assert out["line"] == "cancelled · run 1 succeeded"
    # The mark stays the one of the state the page has: only the words say more.
    assert out["shown"] == "cancelled"
    assert out["words"] == ["cancelled; run 1 succeeded", "cancelled", "failed; run 2 succeeded"]


WORDS_SCRIPT = r"""
import { projectStore, page, checkpoint, said, ago, times } from './harness.js';
import { checkpointDid } from './js/util.js';
import { reviewBar } from './js/page.js';
import * as overview from './js/overview.js';
import * as home from './js/home.js';

const out = {};
/* a rejection with and without a reason */
const noWhy = page('hypothesis', 'H2', { review: 'rejected', author: 'agent:theorist', meta: { decided_by: 'human', decided_at: ago(120) } });
const why = page('hypothesis', 'H3', { review: 'rejected', author: 'agent:theorist', meta: { decided_by: 'human', decided_at: ago(120), decision_reason: 'Not what the sources say' } });
/* a proposal the person edited: proposed 23 minutes ago, saved just now */
const edited = page('hypothesis', 'H5', { review: 'proposed', author: 'agent:theorist', round: 2, created: ago(23 * 60), updated: ago(3), meta: { updated_by: 'human' } });
const store = projectStore({ pages: [noWhy, why, edited], project: { state: checkpoint('hypothesize', 'design'), next_step: said({ will_run: ['design'], then: 'run' }) } });
const bar = (p) => String(reviewBar(store, store.byPath.get(p.path), {}));
out.noWhy = bar(noWhy); out.why = bar(why); out.edited = bar(edited);
out.when = store.pending().map((it) => it.when === edited.created);
/* what the phases did is the checkpoint's own field; an older checkpoint has only the sentence */
out.did = checkpointDid({ did: 'Round 2, design: 2 experiments are planned.', summary: 'Round 2, design: 2 experiments are planned. 9 questions await your review. Next: running what you approved.' });
out.didOld = checkpointDid({ summary: 'The theorist proposed 2 hypotheses. 2 hypotheses and 6 findings await your review. Next: designing experiments for the hypotheses you accepted.' });
out.didText = checkpointDid('The theorist proposed 2 hypotheses. 2 hypotheses await your review.');
/* the round as a line: a phase that ran twice is one mark, and one that comes again says so */
const phases = ['intake', 'research', 'hypothesize', 'design', 'run', 'run', 'analyze'].map((phase) => ({ round: 1, phase, status: 'completed', produced: [], seconds: 1 }));
const twice = projectStore({ pages: [], project: { state: { status: 'checkpoint', round: 1, phase: 'analyze', next_phase: 'hypothesize', phases, spent: {} }, next_step: said({ will_run: ['hypothesize'], then: 'design' }) } });
const rounds = typeof overview.roundsLine === 'function' ? String(overview.roundsLine(twice)) : '';
out.rounds = { runs: times(rounds, '</i>runs'), twice: rounds.includes('runs (twice)'), again: rounds.includes('hypotheses again') };
/* the library: counts in the singular where there is one, and each number of the small map named */
out.counts = typeof home.countsLine === 'function' ? home.countsLine({ idea: 1, source: 1, hypothesis: 1, experiment: 1, finding: 1 }) : '';
out.many = typeof home.countsLine === 'function' ? home.countsLine({ idea: 1, source: 14, hypothesis: 5, experiment: 4, finding: 2 }) : '';
const small = String(overview.stripMapSmall({ idea: 1, source: 14, topic: 4, question: 6 }, { question: 1 }));
out.small = { sources: small.includes('14 sources'), waiting: /6 questions, 1 (of them )?waiting/.test(small) };
console.log(JSON.stringify(out));
"""


@needs_node
def test_the_words_say_what_is_so(tmp_path):
    """Sentences that said more than was so, or the wrong time: a rejection without a
    reason, a proposal the person edited, the checkpoint's account, the round as a line,
    the counts of the library."""
    out = _client(tmp_path, WORDS_SCRIPT)
    assert "Agents see its title, nothing else." in out["noWhy"], out["noWhy"]
    assert "this reason" not in out["noWhy"]
    assert "Agents see its title and this reason, nothing else." in out["why"]
    # "Proposed by" is followed by when it was proposed; the edit is said where it belongs.
    assert "in round 2, 23 min ago" in out["edited"], out["edited"]
    assert out["when"] == [True]
    assert out["did"] == "2 experiments are planned."
    assert out["didOld"] == "The theorist proposed 2 hypotheses."
    assert out["didText"] == "The theorist proposed 2 hypotheses."
    assert out["rounds"] == {"runs": 1, "twice": True, "again": True}, out["rounds"]
    assert out["counts"] == "1 source · 1 hypothesis · 1 experiment · 1 finding"
    assert out["many"] == "14 sources · 5 hypotheses · 4 experiments · 2 findings"
    assert out["small"] == {"sources": True, "waiting": True}, out["small"]


DECIDED_SCRIPT = r"""
import { projectStore, page, checkpoint, said, ago } from './harness.js';
import * as inbox from './js/inbox.js';
import * as pages from './js/page.js';

const topic = page('topic', 'why-warmup-helps', { id: null, title: 'Why warmup helps', author: 'agent:scout' });
const h2 = page('hypothesis', 'H2', { state: 'supported', author: 'agent:theorist', meta: { decided_by: 'human', decided_at: ago(7200) } });
const turned = { id: 'ep-1', target: topic.path, author: 'agent:analyst', round: 1, created: ago(3600), summary: 'Say what F1 adds', fields: {}, relations_add: {}, review: 'rejected', decided_by: 'human', decided_at: ago(60), decision_reason: 'Not what the sources say', changes_text: true };
const taken = { id: 'ep-2', target: h2.path, author: 'agent:analyst', round: 1, created: ago(3600), summary: 'Mark H2 as supported', fields: { state: 'supported' }, relations_add: {}, review: 'accepted', decided_by: 'human', decided_at: ago(30), decision_reason: null, changes_text: false };
const store = projectStore({ pages: [topic, h2], project: { state: checkpoint('analyze', 'hypothesize'), next_step: said({ will_run: ['hypothesize'], then: 'design' }) } });
const out = { listed: null, page: '' };
if (typeof store.takeDecided === 'function' && typeof inbox.decidedItems === 'function') {
  store.takeDecided([Object.assign({ page: topic, body: 'x' }, turned), Object.assign({ page: h2 }, taken), { id: 'ep-3', target: topic.path, review: 'proposed' }]);
  out.listed = inbox.decidedItems(store).map((it) => [it.edit ? 'edit' : 'page', it.page.path === topic.path ? 'topic' : 'H2', it.review, it.reason || '']);
}
if (typeof pages.decidedEditsHTML === 'function') out.page = String(pages.decidedEditsHTML(store, { page: topic, decided_proposals: [turned] }));
out.none = typeof pages.decidedEditsHTML === 'function' ? String(pages.decidedEditsHTML(store, { page: topic })) : null;
console.log(JSON.stringify(out));
"""


@needs_node
def test_a_decided_suggested_edit_can_be_found_again(tmp_path):
    """A suggested edit that was decided is listed among what was decided, the latest
    first, with the decision and the reason, and the page it was about says what was
    suggested and what became of it (``decided_proposals`` of its view)."""
    out = _client(tmp_path, DECIDED_SCRIPT)
    assert out["listed"] == [
        ["edit", "H2", "accepted", ""],
        ["edit", "topic", "rejected", "Not what the sources say"],
        ["page", "H2", "accepted", ""],
    ], out["listed"]
    assert "Say what F1 adds" in out["page"] and "Not what the sources say" in out["page"]
    assert "rejected" in out["page"].lower() and "the analyst" in out["page"]
    # A server that does not list them leaves the page as it was.
    assert out["none"] == ""
    assert "review=all" in _js("api.js") or "'all'" in _js("store.js")


SHORT_WORKER_SCRIPT = r"""
import { projectStore, checkpoint, said, ago } from './harness.js';

/* the server as the page reaches it: what is asked for, and what the project view answers */
const asked = [];
let view = null;
globalThis.fetch = async (url) => {
  asked.push(String(url).replace(/^.*\/api\/projects\/p-1/, '') || '/');
  const body = /\/api\/projects\/p-1$/.test(url) ? view
    : { pages: [], edges: [], dangling: [], proposals: [], requests: [], experiments: [] };
  return { ok: true, status: 200, text: async () => JSON.stringify(body) };
};
const project = (worker) => ({ id: 'p-1', title: 'A project', settings: { limits: {} }, state: checkpoint('run', 'analyze'), next_step: said({ then: 'analyze' }), worker });
const busy = { state: 'running', kind: 'advance', phase: null, verified: true, last_exit: null };
const idle = { state: 'idle', kind: null, last_exit: { kind: 'advance', code: 3, at: ago(1), stopped: false, interrupted_phase: null, reason: 'needs_model' } };
const reasons = [];
const store = projectStore({ project: project(busy) });
store.onChange = (reason) => reasons.push(reason);
const out = { has: typeof store.checkWorker === 'function' };
if (out.has) {
  /* the page holds a worker as busy, and something was heard a moment ago: nothing is asked */
  store.heardAt = Date.now();
  await store.checkWorker();
  out.heard = asked.length;
  /* nothing was heard for a while, and the worker is still at work: one look, and no more */
  store.heardAt = Date.now() - 60000;
  view = project(busy);
  await store.checkWorker();
  out.still = asked.slice();
  /* nothing was heard, and the worker is gone (it ended between two looks of the stream): the lists are read again */
  asked.length = 0;
  store.heardAt = Date.now() - 60000;
  view = project(idle);
  await store.checkWorker();
  await store._waiting;
  out.gone = { looked: asked[0], lists: asked.length > 1, worker: store.project.worker.state, reason: store.project.worker.last_exit.reason, refreshed: reasons.includes('refresh') };
  /* a project nobody works on is never asked about */
  asked.length = 0;
  store.heardAt = Date.now() - 60000;
  await store.checkWorker();
  out.idle = asked.length;
}
console.log(JSON.stringify(out));
"""


@needs_node
def test_a_worker_that_ended_between_two_looks_of_the_stream_is_noticed(tmp_path):
    """A worker that starts nothing ends within half a second and writes no event: the
    event stream, which looks twice a second, may never report it.  A page that holds a
    worker as busy and has heard nothing for a while asks the project view itself, and
    reads the lists again when the worker is gone."""
    out = _client(tmp_path, SHORT_WORKER_SCRIPT)
    assert out["has"], "the store never asks whether a worker it holds as busy is still there"
    assert out["heard"] == 0
    assert out["still"] == ["/"], out["still"]
    gone = out["gone"]
    assert gone["looked"] == "/" and gone["lists"] and gone["refreshed"], gone
    assert gone["worker"] == "idle" and gone["reason"] == "needs_model", gone
    assert out["idle"] == 0
    assert "checkWorker()" in _js("main.js"), "nothing calls the look at a silent worker"


UNFILLED_SCRIPT = r"""
import { projectStore, page, checkpoint, said } from './harness.js';
import { engineView, nextStep, strip, continueNotes } from './js/shell.js';
import { shownState, earlierSuccess } from './js/vocab.js';
import { statusParts, decidedEditsHTML, reviewBar } from './js/page.js';
import { decidedItems } from './js/inbox.js';
import { checkpointDid } from './js/util.js';
import { countsLine } from './js/home.js';

const oops = { error: { code: 'internal', message: 'It could not be worked out.' } };
const e1 = page('experiment', 'E1', { state: 'approved', will_not_run: oops, last_success: oops });
const e2 = page('experiment', 'E2', { state: 'cancelled', will_not_run: 7, last_success: 'run 1' });
const store = projectStore({ pages: [e1, e2], experiments: [Object.assign({ runs: 1 }, e1), Object.assign({ runs: 1 }, e2)],
  project: { demonstration: oops, state: checkpoint('design', 'run'), checkpoint: { did: oops, summary: 'Two experiments are planned. 2 experiments await your review.' },
    next_step: said({ will_run: ['run', 'analyze'], then: 'hypothesize', experiments: ['E1'], not_running: oops, blocked: oops }),
    worker: { state: 'idle', last_exit: { kind: 'advance', code: 0, at: null, stopped: false, reason: oops } } } });
store.takeDecided(oops);
const e = engineView(store), step = nextStep(store, e);
console.log(JSON.stringify({
  label: step.label, held: step.held, need: step.need, crashed: !!e.crashed, notes: String(continueNotes(store, e)),
  strip: String(strip(store)).length > 0,
  shown: [shownState(store.byPath.get(e1.path)), shownState(store.byPath.get(e2.path))],
  parts: [statusParts(store, store.byPath.get(e1.path)), statusParts(store, store.byPath.get(e2.path))],
  won: earlierSuccess(store.byPath.get(e2.path)),
  did: checkpointDid(store.project.checkpoint),
  decided: [decidedItems(store).length, String(decidedEditsHTML(store, { page: e1, decided_proposals: oops }))],
  counts: countsLine(oops),
}));
"""


@needs_node
def test_a_field_the_server_could_not_fill_changes_nothing(tmp_path):
    """Each of the new fields may be missing (an older server) or hold something else than
    it should (the server could not work it out): the app then shows what it showed before
    the field existed, and nothing breaks."""
    out = _client(tmp_path, UNFILLED_SCRIPT)
    assert out["label"] == "Run the approved experiment, then analyse it", out
    assert out["held"] == [] and out["need"] is None and out["crashed"] is False
    assert out["notes"] == "" and out["strip"] is True
    assert out["shown"] == ["approved", "cancelled"]
    assert out["parts"] == [["approved", "runs next"], ["cancelled"]]
    assert out["won"] is None
    assert out["did"] == "Two experiments are planned."
    assert out["decided"] == [0, ""]
    assert out["counts"] == "idea only"


def test_the_client_reads_what_the_server_now_says():
    """Fields the server added for what a newcomer met: the kinds a person can write, the
    mark of a demonstration, what works without credentials, the keys of the tooltips."""
    overlays, home, shell = _js("overlays.js"), _js("home.js"), _js("shell.js")
    # The New dialog offers the kinds the server lists (a lesson among them), not its own.
    assert "creatable_kinds" in overlays, "the New dialog still carries its own list of kinds"
    # What a demonstration says it spent is its story's: said where the cost is shown.
    assert "demonstration" in home and "demonstration" in shell
    assert "without_credentials" in home, "Setup does not say what works without credentials"
    # The key of a view is drawn as a key, not written like a count beside the name.
    assert "(${i + 1})" not in shell, "the tooltip of a view still reads like a count"
    assert "data-tipkey" in shell and "tipkey" in _js("actions.js")
    # The paper view asks again when the write-up's worker is gone, not at its next poll.
    paper = _js("paper.js")
    assert "paper.running && !busy" in paper or "stale" in paper
    css = _text(STATIC_DIR / "css" / "app.css")
    for selector in (".go-need", ".go-held", ".in-why", ".decided-edits", ".st-also", "#tip kbd"):
        assert selector in css, f"the stylesheet has no rule for {selector}"
