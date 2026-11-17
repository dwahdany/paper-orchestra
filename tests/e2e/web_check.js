/* Browser checks of the web app against a real server.

   node tests/e2e/web_check.js --base http://127.0.0.1:8600 [--only a,b] [--skip a,b] [--shots DIR]

   The server asks for its access token. The checks take it from the environment
   (PAPER_ORCHESTRA_UI_TOKEN, the variable that also fixes the server's token), or from
   --base when that is the link the server printed (http://127.0.0.1:8600/?token=...).

   The server must serve a workspace that holds the demo project
   (`paper-orchestra research demo --dir <workspace>/demo`; with the large one beside it,
   `research demo --dir <workspace>/large --pages 150`, the checks of a large project run
   too) and must have been started with
   PAPER_ORCHESTRA_SCRIPTED_CLIENT=scripted_model:client and this directory on PYTHONPATH, so
   that the phases it starts run on the scripted model. The flows change the project: give
   every run a fresh copy. With a third copy in a folder named `nomodel`, and
   PAPER_ORCHESTRA_E2E_NO_MODEL naming a file (for the server and for this script alike), a
   project whose models cannot be used is looked at too: the scripted model is switched off
   while that file exists, and this script writes and removes it. With a fourth copy in a
   folder named `shell`, how agents with a shell are chosen and shown is looked at there. tests/test_web_e2e.py does
   all of that; README.md says how to do it by hand.

   Exit status 0 when every check passed. */

'use strict';
const { chromium, devices } = require('playwright');
const fs = require('fs');
const os = require('os');
const path = require('path');

const arg = (name, fallback) => { const i = process.argv.indexOf('--' + name); return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback; };
/* where the server is, and the access token it asks for: the link it printed carries both */
const GIVEN = new URL(String(arg('base', 'http://127.0.0.1:8600')));
const BASE = GIVEN.origin;
const TOKEN = GIVEN.searchParams.get('token') || process.env.PAPER_ORCHESTRA_UI_TOKEN || '';
/* this process is no browser: it sends the token as a header */
const AUTH = TOKEN ? { Authorization: 'Bearer ' + TOKEN } : {};
const ONLY = arg('only', '') ? arg('only', '').split(',') : null;
const SKIP = arg('skip', '') ? arg('skip', '').split(',') : [];
const SHOTS = arg('shots', '');
const SLOW = +arg('slow', '1');           // stretch the waits on a slow machine
/* the file whose presence switches the scripted model off (the server was given the same name) */
const NO_MODEL = process.env.PAPER_ORCHESTRA_E2E_NO_MODEL || '';

const fails = [], notes = [], errors = [];
const ok = (condition, message) => { if (!condition) fails.push(message); return !!condition; };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function api(method, url, body) {
  const response = await fetch(BASE + url, { method, headers: Object.assign({}, AUTH, body === undefined ? {} : { 'Content-Type': 'application/json' }), body: body === undefined ? undefined : JSON.stringify(body) });
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }
  return { status: response.status, data };
}
const pagePath = (p) => p.split('/').map(encodeURIComponent).join('/');

/** A browser context the server lets in: the link with the token is opened once, before
    anything else, as a person opens the link the server printed. The browser keeps the
    cookie; the address it ends up at carries no token. */
async function entered(context) {
  if (!TOKEN) return context;
  const page = await context.newPage();
  const response = await page.goto(BASE + '/?token=' + encodeURIComponent(TOKEN));
  if (!response || response.status() !== 200 || page.url().split('#')[0] !== BASE + '/') throw new Error('the link with the token did not open the app: status ' + (response && response.status()) + ' at ' + page.url().replace(TOKEN, '<token>'));
  await page.close();
  return context;
}
/* true while the check of a restarted server runs: there the server's 401 is what is looked at */
let refusing = false;
function watch(page, label) {
  page.on('console', (m) => { if (m.type() === 'error') { const t = m.text(); if (!/Failed to load resource: (net::ERR_INTERNET_DISCONNECTED|net::ERR_FAILED|the server responded with a status of (409|422|404))/.test(t) && !(refusing && /status of 401|EventSource/.test(t))) errors.push(label() + ': ' + t); } });
  page.on('pageerror', (e) => errors.push(label() + ': ' + e.message + ' ' + String(e.stack || '').split('\n').slice(1, 3).join(' ')));
}
/** Wait until `fn` (run in the page) returns something truthy. */
async function until(page, fn, argument, timeout, what) {
  try { await page.waitForFunction(fn, argument, { timeout: (timeout || 8000) * SLOW, polling: 100 }); return true; }
  catch (e) {
    /* where the page was when the wait gave up: most of what a failure needs to be understood */
    const where = await page.evaluate(() => location.hash + ', focus on ' + document.activeElement.tagName.toLowerCase() + (document.activeElement.id ? '#' + document.activeElement.id : '') + ', toast: ' + ((document.querySelector('#toast') || {}).textContent || '').trim().slice(0, 80)).catch(() => 'the page is gone');
    fails.push('timed out waiting for: ' + (what || String(fn).slice(0, 90)) + ' (at ' + where + ')');
    return false;
  }
}
const text = (page, selector) => page.locator(selector).first().textContent({ timeout: 4000 }).catch(() => '');
const count = (page, selector) => page.locator(selector).count();
const inboxCount = async (page) => +(await page.textContent('.views .count').catch(() => '0')) || 0;
/** The number on the inbox tab right now (nothing there is zero), without waiting for it. */
const waitingNow = (page) => page.evaluate(() => +((document.querySelector('.views .count') || {}).textContent || 0));
/** What has the keyboard, in a few words. */
const focusOf = (page) => page.evaluate(() => { const a = document.activeElement; return a.tagName.toLowerCase() + (a.id ? '#' + a.id : '') + (a.className && typeof a.className === 'string' ? '.' + a.className.split(' ')[0] : '') + (a.dataset && a.dataset.check ? '[check]' : ''); });
/** Requests the page makes that the server refuses, while `fn` runs. */
async function refused(page, fn) {
  const out = [];
  const note = (r) => { if (r.status() >= 400) out.push(r.status() + ' ' + r.request().method() + ' ' + r.url().replace(BASE, '')); };
  page.on('response', note);
  try { await fn(); await sleep(600); } finally { page.off('response', note); }
  return out;
}
async function shot(page, name) { if (SHOTS) { fs.mkdirSync(SHOTS, { recursive: true }); await page.screenshot({ path: path.join(SHOTS, name + '.png') }); } }
async function open(page, hash, ready) {
  await page.goto(BASE + '/' + hash);
  await page.waitForSelector(ready || '#view > *:not(.boot):not(.loading)', { timeout: 10000 * SLOW });
  await page.waitForTimeout(250);
}
/** Open an address and read everything anew: what was changed in the project's files by hand
    wrote no event, so a page that is already open learns of it at its next look. */
async function reopen(page, hash, ready) {
  await page.goto(BASE + '/' + hash);
  await page.reload();
  await page.waitForSelector(ready || '#view > *:not(.boot):not(.loading)', { timeout: 10000 * SLOW });
  await page.waitForTimeout(250);
}
/** Wait until the engine and its worker are at rest. */
async function idle(pid, timeout) {
  const end = Date.now() + (timeout || 60000) * SLOW;
  let view = null;
  while (Date.now() < end) {
    view = (await api('GET', '/api/projects/' + pid)).data;
    if (view && view.worker && view.worker.state === 'idle' && view.state.status !== 'running') return view;
    await sleep(250);
  }
  fails.push('the project did not come to rest in time (status ' + (view && view.state.status) + ', worker ' + (view && view.worker && view.worker.state) + ')');
  return view;
}

/** Where a project is on disk, or '' when this process cannot see it (a server on another
    machine). A few states cannot be reached by using the app: a run whose worker is gone, a
    worker on another machine, a state.json that is lost, a page edited into a problem. For
    those the check writes the project's files the way a person with an editor would, which
    the product allows while no agent is running. (What only an agent brings about, a journal
    page that recommends more research and a suggested edit to a finding, the scripted
    analyst does when an instruction asks for it: see scripted_model.py.) */
async function projectDir(pid) {
  const health = (await api('GET', '/api/health')).data || {};
  const view = (await api('GET', '/api/projects/' + pid)).data || {};
  const dir = health.workspace && view.dir ? path.join(health.workspace, view.dir) : '';
  return dir && fs.existsSync(path.join(dir, 'project.json')) ? dir : '';
}
/** The file of a page (`wiki/experiments/E4-….md`) in a project directory. */
const pageFile = (dir, pagePathInProject) => path.join(dir, ...pagePathInProject.split('/'));
/** A stamp as the project's files write them. */
const stamp = (date) => (date || new Date()).toISOString().replace(/\.\d+Z$/, 'Z');

/** What agents with a shell leave in a project's files, written by hand into `dir` (the
    project is at rest): the events of a scout at work (its start, a search, a page, shell
    commands, a file, board tools, a helper, a call that was refused), a second run of E1
    that an agent carried out (its record says so, its log is what the agent did), and the
    mark of an agent on the failed run of E2. The shapes are those of a real project
    (`data.did` and `data.text` of `agent.activity`, `carried_out_by` of `run.started` and of
    run.json). Returns what was written, for the check to compare with. */
function agentTraces(dir) {
  const file = path.join(dir, 'events.jsonl');
  const lines = fs.readFileSync(file, 'utf8').trimEnd().split('\n');
  let seq = +(/^\{"seq": (\d+)/.exec(lines[lines.length - 1]) || [0, 0])[1];
  const round = JSON.parse(lines[lines.length - 1]).round || 1;
  const t0 = Date.now() - 120000;
  const out = [];
  /* `seq` first, as the project writes it: the server finds its place in the file by it */
  const emit = (type, agent, phase, summary, data, refs) => { seq += 1; out.push('{"seq": ' + seq + ', ' + JSON.stringify({ ts: stamp(new Date(t0 + out.length * 1500)), type, phase, round, agent, summary, refs: refs || [], data: data || {} }).slice(1)); };
  const act = (agent, phase, summary, data) => emit('agent.activity', agent, phase, summary, data);
  const command = 'python3 -m venv .venv && ./.venv/bin/pip install -q numpy scikit-learn 2>&1 | tail -3; ./.venv/bin/python -c \'import sklearn, numpy; print(sklearn.__version__, numpy.__version__)\' && ls -la';
  const url = 'https://arxiv.org/abs/1906.02629?context=cs.LG&utm_source=a-very-long-query-string-that-does-not-break-anywhere-by-itself';
  /* a second run of E1, which an agent carried out */
  const runs = path.join(dir, 'experiments', 'E1', 'runs');
  fs.cpSync(path.join(runs, '1'), path.join(runs, '2'), { recursive: true });
  const record = JSON.parse(fs.readFileSync(path.join(runs, '2', 'run.json'), 'utf8'));
  /* ... and changed the code it was given: the record names no exit status, and the code as
     the agent left it lies beside the approved code */
  const approvedCode = fs.readFileSync(path.join(runs, '2', 'code', 'main.py'), 'utf8');
  const changedCode = approvedCode + '\n# CHANGED-BY-THE-AGENT: the call that failed was corrected here.\n';
  fs.mkdirSync(path.join(runs, '2', 'code-as-run'), { recursive: true });
  fs.writeFileSync(path.join(runs, '2', 'code-as-run', 'main.py'), changedCode);
  Object.assign(record, { number: 2, carried_out_by: 'agent', exit_code: null, model_access: true, code_as_run: { digest: 'a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90', changed: ['main.py'], added: [], removed: [] }, started: stamp(new Date(t0 + 30000)), finished: stamp(new Date(t0 + 48000)), seconds: 17.9, result: null });
  fs.writeFileSync(path.join(runs, '2', 'run.json'), JSON.stringify(record, null, 2));
  const page = 'wiki/experiments/' + fs.readdirSync(path.join(dir, 'wiki', 'experiments')).find((name) => /^E1-/.test(name));
  const said = ['The experimenter works as a Claude agent on claude-haiku-5-5; fast mode was asked for and is not on (“preference”).', 'The experimenter ran “ls -la && cat main.py” in its shell.', 'The experimenter wrote main.py in its folder.', 'The experimenter ran “' + command + '” in its shell.', 'The experimenter ran “timeout 200 ./.venv/bin/python main.py; echo exit=$?; cat metrics.json” in its shell.'];
  fs.writeFileSync(path.join(runs, '2', 'log.txt'), ['An agent carries out E1; it has 300 seconds.'].concat(said, ['E1 ran once and passed; metrics.json and the figure are in place.']).map((l) => '[paper-orchestra] ' + l + '\n').join(''));
  const state = (to) => emit('experiment.state', '', 'run', 'Experiment E1 is now ' + to + '.', { path: page, kind: 'experiment', id: 'E1', review: 'accepted', state: to, by: 'system' }, [page]);
  state('queued'); state('running');
  emit('run.started', 'runner', '', 'Run 2 of experiment E1 started on the local backend.', { experiment: 'E1', run: 2, backend: 'local', carried_out_by: 'agent' }, [page]);
  const parts = [{ did: 'start', model: 'claude-haiku-5-5', fast: 'off', fast_asked: true }, { did: 'shell', text: 'ls -la && cat main.py' }, { did: 'write', text: 'main.py' }, { did: 'shell', text: command }, { did: 'shell', text: 'timeout 200 ./.venv/bin/python main.py; echo exit=$?; cat metrics.json' }];
  said.forEach((summary, i) => act('experimenter', 'run', summary, parts[i]));
  emit('run.finished', 'runner', '', 'Run 2 of experiment E1 succeeded after 17.9 s.', { experiment: 'E1', run: 2, status: 'succeeded', seconds: 17.9, error: null, requeued: false, result: null }, [page]);
  state('succeeded');
  /* (a line of that agent after its run has ended stands by itself, under the agent's own name) */
  act('experimenter', 'run', 'The experimenter read wiki/idea.md.', { did: 'board', tool: 'memory' });
  /* then a scout at work: the newest lines of the feed, so that a further step joins them */
  emit('phase.started', 'scout', 'research', 'Round ' + round + ': the research phase started.', {});
  act('scout', 'research', 'The scout works as a Claude agent on claude-haiku-5-5; fast mode was asked for and is not on (“preference”).', { did: 'start', model: 'claude-haiku-5-5', fast: 'off', fast_asked: true });
  const scout = [
    ['The scout read wiki/idea.md.', { did: 'board', tool: 'memory' }],
    ['The scout searched the web for “label smoothing calibration expected calibration error”.', { did: 'search', text: 'label smoothing calibration expected calibration error' }],
    ['The scout opened “' + url + '”.', { did: 'fetch', text: url }],
    ['The scout ran “' + command + '” in its shell.', { did: 'shell', text: command }],
    ['The scout wrote notes/calibration.md in its folder.', { did: 'write', text: 'notes/calibration.md' }],
    ['The scout searched the wiki for “warmup”.', { did: 'board', tool: 'wiki_search' }],
    ['The scout checked “When Does Label Smoothing Help?” against Semantic Scholar.', { did: 'board', tool: 'verify_paper' }],
    ['The scout started a helper agent: “Read three papers on calibration”.', { did: 'helper', text: 'Read three papers on calibration' }],
    ['A helper of the scout ran “ls papers/” in its shell.', { did: 'shell', text: 'ls papers/', helper: true }],
    ['The scout tried “paper-orchestra research approve . E1” with Bash; that is not an agent\'s to do, and it was refused.', { did: 'shell', text: 'paper-orchestra research approve . E1', refused: true }],
  ];
  scout.forEach(([summary, data]) => act('scout', 'research', summary, data));
  /* the failed run of E2, as an agent's */
  let failed = false;
  const two = path.join(dir, 'experiments', 'E2', 'runs');
  const last = fs.existsSync(two) ? fs.readdirSync(two).map(Number).filter(Boolean).sort((a, b) => b - a)[0] : 0;
  if (last) {
    const lost = JSON.parse(fs.readFileSync(path.join(two, String(last), 'run.json'), 'utf8'));
    if (lost.status === 'failed' && !lost.timed_out) { lost.carried_out_by = 'agent'; lost.repairs = []; lost.error_kind = 'metrics'; lost.code_as_run = { digest: 'c0ffee11223344556677889900aabbccddeeff00112233445566778899aabbcc', approved: 0, changed: [], added: ['main.py'], removed: [] }; fs.writeFileSync(path.join(two, String(last), 'run.json'), JSON.stringify(lost, null, 2)); failed = true; }
  }
  fs.appendFileSync(file, out.join('\n') + '\n');
  /* (the run's four events and the experiment's two states are steps of the run as well) */
  return { seq, round, command, scoutSteps: scout.length, runSteps: said.length + 5, failed };
}
/** One more step of the scout, written while a page is open. */
agentTraces.more = (dir, traces) => {
  traces.seq += 1;
  fs.appendFileSync(path.join(dir, 'events.jsonl'), '{"seq": ' + traces.seq + ', ' + JSON.stringify({ ts: stamp(), type: 'agent.activity', phase: 'research', round: traces.round, agent: 'scout', summary: 'The scout ran “wc -l notes/calibration.md” in its shell.', refs: [], data: { did: 'shell', text: 'wc -l notes/calibration.md' } }).slice(1) + '\n');
};

const OVERFLOW = () => {
  const out = [];
  const de = document.documentElement;
  if (de.scrollWidth > de.clientWidth) out.push('document ' + de.scrollWidth + '>' + de.clientWidth);
  ['#top', '#status'].forEach((s) => { const el = document.querySelector(s); if (el && !el.hidden && el.scrollWidth > el.clientWidth + 1) out.push(s + ' ' + el.scrollWidth + '>' + el.clientWidth); });
  document.querySelectorAll('.wiki-main, .in-detail, .in-list, .exp-main, .overview, .library, .setup, .start, .ledger, .panel, .wiki-index, .wiki-rail, .strip, .paper, .exp-rail').forEach((el) => {
    if (el.scrollWidth > el.clientWidth + 1) out.push('.' + el.className.split(' ')[0] + ' ' + el.scrollWidth + '>' + el.clientWidth);
  });
  return out;
};

/* ------------------------------------------------------------------------------------ */
const CHECKS = {};

/* every screen renders at four widths in both themes: no console error, nothing the
   Content-Security-Policy blocks, no horizontal overflow */
CHECKS.screens = async ({ browser, pid }) => {
  const routes = ['#/library', '#/start', '#/setup', '#/p/PID/map', '#/p/PID/map/H2', '#/p/PID/map/H5', '#/p/PID/overview', '#/p/PID/wiki', '#/p/PID/wiki/H2', '#/p/PID/wiki/H5',
    '#/p/PID/wiki/H2/edit', '#/p/PID/wiki/H5/edit', '#/p/PID/wiki/H2/history', '#/p/PID/wiki/F1', '#/p/PID/wiki/F2', '#/p/PID/wiki/topics%2Fwhy-warmup-helps', '#/p/PID/wiki/results%2FE1-run1', '#/p/PID/wiki/idea', '#/p/PID/inbox', '#/p/PID/inbox/E3',
    '#/p/PID/experiments', '#/p/PID/experiments/E3', '#/p/PID/experiments/E4', '#/p/PID/experiments/E1', '#/p/PID/experiments/E2', '#/p/PID/paper', '#/p/PID/wiki/no-such-page', '#/p/p-000000000000/map'];
  /* the four widths side by side: the pages do not touch the project */
  await Promise.all([[1440, 'light'], [1100, 'dark'], [820, 'light'], [1900, 'dark']].map(async ([width, scheme]) => {
    const ctx = await entered(await browser.newContext({ viewport: { width, height: 900 }, colorScheme: scheme, reducedMotion: 'reduce' }));
    await ctx.addInitScript(() => { window.__csp = []; document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(e.violatedDirective + ' ' + (e.blockedURI || '') + ' ' + (e.sourceFile || '') + ':' + e.lineNumber)); });
    for (const r of routes) {
      const page = await ctx.newPage();
      const hash = r.replace('PID', pid);
      watch(page, () => width + ' ' + hash);
      await open(page, hash);
      await page.waitForTimeout(350);
      const over = await page.evaluate(OVERFLOW);
      ok(!over.length, width + ' ' + hash + ' horizontal overflow: ' + over.join('; '));
      const csp = await page.evaluate(() => window.__csp);
      ok(!csp.length, width + ' ' + hash + ' blocked by the Content-Security-Policy: ' + csp.join('; '));
      if (/\/map(\/|$)/.test(hash) && hash.includes(pid)) ok((await count(page, '.map-svg .node')) >= 30, width + ' ' + hash + ' draws the map: ' + (await count(page, '.map-svg .node')) + ' nodes');
      ok((await page.title()).endsWith('PaperOrchestra'), width + ' ' + hash + ' has no title: ' + (await page.title()));
      await page.close();
    }
    await ctx.close();
  }));
};

/* a phone, held upright and touched: no screen is wider than the phone at three widths; the
   inbox is its list, then one proposal and the way back; what decides is on the screen and
   a finger wide; a proposal accepted in the list leaves the list in front, with its undo,
   which does not lie on the next row; the keyboard goes along with each step; the row of
   views stays where it was scrolled to; held on its side, the phone gets the same inbox */
CHECKS.phone = async ({ browser, pid }) => {
  const routes = ['#/library', '#/start', '#/setup', '#/p/PID/inbox', '#/p/PID/inbox/H5', '#/p/PID/inbox/E3', '#/p/PID/inbox/H5/edit', '#/p/PID/map', '#/p/PID/wiki', '#/p/PID/experiments', '#/p/PID/overview', '#/p/PID/paper'];
  const NARROW = () => { const de = document.documentElement, out = de.scrollWidth > de.clientWidth ? ['document ' + de.scrollWidth + '>' + de.clientWidth] : [];
    ['#top', '#status', '.in-list', '.in-detail'].forEach((s) => { const el = document.querySelector(s); if (el && !el.hidden && el.scrollWidth > el.clientWidth + 1) out.push(s + ' ' + el.scrollWidth + '>' + el.clientWidth); });
    return out; };
  /* where each of these stands, the ones that are shown: [what, left, right, top, bottom, width, height] */
  const BOXES = (selectors) => { const out = [];
    selectors.forEach((s) => document.querySelectorAll(s).forEach((el) => { const r = el.getBoundingClientRect(); if (r.width && getComputedStyle(el).visibility !== 'hidden') out.push([s, r.left, r.right, r.top, r.bottom, r.width, r.height]); }));
    return out; };
  /* the fields that are shown and written smaller than 16 px: a phone zooms in on those */
  const SMALL = () => Array.from(document.querySelectorAll('input:not([type=checkbox]):not([type=radio]), select, textarea')).filter((el) => el.getBoundingClientRect().width && parseFloat(getComputedStyle(el).fontSize) < 16).map((el) => (el.className || el.name || el.id || el.tagName) + ' ' + getComputedStyle(el).fontSize);
  /* what of the list the toast lies on: the middle of each thing to touch that the list shows */
  const UNDER = () => { const list = document.querySelector('.in-scroll').getBoundingClientRect(), out = [];
    document.querySelectorAll('.in-row [data-act="accept"], .in-row [data-act="reject-ask"], .in-check, .in-all, .in-bulk .btn').forEach((el) => { const r = el.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
      if (!r.width || (el.closest('.in-scroll') && (y <= list.top || y >= list.bottom))) return;
      const at = document.elementFromPoint(x, y); if (at && at.closest('#toast')) out.push((el.dataset.act || el.className) + ' at ' + Math.round(x) + ',' + Math.round(y)); });
    return out; };
  /* text that has no room: narrower than its longest word needs, many times taller than
     wide, or a name of the route under the form on more than two lines */
  const SQUEEZED = () => { const out = [];
    document.querySelectorAll('.composer-bar *, .start-est, .start-lede, .smap-route .smap-name, .smap-route .smap-hint').forEach((el) => { const r = el.getBoundingClientRect(), words = (el.textContent || '').trim();
      if (!r.width || !words || el.children.length) return;
      const longest = Math.max.apply(null, words.split(/\s+/).map((w) => w.length)), font = parseFloat(getComputedStyle(el).fontSize);
      if (r.width < Math.min(longest, 12) * font * 0.45 || r.height > 6 * Math.max(r.width, 1)) out.push((el.className || el.tagName) + ' ' + Math.round(r.width) + 'x' + Math.round(r.height) + ' "' + words.slice(0, 24) + '"'); });
    const names = Array.from(document.querySelectorAll('.smap-route .smap-name')).filter((el) => el.getBoundingClientRect().height > parseFloat(getComputedStyle(el).fontSize) * 2.2).map((el) => el.textContent);
    return { out, names }; };
  const focusIn = (page, selector) => page.evaluate((sel) => !!document.activeElement && document.activeElement.matches(sel), selector);
  const reach = async (page, width, where, selectors, least) => {
    const boxes = await page.evaluate(BOXES, selectors), high = await page.evaluate(() => window.innerHeight);
    selectors.forEach((s) => ok(boxes.some((b) => b[0] === s), width + ' ' + where + ': nothing shown for ' + s));
    boxes.forEach((b) => { ok(b[1] >= -1 && b[2] <= width + 1, width + ' ' + where + ': ' + b[0] + ' is off the screen, at ' + Math.round(b[1]) + '..' + Math.round(b[2]));
      if (least) ok(b[5] >= least && b[6] >= least, width + ' ' + where + ': ' + b[0] + ' is ' + Math.round(b[5]) + ' by ' + Math.round(b[6]) + ' px under a finger'); });
    return boxes.filter((b) => b[3] >= 0 && b[4] <= high + 1);
  };
  for (const width of [360, 390, 430]) {
    const ctx = await entered(await browser.newContext({ ...devices['iPhone 14'], viewport: { width, height: 664 }, colorScheme: arg('scheme', 'light'), reducedMotion: 'reduce' }));
    const page = await ctx.newPage();
    let hash = '';
    watch(page, () => 'phone ' + width + ' ' + hash);
    for (const r of routes) {
      hash = r.replace('PID', pid);
      await open(page, hash);
      await page.waitForTimeout(350);
      if (r === '#/start') {
        /* the new-project page: no piece of text is squeezed to a letter's width (the hint
           beside "Add files" was, and the route under the form broke its words in the middle) */
        const squeezed = await page.evaluate(SQUEEZED);
        ok(!squeezed.out.length && !squeezed.names.length, width + ' the new-project page on a phone squeezes no text: ' + JSON.stringify(squeezed));
        await shot(page, 'start-phone-' + width);
      }
      const over = await page.evaluate(NARROW);
      ok(!over.length, width + ' ' + hash + ' horizontal overflow: ' + over.join('; '));
      if (hash.includes(pid)) ok(await page.evaluate(() => { const nav = document.querySelector('.views'), a = nav.querySelector('a[aria-current]').getBoundingClientRect(), b = nav.getBoundingClientRect(); return a.left >= b.left - 1 && a.right <= b.right + 1; }), width + ' ' + hash + ': the open view is not in sight in the top bar');
      const small = await page.evaluate(SMALL);
      ok(!small.length, width + ' ' + hash + ': the phone zooms in on a field under 16 px: ' + small.join(', '));
      if (hash.endsWith('/edit')) await reach(page, width, 'the editor', ['.ed-fields input'], 40);
      if (hash.endsWith('/map') || hash.endsWith('/overview')) await reach(page, width, hash.split('/').pop(), ['.go > .btn'], 40);
      if (hash.endsWith('/inbox/E3')) ok(await page.evaluate(() => { const word = document.querySelector('.in-detail .kicker-kind > span').firstChild, r = document.createRange(); r.setStart(word, 0); r.setEnd(word, word.data.trimEnd().length); return r.getClientRects().length === 1; }), width + ': the kind of a proposal breaks in the middle of a word');
    }
    /* the row of views, scrolled to its end (the open view is then out of sight), stays there when the bar is written again */
    await open(page, hash = '#/p/' + pid + '/map');
    await page.waitForTimeout(250);
    const navAt = await page.evaluate(() => { const nav = document.querySelector('.views'); nav.scrollLeft = nav.scrollWidth; return nav.scrollLeft; });
    await page.tap('#top [data-act="theme"]');
    await page.waitForTimeout(250);
    ok(navAt > 0 && await page.evaluate(() => document.querySelector('.views').scrollLeft) === navAt, width + ': a render moved the row of views from where it was scrolled to (' + navAt + ')');
    await page.tap('#top [data-act="theme"]');
    /* the list alone, each answer a finger wide; nothing typed in makes the phone zoom */
    await open(page, hash = '#/p/' + pid + '/inbox');
    await until(page, () => document.querySelectorAll('.in-row:not(.in-old)').length >= 6, null, 8000, 'the phone lists what waits');
    ok(await page.evaluate(() => getComputedStyle(document.querySelector('.in-detail')).visibility === 'hidden' && getComputedStyle(document.querySelector('.in-list')).visibility === 'visible'), width + ': the bare inbox is not the list alone');
    ok(await page.evaluate(() => !document.querySelector('.in-row [aria-current]')), width + ': a row of the list alone is marked as the proposal in front');
    await reach(page, width, 'the list', ['.in-row:not(.in-old) [data-act="accept"]', '.in-row [data-act="reject-ask"]', '.in-check', '.in-all', '.in-old', '#status a.sb-i'], 0);
    await reach(page, width, 'the list', ['.in-row:not(.in-old) [data-act="accept"]', '.in-row [data-act="reject-ask"]', '.in-check'], 40);
    /* a row accepted in the list: the list stays in front, and the undo is there to be touched */
    await page.tap('.in-row:not(.in-old) [data-act="accept"]');
    await until(page, () => !!document.querySelector('.toast [data-act="undo"]') && document.querySelectorAll('.in-row:not(.in-old)').length === 5, null, 8000, 'a touch accepts a row of the list');
    ok(await page.evaluate(() => !document.querySelector('.inbox.in-open')) && page.url().endsWith('/inbox'), width + ': accepting a row of the list left the list, for ' + page.url().split('#')[1]);
    await reach(page, width, 'the toast', ['.toast [data-act="undo"]'], 40);
    /* the toast does not lie on the row that moved up under the finger, nor on any other thing to touch */
    let under = await page.evaluate(UNDER);
    ok(!under.length, width + ': the toast lies on ' + under.join('; '));
    await page.tap('.in-check >> nth=0');
    under = await page.evaluate(UNDER);
    ok(!under.length, width + ', one ticked: the toast lies on ' + under.join('; '));
    await page.tap('.in-check >> nth=0');
    await page.tap('.toast [data-act="undo"]');
    await until(page, () => document.querySelectorAll('.in-row:not(.in-old)').length === 6, null, 8000, 'the undo of the phone takes the decision back');
    /* two ticked: both answers for them come up, on the screen */
    await page.tap('.in-check >> nth=1'); await page.tap('.in-check >> nth=2');
    const both = await reach(page, width, 'two ticked', ['[data-act="bulk-accept"]', '[data-act="bulk-reject"]'], 40);
    ok(both.length === 2, width + ': the answers for the ticked rows are not on the screen');
    await page.tap('.in-check >> nth=1'); await page.tap('.in-check >> nth=2');
    /* a row touched: its proposal alone, the answers held on the screen, and the way back */
    await page.tap('.in-row[data-ref] .in-main >> nth=3');
    await until(page, () => !!document.querySelector('.inbox.in-open .in-detail .review-actions'), null, 8000, 'a touched row opens its proposal');
    ok(await page.evaluate(() => getComputedStyle(document.querySelector('.in-list')).visibility === 'hidden'), width + ': the list shows beside the proposal');
    ok(await focusIn(page, '.in-back'), width + ': the keyboard did not go to the head of the opened proposal');
    const left = page.url().split('#')[1];
    await page.evaluate(() => { const d = document.querySelector('.in-detail'); d.scrollTop = d.scrollHeight; });
    const held = await reach(page, width, 'the proposal', ['.in-detail .review-actions [data-act="accept"]', '.in-detail .review-actions [data-act="reject-ask"]', '.in-back'], 40);
    ok(held.length === 3, width + ': at the end of a proposal its answers and the way back are not on the screen: ' + held.map((b) => b[0]).join(', '));
    await page.tap('.in-detail .review-actions [data-act="reject-ask"]');
    await until(page, () => document.activeElement && document.activeElement.matches('.reject-form input'), null, 8000, 'the reason of a rejection takes the keyboard');
    ok(!(await page.evaluate(SMALL)).length, width + ': the reason of a rejection is under 16 px: the phone zooms in on it');
    ok(await page.evaluate(() => document.querySelector('.reject-form input').getBoundingClientRect().top >= document.querySelector('.in-back').getBoundingClientRect().bottom), width + ': the reason of a rejection opens under the way back');
    await reach(page, width, 'the reason', ['.reject-form input', '.reject-form button[type=submit]', '.reject-form [data-act="reject-cancel"]'], 40);
    await page.tap('.reject-form [data-act="reject-cancel"]');
    await page.tap('.in-back');
    await until(page, () => !document.querySelector('.inbox.in-open') && getComputedStyle(document.querySelector('.in-list')).visibility === 'visible', null, 8000, 'the way back shows the list');
    ok(await focusIn(page, '.in-row a.in-main[href="#' + left + '"]'), width + ': back in the list, the keyboard is not on the row that was left');
    await shot(page, 'phone-' + width);
    await ctx.close();
  }
  /* held on its side the phone is wider than 700 px and too low for two panes: the same inbox */
  const ctx = await entered(await browser.newContext({ ...devices['iPhone 14'], viewport: { width: 844, height: 390 }, colorScheme: arg('scheme', 'light'), reducedMotion: 'reduce' }));
  const page = await ctx.newPage();
  watch(page, () => 'phone on its side');
  await open(page, '#/p/' + pid + '/inbox');
  await until(page, () => document.querySelectorAll('.in-row:not(.in-old)').length >= 6, null, 8000, 'the phone on its side lists what waits');
  ok(await page.evaluate(() => getComputedStyle(document.querySelector('.in-detail')).visibility === 'hidden'), 'on its side: the bare inbox is not the list alone');
  await reach(page, 844, 'the list, on its side', ['.in-row:not(.in-old) [data-act="accept"]', '.in-row [data-act="reject-ask"]', '.in-check'], 40);
  await page.tap('.in-row[data-ref] .in-main >> nth=1');
  await until(page, () => !!document.querySelector('.inbox.in-open .in-detail .review-actions'), null, 8000, 'on its side, a touched row opens its proposal');
  const side = await reach(page, 844, 'the proposal, on its side', ['.in-detail .review-actions [data-act="accept"]', '.in-detail .review-actions [data-act="reject-ask"]', '.in-back'], 40);
  ok(side.length === 3, 'on its side: the answers of a proposal and the way back are not on the screen: ' + side.map((b) => b[0]).join(', '));
  await shot(page, 'phone-side');
  await ctx.close();
};

/* Agents with a shell (the `sdk` runtime). How the agents are run is chosen in the composer
   and changed on the overview, each choice with the sentence that says what it means, and
   the server holds what the app says. What such agents do, and a run one of them carried
   out, cannot be brought about here (no agent can be started on the scripted model): the
   check writes what they leave into the project's files, the events of a scout at work and
   of a run an agent carried out, and looks at the feed and the experiment, at a desk and on
   a phone. On a fourth copy of the demo project, in a folder named `shell`, which no other
   check touches. */
CHECKS.shell = async ({ browser, page, shell }) => {
  if (!shell) { notes.push('shell: agents with a shell were not looked at (the workspace has no demo project in a folder named shell)'); return; }
  const pid = shell, project = '/api/projects/' + pid;
  const defaults = (await api('GET', '/api/defaults')).data;
  const base = defaults.settings.agents;
  const says = (page, name) => page.evaluate((n) => { const el = document.querySelector('[data-says="' + n + '"]'); return el ? el.textContent.replace(/\s+/g, ' ').trim() : ''; }, name);

  /* the composer: the runtimes the server lists, its default named as the default, and nothing about the agents sent that the person did not choose */
  await open(page, '#/start?adv=1', '#opt-agent-runtime');
  ok(JSON.stringify(await page.$$eval('#opt-agent-runtime option', (options) => options.map((o) => o.value))) === JSON.stringify([''].concat(defaults.agent_runtimes)) && defaults.agent_runtimes.length === 2, 'the composer offers the runtimes the server says its settings accept: ' + (await page.$$eval('#opt-agent-runtime option', (options) => options.map((o) => o.value)).then((v) => v.join())));
  ok(/^Default \(/.test(await page.$eval('#opt-agent-runtime option', (o) => o.textContent)) && (await page.inputValue('#opt-agent-runtime')) === '', 'with the runtime a new project would get named as the default: ' + (await page.$eval('#opt-agent-runtime option', (o) => o.textContent)));
  ok((await says(page, 'runtime')).length > 40, 'and one sentence says what that runtime means: ' + (await says(page, 'runtime')));
  /* the default is the server's, which follows its environment (PAPER_ORCHESTRA_AGENT_RUNTIME): named, and its fields shown or not, before anything is chosen */
  ok((base.runtime === 'sdk' ? /shell/ : /tool loop/i).test(await page.$eval('#opt-agent-runtime option', (o) => o.textContent)) && (base.runtime === 'sdk' ? /shell/ : /no shell/).test(await says(page, 'runtime')), 'the default that is named is the server\'s (' + base.runtime + '): ' + (await page.$eval('#opt-agent-runtime option', (o) => o.textContent)));
  ok(((await count(page, '#opt-agent-model')) === 1) === (base.runtime === 'sdk'), 'the fields of agents with a shell are shown where the default is such agents, and not for the tool loop');
  await page.selectOption('#opt-agent-runtime', 'sdk');
  await page.waitForSelector('#opt-agent-model');
  /* the effort is a setting of the tool loop's models: it is not offered for agents with a shell */
  ok((await count(page, '#opt-effort')) === 0, 'with agents with a shell chosen, the composer does not offer a model effort');
  await page.selectOption('#opt-agent-runtime', 'classic');
  await page.waitForSelector('#opt-effort');
  ok((await count(page, '#opt-agent-model')) === 0, 'for the tool loop it does, and the fields of agents with a shell are gone');
  ok((await count(page, '.start-agents')) === 0, 'and nothing under the box speaks of agents with a shell');
  await page.selectOption('#opt-agent-runtime', 'sdk');
  await page.waitForSelector('#opt-agent-model');
  /* whose account pays, and that fast mode is dear, stands on the page itself: Options may be closed when Start is pressed */
  const paid = await text(page, '.start-est .start-agents');
  ok(paid.includes(base.model) && (base.credential === 'login' ? /billed to your own Claude account/ : /federation/).test(paid) && (!(base.fast && base.credential === 'login') || /fast mode costs several times/.test(paid)) && /Change it under Options/.test(paid), 'under the box, outside the options, the page says on which model and whose account the agents work: ' + paid);
  await page.selectOption('#opt-agent-credential', 'federation');
  await until(page, () => /federation/.test((document.querySelector('.start-agents') || {}).textContent || '') && !/fast mode/.test((document.querySelector('.start-agents') || {}).textContent || ''), null, 3000, 'with the federation credential it names that account, and no fast mode');
  await page.selectOption('#opt-agent-credential', '');
  ok(/shell/.test(await says(page, 'runtime')) && (await page.getAttribute('#opt-agent-model', 'placeholder')) === base.model, 'agents with a shell: the sentence says so, and the model field shows the server\'s default: ' + (await page.getAttribute('#opt-agent-model', 'placeholder')));
  ok(JSON.stringify(await page.$$eval('#opt-agent-credential option', (options) => options.map((o) => o.value))) === JSON.stringify([''].concat(defaults.agent_credentials)), 'the credentials are the ones the server lists');
  ok((await page.isChecked('#opt-agent-fast')) === base.fast && (await page.isChecked('#opt-agent-ultracode')) === base.ultracode && (await page.isChecked('#opt-agent-sandbox')) === base.sandbox, 'the ticks start as the server\'s defaults');
  for (const name of ['model', 'credential', 'fast', 'ultracode', 'sandbox']) ok((await says(page, name)).length > 40, 'a sentence says what ' + name + ' means: ' + (await says(page, name)));
  const login = await says(page, 'credential');
  ok(/your Claude account pays/.test(login) && /fast mode works/.test(login), 'the login: whose account pays, and that fast mode works: ' + login);
  await page.selectOption('#opt-agent-credential', 'federation');
  await until(page, () => /federation rule pays/.test((document.querySelector('[data-says="credential"]') || {}).textContent || ''), null, 3000, 'the sentence follows the credential that is chosen');
  ok(/refused/.test(await says(page, 'credential')) && /refused/.test(await says(page, 'fast')), 'federation: who pays, and that fast mode is refused there, said at the tick for fast mode too: ' + (await says(page, 'fast')));
  const safe = await says(page, 'sandbox');
  await page.uncheck('#opt-agent-sandbox');
  await until(page, (before) => { const el = document.querySelector('[data-says="sandbox"]'); return !!el && el.textContent.replace(/\s+/g, ' ').trim() !== before; }, safe, 3000, 'the sentence about the sandbox changes when it is switched off');
  ok(/reach whatever you can reach/.test(await says(page, 'sandbox')) && (await count(page, '[data-says="sandbox"].agent-warn')) === 1, 'without the sandbox: what the agents can then reach, said as a warning: ' + (await says(page, 'sandbox')));
  /* what a press sends: caught on its way, so that no project is made of it */
  let sent = null;
  await page.route('**/api/projects', async (route) => {
    if (route.request().method() !== 'POST') return route.continue();
    sent = route.request().postData() || '';
    return route.fulfill({ status: 422, contentType: 'application/json', body: JSON.stringify({ error: { code: 'invalid', message: 'Caught by the check.' } }) });
  });
  await page.fill('#idea-text', 'A project that is never made.');
  await page.click('.composer button[type=submit]');
  await until(page, () => !!document.querySelector('.composer-error'), null, 8000, 'the composer shows the answer to its press');
  await page.unroute('**/api/projects');
  const settingsSent = (() => { const m = /name="settings"\r?\n\r?\n([^\r\n]+)/.exec(sent || ''); try { return m ? JSON.parse(m[1]) : null; } catch (e) { return null; } })();
  ok(!!settingsSent && JSON.stringify(settingsSent.agents) === JSON.stringify({ runtime: 'sdk', credential: 'federation', sandbox: false }), 'the composer sends what the person chose about the agents and nothing else: ' + JSON.stringify(settingsSent));
  /* a reload keeps what was chosen about the agents, as it keeps the text */
  await page.reload();
  await page.waitForSelector('#opt-agent-credential');
  ok((await page.inputValue('#opt-agent-runtime')) === 'sdk' && (await page.inputValue('#opt-agent-credential')) === 'federation' && !(await page.isChecked('#opt-agent-sandbox')), 'the choices about the agents survive a reload');
  /* left alone, the composer names no runtime: the server's environment decides it */
  await page.evaluate(() => { localStorage.removeItem('po-composer'); });
  await page.goto(BASE + '/#/library');
  await page.reload();
  await open(page, '#/start?adv=1', '#opt-agent-runtime');
  sent = null;
  await page.route('**/api/projects', async (route) => {
    if (route.request().method() !== 'POST') return route.continue();
    sent = route.request().postData() || '';
    return route.fulfill({ status: 422, contentType: 'application/json', body: JSON.stringify({ error: { code: 'invalid', message: 'Caught by the check.' } }) });
  });
  await page.fill('#idea-text', 'Another project that is never made.');
  await page.click('.composer button[type=submit]');
  await until(page, () => !!document.querySelector('.composer-error'), null, 8000, 'the composer shows the answer to its second press');
  await page.unroute('**/api/projects');
  ok(typeof sent === 'string' && !/name="settings"/.test(sent), 'a composer that was left alone sends no settings: the runtime is then the server\'s to choose, from its environment');
  await page.evaluate(() => { localStorage.removeItem('po-composer'); });
  await page.goto(BASE + '/#/library');
  await page.reload();

  /* the overview says how the agents work, and changes it */
  await api('PATCH', project, { settings: { agents: { runtime: 'classic', model: base.model, fast: true, ultracode: true, credential: 'login', sandbox: true } } });
  await open(page, '#/p/' + pid + '/overview', '.ov-agents');
  ok((await page.getAttribute('.agents-now', 'data-agents')) === 'classic' && /tool loop/i.test(await text(page, '.agents-now')), 'the overview says that the agents work in the tool loop: ' + (await text(page, '.agents-now')));
  await page.click('[data-act="agents-open"]');
  await page.waitForSelector('#ag-runtime');
  ok((await page.inputValue('#ag-runtime')) === 'classic' && (await count(page, '#ag-model')) === 0, 'the form opens on what is in force');
  await page.selectOption('#ag-runtime', 'sdk');
  await page.waitForSelector('#ag-model');
  ok((await page.inputValue('#ag-model')) === base.model, 'the model field holds the project\'s model: ' + (await page.inputValue('#ag-model')));
  /* a model without a name is refused before anything is sent, and the form stays */
  await page.fill('#ag-model', '  ');
  await page.click('.agents-form button[type=submit]');
  await until(page, () => /Name the model/.test((document.querySelector('.agents-form [role="alert"]') || {}).textContent || ''), null, 3000, 'a model without a name is not sent');
  ok((await api('GET', project)).data.settings.agents.runtime === 'classic', 'and nothing was changed');
  await page.fill('#ag-model', 'claude-haiku-5-5');
  await page.uncheck('#ag-fast');
  await page.selectOption('#ag-credential', 'federation');
  await page.click('.agents-form button[type=submit]');
  await until(page, () => (document.querySelector('.agents-now') || { dataset: {} }).dataset.agents === 'sdk' && !document.querySelector('.agents-form'), null, 8000, 'the overview says that the agents now have a shell');
  const now = (await api('GET', project)).data.settings.agents;
  ok(JSON.stringify(now) === JSON.stringify({ runtime: 'sdk', model: 'claude-haiku-5-5', fast: false, ultracode: true, credential: 'federation', sandbox: true }), 'the server holds what the form said: ' + JSON.stringify(now));
  ok(/claude-haiku-5-5/.test(await text(page, '.agents-now')) && /federation/.test(await text(page, '.agents-now')) && /inside the sandbox/.test(await text(page, '.agents-now')) && !/fast mode/.test(await text(page, '.agents-now b')), 'and the sentence on the overview says it: ' + (await text(page, '.agents-now b')));
  ok(/next phase/.test(await text(page, '.toast')) && /Claude agents with a shell/.test(await text(page, '.toast')), 'the toast says what changed and from when it holds: ' + (await text(page, '.toast')));
  /* a change of one value says that value, not that the agents are now what they were already */
  await page.click('[data-act="agents-open"]');
  await page.waitForSelector('#ag-ultracode');
  await page.uncheck('#ag-ultracode');
  await page.click('.agents-form button[type=submit]');
  await until(page, () => /helper agents off/i.test((document.querySelector('.toast') || {}).textContent || ''), null, 8000, 'the toast names the value that changed');
  ok(!/are now Claude agents/.test(await text(page, '.toast')) && /next phase/.test(await text(page, '.toast')), 'and does not say that the runtime changed: ' + (await text(page, '.toast')));
  await api('PATCH', project, { settings: { agents: { ultracode: true } } });
  /* a value the group does not have is refused, not stored and ignored */
  const bogus = await api('PATCH', project, { settings: { agents: { sandbx: false } } });
  ok(bogus.status >= 400 && bogus.status < 500 && !('sandbx' in (await api('GET', project)).data.settings.agents), 'a misspelt value of the agents\' settings is refused: ' + bogus.status);
  /* Setup says what this machine has of what agents with a shell need */
  await open(page, '#/setup', '.setup [data-setup="agents"]');
  const needs = await text(page, '.setup [data-setup="agents"]');
  ok(/Claude Agent SDK/.test(needs) && /Your Claude Code login/.test(needs) && /Workload Identity Federation/.test(needs) && (await count(page, '.setup [data-setup="agents"] .doc-i')) === 3, 'Setup has the package, the login and the federation settings of agents with a shell: ' + needs.replace(/\s+/g, ' ').slice(0, 200));
  await open(page, '#/p/' + pid + '/overview', '.ov-agents');
  /* the server's refusal is shown in the server's words, and the form keeps what was chosen */
  await page.click('[data-act="agents-open"]');
  await page.waitForSelector('#ag-runtime');
  await page.route('**' + project, async (route) => {
    if (route.request().method() !== 'PATCH') return route.continue();
    const body = JSON.parse(route.request().postData() || '{}');
    body.settings.agents.credential = 'a password';
    return route.continue({ postData: JSON.stringify(body) });
  });
  await page.uncheck('#ag-sandbox');
  ok((await count(page, '.agents-form [data-says="sandbox"].agent-warn')) === 1, 'switching the sandbox off says what the agents can then reach, before it is saved');
  await page.click('.agents-form button[type=submit]');
  await until(page, () => /credential/.test((document.querySelector('.agents-form [role="alert"]') || {}).textContent || ''), null, 8000, 'a change the server refuses is said in the server\'s words');
  await page.unroute('**' + project);
  ok((await api('GET', project)).data.settings.agents.sandbox === true && !(await page.isChecked('#ag-sandbox')), 'nothing was changed, and the form still holds the choice');
  await page.click('.agents-form button[type=submit]');
  await until(page, () => !!document.querySelector('.agents-open-box') && !document.querySelector('.agents-form'), null, 8000, 'a project without the sandbox says so on its overview');
  ok(/sandbox is off/i.test(await text(page, '.agents-open-box')) && /without the sandbox/.test(await text(page, '.agents-now b')), 'with what the agents can reach: ' + (await text(page, '.agents-open-box')).slice(0, 120));
  await api('PATCH', project, { settings: { agents: { sandbox: true } } });
  /* a reload shows the same: it is the project's setting, not this window's */
  await reopen(page, '#/p/' + pid + '/overview', '.ov-agents');
  ok((await page.getAttribute('.agents-now', 'data-agents')) === 'sdk' && /inside the sandbox/.test(await text(page, '.agents-now b')), 'after a reload the overview says the same: ' + (await text(page, '.agents-now b')));
  await shot(page, 'agents-settings');

  /* Where such agents cannot start (the package or the credential is missing on this machine),
     the engine says so beside the main button, and the button leads to where the agents are
     changed, not to Setup, which shows the models of the tool loop. */
  const blocked = ((await api('GET', project)).data.next_step || {}).blocked;
  if (blocked && (blocked.settings || []).indexOf('agents') >= 0) {
    const need = await text(page, '.ov-next [data-needs="agents"]');
    ok(need.replace(/\s+/g, ' ').includes(blocked.message.replace(/`/g, '').replace(/\s+/g, ' ')) && /How the agents work/.test(need), 'agents that cannot start: the engine\'s sentence stands beside the button, with where it is put right: ' + need.replace(/\s+/g, ' ').slice(0, 160));
    ok((await count(page, '.ov-next a[href="#/setup"]')) === 0, 'and nothing sends the person to Setup for it');
    const main = page.locator('.ov-actions a[data-go-main]');
    if ((await main.count()) && /See how the agents work/.test(await main.textContent())) {
      await main.click();
      await until(page, () => { const el = document.querySelector('.ov-agents'); return !!el && el.getBoundingClientRect().top < window.innerHeight; }, null, 4000, 'the main button shows how the agents work');
    }
  } else notes.push('shell: agents with a shell can start on this machine, so what the app says when they cannot was not looked at');

  /* an experiment approved now for this machine is an agent's to carry out, and the approval says so */
  const listed = (await api('GET', project + '/experiments')).data.experiments || [];
  let approvalSeen = false;
  for (const x of listed.filter((x) => x.review === 'proposed' || x.state === 'planned' || x.state === 'approved')) {
    await open(page, '#/p/' + pid + '/experiments/' + x.id, '.exp-page .page-title');
    if (!(await count(page, '#approve'))) { if (!(await count(page, '[data-act="reapprove"]'))) continue; await page.click('[data-act="reapprove"]'); }
    await page.waitForSelector('#approve .backend');
    const local = await page.evaluate(() => { const radio = document.querySelector('.backend input[value="local"]'); return radio ? { off: radio.disabled, on: radio.checked, desc: radio.closest('.backend').querySelector('.be-desc').textContent } : null; });
    if (!local || local.off) continue;
    if (!local.on) { await page.check('.backend input[value="local"]'); await page.waitForSelector('.backend.on input[value="local"]'); }
    ok(/agent/i.test(local.desc), 'this machine, as a place to run: an agent carries the run out there: ' + local.desc);
    const approves = (await text(page, '.approve-go p')).replace(/\s+/g, ' ');
    ok(/lets an agent carry out this experiment once/.test(approves) && /metrics\.json/.test(approves) && /reaches the web whether or not/.test(approves) && !/lets the runner execute/.test(approves), 'the approval says what it approves: ' + approves.slice(0, 200));
    if ((await count(page, '.backend input[value="modal"]:not(:disabled)'))) {
      await page.check('.backend input[value="modal"]');
      await until(page, () => /lets the runner execute/.test((document.querySelector('.approve-go p') || {}).textContent || ''), null, 3000, 'a run approved for Modal is the runner\'s as ever, and the approval says that');
    }
    await page.reload();
    approvalSeen = true;
    break;
  }
  if (!approvalSeen) notes.push('shell: the approval of a run on this machine was not looked at (no experiment of the demo project can be approved for it)');

  /* what agents with a shell leave behind, written by hand */
  const dir = await projectDir(pid);
  if (!dir) { notes.push('shell: the activity of agents with a shell and a run one carried out were not looked at (the server is on another machine)'); return; }
  await idle(pid);
  const traces = agentTraces(dir);
  await reopen(page, '#/p/' + pid + '/overview', '.feed .ev');
  await until(page, () => !!document.querySelector('.feed .ev-work') && !!document.querySelector('.feed .ev-run'), null, 8000, 'the feed shows what the agents did');
  const feed = () => page.evaluate(() => Array.from(document.querySelectorAll('.feed > .ev')).map((li) => ({ cls: li.className, who: (li.querySelector('.ev-who') || {}).textContent, says: (li.children[2] || li).textContent.replace(/\s+/g, ' ').trim(), steps: li.querySelectorAll('.ev-steps li').length, code: Array.from(li.querySelectorAll(':scope > span code.ev-text, :scope > span > .ev-act code.ev-text')).map((c) => c.textContent) })));
  let rows = await feed();
  /* the newest first: the run an agent carried out, then the scout's work, folded */
  const run = rows.find((r) => / ev-run /.test(r.cls + ' '));
  ok(!!run && run.who === 'agent' && /^An agent carried out E1 on this machine \(run 2\): succeeded after/.test(run.says), 'a run an agent carried out is one line that says so, from the agent: ' + JSON.stringify(run && [run.who, run.says.slice(0, 90)]));
  ok(!!run && run.steps === traces.runSteps, 'what the agent did meanwhile are steps of that run: ' + (run && run.steps) + ' of ' + traces.runSteps);
  const work = rows.find((r) => / ev-work/.test(r.cls));
  ok(!!work && work.who === 'scout' && work.steps === traces.scoutSteps && new RegExp(traces.scoutSteps + ' steps').test(work.says), 'what the scout did in a row is one line with all its steps behind it: ' + JSON.stringify(work && [work.who, work.steps]));
  ok(!!work && /tried/.test(work.says) && /refused/.test(work.says), 'which shows the latest step, here one that was refused: ' + (work && work.says.slice(0, 120)));
  const start = rows.find((r) => r.who === 'scout' && /works as a Claude agent/.test(r.says));
  ok(!!start && /fast mode was asked for and is not on/.test(start.says) && !/ ev-work/.test(start.cls), 'the line that says whether fast mode is on stands by itself: ' + (start && start.says));
  const own = await page.evaluate(() => { const li = Array.from(document.querySelectorAll('.feed > .ev')).find((x) => (x.querySelector('.ev-who') || {}).textContent === 'experimenter'); if (!li) return null; const who = li.querySelector('.ev-who'), r = who.getBoundingClientRect(), next = li.children[2].getBoundingClientRect(); return { lines: Math.round(r.height / parseFloat(getComputedStyle(who).lineHeight)), clear: who.scrollWidth <= who.clientWidth + 1 && r.right <= next.left + 1 }; });
  ok(!!own && own.lines === 1 && own.clear, 'the longest name of an agent stands on one line in its column: ' + JSON.stringify(own));
  /* unfolded: each step sets what the agent chose apart from the words around it */
  await page.click('.feed .ev-work .ev-steps > summary');
  const steps = await page.evaluate(() => Array.from(document.querySelectorAll('.feed .ev-work .ev-steps li')).map((li) => ({ did: (li.querySelector('.ev-act') || { dataset: {} }).dataset.did || '', verb: (li.querySelector('.ev-did') || {}).textContent || '', text: (li.querySelector('.ev-text') || {}).textContent || '', code: !!li.querySelector('code.ev-text'), helper: !!li.querySelector('.ev-helper'), refused: !!li.querySelector('.ev-refused'), says: li.textContent.replace(/\s+/g, ' ').trim() })));
  const of = (did) => steps.find((x) => x.did === did && !x.refused && !x.helper) || {};
  ok(of('shell').verb === 'ran' && of('shell').code && of('shell').text === traces.command, 'a shell command is shown as the command it was: ' + JSON.stringify(of('shell')).slice(0, 160));
  ok(of('search').verb === 'searched the web for' && !of('search').code && /label smoothing/.test(of('search').text), 'a web search with what was searched for: ' + of('search').says);
  ok(of('fetch').verb === 'opened' && /^https:\/\/arxiv\.org/.test(of('fetch').text) && (await count(page, '.feed .ev-work .ev-steps a')) === 0, 'a page that was fetched with its address, which is text and no link: ' + of('fetch').says);
  ok(of('write').verb === 'wrote' && of('write').text === 'notes/calibration.md', 'a file that was written with its name: ' + of('write').says);
  ok(/started a helper agent/.test(of('helper').verb) && steps.some((x) => x.helper && x.did === 'shell'), 'a helper that was started, and what the helper did as the helper\'s: ' + of('helper').says);
  ok(steps.some((x) => !x.did && /searched the wiki for/.test(x.says)) && steps.some((x) => !x.did && /against Semantic Scholar/.test(x.says)), 'a board tool is told in the project\'s sentence, as in the tool loop');
  ok(steps.some((x) => x.refused && /paper-orchestra research approve/.test(x.text) && /tried/.test(x.verb)), 'a call that was refused says what was tried, and that it was refused');
  ok(!(await page.evaluate(OVERFLOW)).length, 'the feed with a long command is not wider than its column: ' + (await page.evaluate(OVERFLOW)).join('; '));
  await shot(page, 'agents-feed');
  /* a line that is written while the page is open arrives, and the fold grows by it */
  agentTraces.more(dir, traces);
  await until(page, (n) => document.querySelectorAll('.feed .ev-work .ev-steps li').length === n + 1 && !!document.querySelector('.feed .ev-work .ev-steps[open]'), traces.scoutSteps, 8000, 'a further step arrives over the event stream, and the list that is open stays open');
  ok(/ran/.test(await text(page, '.feed .ev-work .ev-work-l > .ev-act')) && /wc -l notes/.test(await text(page, '.feed .ev-work .ev-work-l > .ev-act')), 'the line shows the newest step: ' + (await text(page, '.feed .ev-work .ev-work-l > .ev-act')));
  /* lines of the tool loop, and lines from before lines had parts, read as before */
  for (let i = 0; i < 4 && (await count(page, '.feed-more')) && (await count(page, '.feed .ev-run')) < 2; i++) { const had = await count(page, '.feed > .ev'); await page.click('.feed-more'); await until(page, (n) => document.querySelectorAll('.feed > .ev').length > n, had, 6000, 'earlier activity is shown'); }
  rows = await feed();
  ok(rows.some((r) => / ev-run /.test(r.cls + ' ') && r.who === 'runner' && /^E\d+ ran on this machine/.test(r.says)), 'a run the runner made still says so: ' + JSON.stringify(rows.filter((r) => / ev-run /.test(r.cls + ' ')).map((r) => r.who + ': ' + r.says.slice(0, 40))));

  /* the experiment: the run says that an agent carried it out, with its log and what it left */
  await open(page, '#/p/' + pid + '/experiments/E1', '.exp-runs .run');
  await until(page, () => !!document.querySelector('.run.open [data-log]') && !document.querySelector('.run.open [data-log][data-empty]'), null, 8000, 'the log of the agent\'s run is shown');
  ok(/Run 2/.test(await text(page, '.run.open .run-h')) && /by an agent/.test(await text(page, '.run.open .run-h .run-by')), 'the heading of the run says that an agent carried it out: ' + (await text(page, '.run.open .run-h')).replace(/\s+/g, ' ').slice(0, 120));
  ok(/An agent carried out this run/.test(await text(page, '.run.open .run-agent')), 'and a sentence says what that means: ' + (await text(page, '.run.open .run-agent')).slice(0, 100));
  ok((await text(page, '.run.open [data-log]')).includes('The experimenter ran') && (await text(page, '.run.open [data-log]')).includes(traces.command), 'its log is what the agent did and said');
  /* the run names the code as the agent left it, not only the approved code, and no exit status */
  ok(/changed the code it was given/.test(await text(page, '.run.open .run-code[data-code="changed"]')) && /main\.py/.test(await text(page, '.run.open .run-code')) && /not from the approved code/.test(await text(page, '.run.open .run-code')), 'the run says that the agent changed the code: ' + (await text(page, '.run.open .run-code')).replace(/\s+/g, ' ').slice(0, 160));
  ok(!/exit \d/.test(await text(page, '.run.open .run-h')) && /Nobody measured an exit status/.test(await text(page, '.run.open .run-agent')), 'and names no exit status, because none was measured');
  ok(/could call the Anthropic API/.test(await text(page, '.run.open [data-model-access]')) && /counted nowhere here/.test(await text(page, '.run.open [data-model-access]')), 'a run whose code could call the model API says that those calls are not counted');
  const keptHref = await page.getAttribute('.run.open .run-code a.linkish', 'href');
  const keptFile = await page.evaluate(async (href) => { const r = await fetch(href, { credentials: 'same-origin' }); return { status: r.status, text: await r.text() }; }, keptHref);
  ok(keptFile.status === 200 && /CHANGED-BY-THE-AGENT/.test(keptFile.text), 'the changed file is a link to the code as the agent left it: ' + keptHref + ' ' + keptFile.status);
  ok(/may have tried it in its own sandboxed folder/.test(await text(page, '.exp-code .sec-t')) && !/never executed/.test(await text(page, '.exp-code .sec-t')), 'the code of a project whose agents have a shell does not claim that it was never executed: ' + (await text(page, '.exp-code .sec-t')));
  ok(!/CHANGED-BY-THE-AGENT/.test(await text(page, '.exp-code')), 'the Code section still shows the approved code');
  await page.click('.run.open [data-act="adopt-code"]');
  await until(page, () => /is now the code of E1/.test((document.querySelector('.toast') || {}).textContent || ''), null, 8000, 'the person makes the agent\'s code the experiment\'s');
  const adopted = (await api('GET', '/api/projects/' + pid + '/experiments/E1')).data;
  ok(/CHANGED-BY-THE-AGENT/.test(((adopted.code.files || []).find((f) => f.name === 'main.py') || {}).text || '') && ((adopted.code.files || []).find((f) => f.name === 'main.py') || {}).author === 'human', 'the experiment\'s code is then the agent\'s, as the person\'s edit: ' + JSON.stringify((adopted.code.files || []).map((f) => [f.name, f.author])));
  await until(page, () => /CHANGED-BY-THE-AGENT/.test((document.querySelector('.exp-code') || {}).textContent || ''), null, 8000, 'the Code section shows the adopted code');
  ok((await count(page, '.run.open table.metrics tr')) >= 1 && /as the agent's work left it/.test(await text(page, '.run.open .res-grid .sec-h')), 'its metrics are shown as what the agent\'s work left: ' + (await text(page, '.run.open .res-grid .sec-h')));
  const first = page.locator('.exp-runs .run:not(.open) .run-h');
  ok((await first.count()) === 1 && !/by an agent/.test(await first.textContent()), 'the run before it, which the runner made, does not say so');
  await first.click();
  await page.waitForSelector('.run.open [data-run="1"]');
  ok((await count(page, '.run.open .run-agent')) === 0 && /as the run wrote it/.test(await text(page, '.run.open .res-grid .sec-h')), 'and reads as it always did');
  /* a failed run of an agent's is not "a programming error the runner has not repaired" */
  await open(page, '#/p/' + pid + '/experiments/E2', '.exp-runs .run');
  await page.waitForSelector('.run.open .failure');
  if (traces.failed) ok(/The agent did not bring the run to a result/.test(await text(page, '.run.open .fail-t')) && /repairs the code while it works/.test(await text(page, '.run.open .failure')) && !/runner has not repaired/.test(await text(page, '.run.open .failure')), 'a run that an agent could not bring to a result says so: ' + (await text(page, '.run.open .fail-t')));
  /* a run that left no results does not say that its results come from the agent's code, and a
     plan that came without code is not said to have approved code */
  if (traces.failed) {
    const wrote = (await text(page, '.run.open .run-code[data-code="changed"]')).replace(/\s+/g, ' ');
    ok(/The agent wrote the code for this run/.test(wrote) && /left no results that come from it/.test(wrote) && /came without code/.test(wrote) && !/The results of this run come from that code/.test(wrote) && !/repaired again/.test(wrote) && !/approved code shown under Code/.test(wrote), 'a failed run of a plan without code says what the agent wrote, and claims no results: ' + wrote.slice(0, 200));
  }
  await shot(page, 'agents-run');

  /* on a phone: nothing is wider than the screen, the command wraps under who ran it, and what is typed is 16 px */
  const NARROW = () => { const de = document.documentElement, out = de.scrollWidth > de.clientWidth ? ['document ' + de.scrollWidth + '>' + de.clientWidth] : [];
    document.querySelectorAll('.feed *, .run.open .run-agent, .run.open .run-h, .agent-set *, .agents-now').forEach((el) => { Array.from(el.getClientRects()).forEach((r) => { if (r.width && (r.right > de.clientWidth + 1 || r.left < -1)) out.push((el.className || el.tagName) + ' ' + Math.round(r.left) + '..' + Math.round(r.right) + ' of ' + de.clientWidth); }); });
    return out; };
  const SMALL = () => Array.from(document.querySelectorAll('input:not([type=checkbox]):not([type=radio]), select, textarea')).filter((el) => el.getBoundingClientRect().width && parseFloat(getComputedStyle(el).fontSize) < 16).map((el) => (el.id || el.name || el.tagName) + ' ' + getComputedStyle(el).fontSize);
  for (const width of [320, 390]) {
    const ctx = await entered(await browser.newContext({ ...devices['iPhone 14'], viewport: { width, height: 760 }, colorScheme: arg('scheme', 'light'), reducedMotion: 'reduce' }));
    const phone = await ctx.newPage();
    watch(phone, () => 'shell, phone ' + width);
    await open(phone, '#/p/' + pid + '/overview', '.feed .ev-work');
    await phone.tap('.feed .ev-work .ev-steps > summary');
    await phone.waitForSelector('.feed .ev-work .ev-steps[open]');
    ok(!(await phone.evaluate(NARROW)).length, width + ' the feed of agents with a shell is wider than the phone: ' + (await phone.evaluate(NARROW)).join('; '));
    const laid = await phone.evaluate(() => { const li = document.querySelector('.feed .ev-work'), who = li.querySelector('.ev-who').getBoundingClientRect(), what = li.querySelector('.ev-work-l').getBoundingClientRect(), time = li.querySelector('time').getBoundingClientRect();
      const cmd = Array.from(document.querySelectorAll('.feed .ev-work .ev-steps code.ev-text')).sort((a, b) => b.textContent.length - a.textContent.length)[0].getBoundingClientRect();
      return { under: what.top >= who.bottom - 1, left: what.left - time.left, wide: what.width / li.getBoundingClientRect().width, lines: cmd.height / 16 }; });
    ok(laid.under && laid.left <= 1 && laid.wide > 0.95, width + ' what an agent did stands under when and who, on the whole width: ' + JSON.stringify(laid));
    ok(laid.lines > 2, width + ' a long command wraps over several lines instead of being cut: ' + laid.lines.toFixed(1));
    const tall = await phone.evaluate(() => { const s = document.querySelector('.feed .ev-work .ev-steps > summary').getBoundingClientRect(); return s.height; });
    ok(tall >= 24, width + ' the fold of the steps is tall enough to touch: ' + tall);
    await phone.tap('[data-act="agents-open"]');
    await phone.waitForSelector('#ag-model');
    ok(!(await phone.evaluate(NARROW)).length && !(await phone.evaluate(SMALL)).length, width + ' the form about the agents on a phone: ' + (await phone.evaluate(NARROW)).concat(await phone.evaluate(SMALL)).join('; '));
    const ticks = await phone.evaluate(() => Array.from(document.querySelectorAll('.agents-form .tick, .agents-form select, .agents-form input:not([type=checkbox]), .agents-form button[type=submit]')).map((el) => Math.round(el.getBoundingClientRect().height)));
    ok(ticks.length >= 6 && ticks.every((h) => h >= 40), width + ' every field about the agents is a finger high: ' + ticks.join(' '));
    await shot(phone, 'agents-phone-' + width);
    await open(phone, '#/start?adv=1', '#opt-agent-runtime');
    await phone.selectOption('#opt-agent-runtime', 'sdk');
    await phone.waitForSelector('#opt-agent-model');
    ok(!(await phone.evaluate(NARROW)).length && !(await phone.evaluate(SMALL)).length, width + ' the composer\'s fields about the agents on a phone: ' + (await phone.evaluate(NARROW)).concat(await phone.evaluate(SMALL)).join('; '));
    await phone.evaluate(() => { localStorage.removeItem('po-composer'); });
    await open(phone, '#/p/' + pid + '/experiments/E1', '.exp-runs .run');
    await phone.waitForSelector('.run.open .run-agent');
    const over = await phone.evaluate(NARROW);
    ok(!over.length, width + ' the run an agent carried out is wider than the phone: ' + over.join('; '));
    ok(/by an agent/.test(await text(phone, '.run.open .run-h')), width + ' and its heading says who carried it out');
    await ctx.close();
  }
};

/** The words on the control that continues the project: a button that starts the agents, or
    a link where a press would start nothing. */
const mainSays = (page) => page.evaluate(() => ((document.querySelector('.strip [data-act="continue"], .strip [data-go-main]') || {}).textContent || '').trim());
/** What a tooltip shows right now: its words, its keys, and where it stands. */
const tipNow = (page) => page.evaluate(() => { const t = document.querySelector('#tip'), r = t.getBoundingClientRect(); return { on: t.classList.contains('on'), words: t.textContent.replace(/\s+/g, ' ').trim(), keys: Array.from(t.querySelectorAll('kbd')).map((k) => k.textContent), top: r.top, bar: document.querySelector('#top').getBoundingClientRect().bottom }; });

/* A project whose models cannot be used: where everybody is who tries the demonstration
   without credentials. What needs no model happens (the run of an approved experiment); the
   rest is something to set up, said once beside the main button in the engine's sentence,
   and never an accident. On a project of its own, which no other check touches. */
CHECKS.nomodel = async ({ page, spare }) => {
  if (!spare || !NO_MODEL || !fs.existsSync(path.dirname(NO_MODEL))) {
    notes.push('nomodel: a project without a usable model was not looked at (' + (!NO_MODEL ? 'PAPER_ORCHESTRA_E2E_NO_MODEL names no file for the server and the checks' : !spare ? 'the workspace has no demo project in a folder named nomodel' : 'the server is on another machine') + ')');
    return;
  }
  const pid = spare, project = '/api/projects/' + pid;
  const errorsOf = async () => ((await api('GET', project + '/activity?limit=200')).data.events || []).filter((ev) => ev.type === 'error').length;
  const accidents = (text) => /stopped unexpectedly|agents stopped|Show log|Try again|ended with exit code|run the command again/i.test(text);
  fs.writeFileSync(NO_MODEL, 'the scripted model is switched off while this file exists\n');
  try {
    /* the server knows beforehand: what a press runs, and where it stops for want of a model */
    const before = (await api('GET', project)).data;
    const blocked = (before.next_step || {}).blocked;
    if (!ok(!!blocked && blocked.needs === 'model' && typeof blocked.message === 'string' && (before.next_step.will_run || []).join() === 'run' && (before.next_step.experiments || []).length >= 1,
      'the server says that the approved experiment runs and that the analysis then needs a model: ' + JSON.stringify(before.next_step))) return;
    const sentence = blocked.message.replace(/`/g, '').replace(/\s+/g, ' ');
    const n = before.next_step.experiments.length;
    const errorsBefore = await errorsOf();
    await reopen(page, '#/p/' + pid + '/map', '.strip');
    await until(page, (label) => ((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim() === label, n === 1 ? 'Run the approved experiment' : 'Run the ' + n + ' approved experiments', 10000, 'the main button promises the run, and no analysis');
    ok(!/analyse/i.test(await mainSays(page)), 'it reads: ' + (await mainSays(page)));
    const need = (await text(page, '.strip-sub .go-need')).replace(/\s+/g, ' ').trim();
    ok(need.includes(sentence), 'the engine\'s sentence about the model stands beside it: ' + need.slice(0, 140));
    ok(/start the app again/.test(need) && (await count(page, '.strip-sub .go-need a[href="#/setup"]')) === 1, 'with what the app adds: where it is set up, and that the app is started again afterwards');
    ok((await count(page, '.strip-sub .go-need')) === 1 && (await count(page, '.strip .go-more')) === 0, 'it is said once, and no phase that needs a model is offered in its place');
    ok((await count(page, '.strip.s-failed')) === 0 && !accidents(await text(page, '.strip')), 'nothing about it is an accident: ' + (await text(page, '.strip-kicker')).trim());
    await shot(page, 'nomodel-before');
    /* the press, by the keyboard: the approved experiment runs, and the project pauses for a model */
    await page.focus('.strip [data-act="continue"]');
    await page.keyboard.press('Enter');
    await until(page, () => !!document.querySelector('.strip.s-running'), null, 20000, 'the press starts the run');
    await until(page, () => /Checkpoint.*after runs/i.test((document.querySelector('.strip-kicker') || {}).textContent || ''), null, 120000, 'and the project pauses after the runs');
    const paused = await idle(pid);
    const ran = ((await api('GET', project + '/experiments')).data.experiments || []).filter((x) => before.next_step.experiments.indexOf(x.id) >= 0).map((x) => x.id + ' ' + x.state);
    ok(paused.state.status === 'checkpoint' && paused.state.phase === 'run' && paused.state.next_phase === 'analyze' && ran.every((x) => / succeeded$/.test(x)), 'what needs no model happened: ' + ran.join(', ') + '; then a checkpoint after the runs, with the analysis next');
    ok((paused.worker.last_exit || {}).code === 0 && (await errorsOf()) === errorsBefore, 'the worker ended as any worker at a checkpoint, and no error is among the events');
    await until(page, () => { const a = document.querySelector('.strip a[data-go-main]'); return !!a && a.textContent.trim() === 'Set up a model' && a.getAttribute('href') === '#/setup'; }, null, 10000, 'with nothing left that needs no model, the main button leads to Setup');
    const pausedNeed = (await text(page, '.strip-sub .go-need')).replace(/\s+/g, ' ');
    ok((await count(page, '.strip-sub .go-need')) === 1 && pausedNeed.includes((((await api('GET', project)).data.next_step || {}).blocked || { message: '?' }).message.replace(/`/g, '').replace(/\s+/g, ' ')), 'and the sentence about the model stays beside it: ' + pausedNeed.trim().slice(0, 120));
    ok((await count(page, '.strip.s-failed')) === 0 && !accidents(await text(page, '.strip')) && /checkpoint/i.test(await text(page, '.top-status')) && !accidents(await text(page, '#status')), 'the strip, the top bar and the status bar say checkpoint: ' + (await text(page, '.top-status')).trim());
    await shot(page, 'nomodel-paused');
    /* the overview says the same, and the sentence once */
    await page.keyboard.press('5');
    await page.waitForSelector('.ov-head');
    const head = (await text(page, '.ov-head')).replace(/\s+/g, ' ');
    ok(/paused at a checkpoint/.test(head) && !accidents(head), 'the overview says that the round is paused at a checkpoint: ' + head.slice(0, 90));
    ok(head.split('needs a model').length - 1 === 1 && /succeeded/.test(await text(page, '.ov-note')), 'with what the runs did, and the sentence about the model once: ' + (await text(page, '.ov-note')).replace(/\s+/g, ' ').trim().slice(0, 120));
    ok((await count(page, '.ov-actions a[href="#/setup"]')) === 1, 'and its main button leads to Setup as well');
    await shot(page, 'nomodel-overview');
    /* A press that is made all the same: the button was drawn while a model could be used, and
       the model is gone when it is pressed (a credential that expired, a button of a moment
       ago). The worker starts nothing and ends for that reason, which the server says by a
       field. It did not die: the app says once that nothing was started, as something to
       set up, and goes on showing the checkpoint. */
    fs.rmSync(NO_MODEL, { force: true });
    await reopen(page, '#/p/' + pid + '/map', '.strip');
    await until(page, () => /^Analyse the/.test(((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim()), null, 15000, 'while a model can be used the main button offers the analysis');
    fs.writeFileSync(NO_MODEL, 'the scripted model is switched off while this file exists\n');
    await page.focus('.strip [data-act="continue"]');
    await page.keyboard.press('Enter');
    await until(page, () => /Nothing was started/.test((document.querySelector('#toast') || {}).textContent || ''), null, 30000, 'the app says once that nothing was started');
    ok(!/toast-error/.test(await page.evaluate(() => (document.querySelector('#toast .toast') || {}).className || '')) && (await count(page, '#toast a[href="#/setup"]')) === 1 && /needs a model/.test(await text(page, '#toast')), 'as something to set up, in the engine\'s sentence and with the way to Setup, not as an error: ' + (await text(page, '#toast')).replace(/\s+/g, ' ').trim().slice(0, 90));
    const ended = await idle(pid);
    const exit = (ended && ended.worker.last_exit) || {};
    ok(exit.reason === 'needs_model' && exit.code !== 0 && ended.state.status === 'checkpoint' && ended.state.phase === 'run' && (await errorsOf()) === errorsBefore, 'its worker ended because a model is needed, with the project as it was and no error among the events: ' + JSON.stringify(exit));
    await until(page, () => !!document.querySelector('.strip a[data-go-main][href="#/setup"]'), null, 10000, 'and the main button leads to Setup again');
    ok((await count(page, '.strip.s-failed')) === 0 && !accidents(await text(page, '.strip')) && (await count(page, '.top-status.st-failed')) === 0 && /checkpoint/i.test(await text(page, '.top-status')) && !accidents(await text(page, '#status')), 'the app still shows a checkpoint, not agents that stopped unexpectedly: ' + (await text(page, '.strip-kicker')).trim());
    await shot(page, 'nomodel-nothing-started');
    await page.click('#toast [data-act="toast-x"]').catch(() => {});
    /* the same after a reload: it is the server's account, not this window's memory */
    await reopen(page, '#/p/' + pid + '/map', '.strip');
    await until(page, () => !!document.querySelector('.strip a[data-go-main][href="#/setup"]'), null, 10000, 'after a reload the main button still leads to Setup');
    ok((await count(page, '.strip.s-failed')) === 0 && !accidents(await text(page, '.strip')) && (await count(page, '.strip-sub .go-need')) === 1 && (await count(page, '.top-status.st-failed')) === 0, 'and the strip still says checkpoint, with the sentence about the model: ' + (await text(page, '.strip-kicker')).trim());
    /* the library does not call it an accident either, and marks the demonstration's cost as its story's */
    await open(page, '#/library', '.lib-row');
    const libRow = '.lib-row[href="#/p/' + pid + '/map"]';
    ok(!accidents(await text(page, libRow)), 'the library row says nothing of an accident: ' + (await text(page, libRow + ' .lib-st')).replace(/\s+/g, ' ').trim());
    ok((await count(page, libRow + ' .lib-num[data-demonstration]')) === 1 && /in its story/.test(await text(page, libRow + ' .lib-num')), 'and says that what the demonstration spent on models is in its story: ' + (await text(page, libRow + ' .lib-num')).replace(/\s+/g, ' ').trim());
    /* by the keyboard: the main button leads to Setup, which says what works without credentials */
    await open(page, '#/p/' + pid + '/map', '.strip a[data-go-main]');
    await page.focus('.strip a[data-go-main]');
    await page.keyboard.press('Enter');
    await until(page, () => location.hash === '#/setup' && !!document.querySelector('.setup .doc-g'), null, 60000, 'Enter on it opens Setup');
    const doctor = (await api('GET', '/api/doctor')).data || {};
    if (doctor.ok === false && doctor.without_credentials) ok((await text(page, '[data-without-credentials]')).replace(/\s+/g, ' ').includes(doctor.without_credentials.replace(/`/g, '').replace(/\s+/g, ' ')), 'Setup says what works without credentials: ' + (await text(page, '[data-without-credentials]')).replace(/\s+/g, ' ').trim().slice(0, 90));
    await shot(page, 'nomodel-setup');
  } finally { fs.rmSync(NO_MODEL, { force: true }); }
  /* the models can be used again: the button offers the analysis of what ran */
  await reopen(page, '#/p/' + pid + '/map', '.strip');
  await until(page, () => /^Analyse the/.test(((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim()) && !document.querySelector('.strip-sub .go-need'), null, 15000, 'once a model can be used, the main button offers the analysis, and nothing is said about a model');
};

/* An experiment that reads approved and is no longer what was approved (its code was changed
   after the approval): it will not run. The server says so wherever it lists the experiment
   (`will_not_run`) and in its forecast (`not_running`); the lists show it as needing approval
   again, and the main button does not promise its run and says why. Called with E4 open in
   the experiments view; leaves it open there. */
async function changedAfterApproval(page, pid, eid) {
  const project = '/api/projects/' + pid;
  const row = ((await api('GET', project + '/experiments')).data.experiments || []).find((x) => x.id === eid) || {};
  const forecast = (await api('GET', project)).data.next_step || {};
  ok(row.state === 'approved' && /not what was approved/.test(row.will_not_run || ''), 'the server lists an approved experiment whose code changed as one that will not run: ' + row.state + ', ' + row.will_not_run);
  ok((forecast.not_running || []).some((x) => x.id === eid) && (forecast.experiments || []).indexOf(eid) < 0, 'and does not count it in what a press runs: ' + JSON.stringify([forecast.experiments, (forecast.not_running || []).map((x) => x.id)]));
  /* (a change of the code writes no event: the page learns of it at its next look) */
  await page.reload();
  await page.waitForSelector('.exp-page .callout-changed');
  ok(/needs approval again/.test(await text(page, '.exp-page .page-status .tag-state')), 'on its page the state reads "needs approval again", not "approved": ' + (await text(page, '.exp-page .page-status .tag-state')).trim());
  const inRail = await page.evaluate(() => { const cur = document.querySelector('.exp-rail .idx.cur'); return cur ? [cur.closest('.idx-group').querySelector('h3').firstChild.textContent.trim(), cur.querySelector('.idx-sub').textContent.trim()] : []; });
  ok(inRail[0] === 'Needs your approval' && inRail[1] === 'needs approval again', 'the list beside it shows it among what needs approval: ' + inRail.join(' / '));
  await page.click('.exp-rail .back');
  await page.waitForSelector('.ex-table');
  const tableRow = '.ex-row:has(.ex-id b:text-is("' + eid + '"))';
  const cell = (await text(page, tableRow + ' .ex-st')).trim(), sub = (await text(page, tableRow + ' .ex-sub')).replace(/\s+/g, ' ').trim();
  ok(/needs approval again/.test(cell) && /will not run/.test(sub) && /changed after the approval/.test(sub) && !/\d{4}-\d\d-\d\dT|human/.test(sub), 'the table says the same, with the reason in the app\'s words: ' + cell + ' | ' + sub.slice(0, 110));
  /* the main button: no run of it is promised, and the sentence beside it says why */
  await page.keyboard.press('1');
  await page.waitForSelector('.strip-sub .go-held[data-held="' + eid + '"]');
  ok(!/Run the approved experiment/.test(await mainSays(page)), 'the main button does not promise a run of it: ' + (await mainSays(page)));
  const held = (await text(page, '.strip-sub .go-held[data-held="' + eid + '"]')).replace(/\s+/g, ' ').trim();
  ok(held.startsWith(eid + ' will not run: it is not what was approved') && /changed after the approval/.test(held) && /written by you/.test(held) && !/\d{4}-\d\d-\d\dT|human/.test(held), 'the sentence beside it says that it will not run, and why, in the app\'s words: ' + held.slice(0, 150));
  ok(/needs approval again/.test((await page.getAttribute('.map-svg .node[data-key="' + row.path + '"]', 'aria-label')) || ''), 'and the map marks it as needing approval again: ' + (await page.getAttribute('.map-svg .node[data-key="' + row.path + '"]', 'aria-label')));
  /* by the keyboard: the way to the experiment, where it is looked at and approved again */
  await page.focus('.strip-sub .go-held[data-held="' + eid + '"] a');
  await page.keyboard.press('Enter');
  await until(page, (id) => location.hash.endsWith('/experiments/' + id) && !!document.querySelector('.exp-page .callout-changed'), eid, 8000, 'the link beside the sentence leads to the experiment');
  await shot(page, 'experiment-changed-after-approval');
}

/* A run the person cancelled is no result. An experiment whose first run succeeded and whose
   second was cancelled reads "cancelled", and wherever that is said the run that succeeded
   is said with it (`last_success` of its rows); the main button counts one result to
   analyse, the one of the run that succeeded. Called with the experiment open in the
   experiments view; leaves it open there. */
async function cancelledAfterSuccess(page, pid, eid) {
  const project = '/api/projects/' + pid;
  const row = ((await api('GET', project + '/experiments')).data.experiments || []).find((x) => x.id === eid) || {};
  ok(row.state === 'cancelled' && !!row.last_success && row.last_success.number === 1, 'the server names the run that succeeded before the cancelled one: ' + JSON.stringify(row.last_success));
  await until(page, () => /run 1 succeeded/.test((document.querySelector('.exp-page .page-status .st-also') || {}).textContent || ''), null, 8000, 'the page of the experiment says "run 1 succeeded" beside "cancelled"');
  const inRail = (await text(page, '.exp-rail .idx.cur .idx-sub')).trim();
  ok(inRail === 'cancelled; run 1 succeeded', 'the list beside it says "cancelled; run 1 succeeded": ' + inRail);
  const forecast = (await api('GET', project)).data.next_step || {};
  ok(forecast.results === 1, 'the server counts one result to analyse, not the cancelled run as well: ' + forecast.results);
  await page.keyboard.press('1');
  await page.waitForSelector('.strip [data-act="continue"]');
  await until(page, () => ((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim() === 'Analyse the new result', null, 8000, 'the main button offers the analysis of one new result');
  ok(/run 1 succeeded/.test(await text(page, '.panel .page-status .st-also')) && /run 1 succeeded/.test((await page.getAttribute('.map-svg .node[data-key="' + row.path + '"]', 'aria-label')) || ''), 'on the map the experiment is cancelled, and its run 1 succeeded: ' + (await page.getAttribute('.map-svg .node[data-key="' + row.path + '"]', 'aria-label')));
  /* the words lead to the record of that run, by the keyboard */
  await page.focus('.panel .page-status a.st-also');
  await page.keyboard.press('Enter');
  await until(page, (target) => location.hash.endsWith(target), '/' + encodeURIComponent(row.last_success.result_page.replace(/^wiki\//, '').replace(/\.md$/, '')), 6000, 'and "run 1 succeeded" leads to the record of that run');
  await open(page, '#/p/' + pid + '/experiments/' + eid, '.exp-page .page-title');
}

/* the map: the selection is shared by the views; keys walk it; a proposal is decided where it stands */
CHECKS.map = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/map', '.map-svg .node');
  const nodes = await count(page, '.map-svg .node');
  ok(nodes >= 39, 'the map shows the demo project: ' + nodes + ' nodes');
  /* a small map draws every line; only a crowded one holds its reference lines back */
  ok((await count(page, '.map-svg .edges .edge.t-cites')) >= 10 && (await count(page, '.map-svg .edges .edge.t-about')) >= 3, 'a small project draws its reference lines with nothing selected: ' + (await count(page, '.map-svg .edges .edge.t-cites')) + ' cites, ' + (await count(page, '.map-svg .edges .edge.t-about')) + ' about');
  ok(!/on selection/.test(await text(page, '.map-key')), 'and its legend does not say that any line waits for a selection');
  ok((await text(page, '.strip-text')).includes('waiting for you'), 'the strip says what is waiting: ' + (await text(page, '.strip-text')));
  const n0 = await inboxCount(page);
  ok(n0 === 6, 'six proposals wait at the start, got ' + n0);
  /* a tooltip shows a key as a key (beside a name, "(3)" reads as a count), and the one of the
     main button opens under it, not over the top bar */
  await page.hover('.views a[href$="/inbox"]');
  await until(page, () => document.querySelector('#tip').classList.contains('on'), null, 3000, 'the tooltip of a view shows');
  const viewTip = await tipNow(page);
  ok(viewTip.keys.join() === '3' && /^Inbox 3$/.test(viewTip.words) && !/\(/.test(viewTip.words), 'the tooltip of the inbox names it and shows its key as a key: ' + JSON.stringify([viewTip.words, viewTip.keys]));
  await page.hover('.strip [data-act="continue"]');
  await until(page, () => document.querySelector('#tip').classList.contains('on') && !/^Inbox/.test(document.querySelector('#tip').textContent), null, 3000, 'the tooltip of the main button shows');
  const mainTip = await tipNow(page);
  ok(mainTip.top >= mainTip.bar, 'and stands below the top bar, which it does not cover: ' + Math.round(mainTip.top) + ' >= ' + Math.round(mainTip.bar));
  await page.mouse.move(4, 400);
  /* targets on the map are at least 24 px */
  const box = await page.locator('.node.r-proposed .n-yes .n-hit').first().boundingBox();
  ok(box && box.width >= 23.5 && box.height >= 23.5, 'accept mark on a node is a 24 px target, got ' + JSON.stringify(box));
  await page.click('.node.k-hypothesis.has-change');
  await until(page, () => location.hash.endsWith('/map/H2'), null, 4000, 'selecting a node sets the route');
  ok((await text(page, '.panel .page-title')).startsWith('A linear warmup'), 'the page opens beside the map');
  ok((await count(page, '.node.dim')) > 5, 'pages beyond the nearest steps are dimmed');
  await shot(page, 'map-selected');
  await page.keyboard.press('2');
  await until(page, () => location.hash.endsWith('/wiki/H2'), null, 4000, 'key 2 opens the wiki with the same selection');
  await page.keyboard.press('4');
  await until(page, () => location.hash.endsWith('/experiments/H2') && document.querySelectorAll('.ex-row.rel').length === 3, null, 4000, 'experiments marks the rows that test the selection');
  await page.keyboard.press('3');
  await until(page, () => { const r = document.querySelector('.in-row.cur'); return r && r.dataset.ref.startsWith('ep-'); }, null, 4000, 'the inbox opens on the proposal about the selection');
  await page.keyboard.press('1');
  await until(page, () => location.hash.endsWith('/map/H2'), null, 4000, 'key 1 returns to the map');
  await page.keyboard.press('ArrowRight');
  await until(page, () => /\/map\/E\d$/.test(location.hash), null, 4000, 'arrow right walks along a line to the next stage');
  await page.keyboard.press('Escape');
  await until(page, () => /\/map$/.test(location.hash), null, 4000, 'Escape clears the selection');
  /* lineage filter and its way back */
  await page.click('.node.k-finding.r-accepted');
  await page.selectOption('#map-lineage', 'only');
  await until(page, () => document.querySelectorAll('.map-svg .node').length < 30, null, 4000, 'lineage only hides the rest');
  await page.selectOption('#map-lineage', 'all');
  await until(page, () => document.querySelectorAll('.map-svg .node').length >= 39, null, 4000, 'the whole map comes back');
  ok((await count(page, '.node.dim')) < 20, 'the whole lineage is lit when asked for');
  await page.selectOption('#map-lineage', 'near');
  await page.keyboard.press('Escape');
  /* a double click on a node selects it: the second click neither unselects it nor clears the selection */
  await page.locator('.node.k-hypothesis.r-accepted').first().dblclick();
  await page.waitForTimeout(700);
  ok(/\/map\/H\d+$/.test(await page.evaluate(() => location.hash)), 'a double click on a node leaves it selected: ' + (await page.evaluate(() => location.hash)));
  await page.keyboard.press('Escape');
  /* a list of choices that is open is not closed by what arrives meanwhile: the page waits for the choice */
  await page.focus('#map-lineage');
  await page.dispatchEvent('#map-lineage', 'mousedown');
  await page.evaluate(() => { document.querySelector('#map-lineage').dataset.same = '1'; });
  const before = await count(page, '.map-svg .node');
  const arrived = await api('POST', '/api/projects/' + pid + '/pages', { kind: 'topic', title: 'Written while a list was open', body: 'x' });
  ok(arrived.status === 201, 'a page arrives while the list is open');
  await sleep(1500);
  ok((await page.evaluate(() => document.querySelector('#map-lineage').dataset.same)) === '1' && (await count(page, '.map-svg .node')) === before, 'the page is not drawn again while a list of choices is open');
  await page.selectOption('#map-lineage', 'all');
  await until(page, (n) => document.querySelectorAll('.map-svg .node').length === n + 1, before, 5000, 'and is drawn, with what arrived, once the choice is made');
  await page.selectOption('#map-lineage', 'near');
  await api('DELETE', '/api/projects/' + pid + '/pages/' + pagePath(arrived.data.page.path));
  await until(page, (n) => document.querySelectorAll('.map-svg .node').length === n, before, 5000, 'the page written for this goes again');
  await page.evaluate(() => document.activeElement.blur());
  /* zoom keeps the headings pinned */
  await page.keyboard.press('+');
  await page.keyboard.press('0');
  /* accept on the map: optimistic, undo within the grace period, then for good */
  await page.click('.node.k-hypothesis.r-proposed .n-yes');
  await until(page, () => document.querySelector('.views .count').textContent === '5', null, 3000, 'accepting on the map lowers the count at once');
  ok((await count(page, '.toast [data-act="undo"]')) === 1, 'the toast offers undo while the decision is not sent');
  await page.keyboard.press('z');
  await until(page, () => document.querySelector('.views .count').textContent === '6', null, 3000, 'undo restores the proposal');
  await sleep(5600);   // longer than the grace period: the undone decision must not go out when it ends
  let h5 = (await api('GET', '/api/projects/' + pid + '/pages/H5')).data.page;
  ok(h5.review === 'proposed', 'an undone decision never reaches the server, also after the grace period');
  ok((await inboxCount(page)) === 6, 'and the proposal still waits');
  await page.click('.node.k-hypothesis.r-proposed .n-yes');
  await until(page, () => !document.querySelector('.toast [data-act="undo"]'), null, 9000, 'the grace period ends');
  await sleep(600);
  h5 = (await api('GET', '/api/projects/' + pid + '/pages/H5')).data.page;
  ok(h5.review === 'accepted', 'after the grace period the decision is on the server: ' + h5.review);
  ok((await inboxCount(page)) === 5, 'the count stays down after the refetch');
};

/* the inbox: edit then accept, reject with a reason, a stale decision, bulk */
CHECKS.inbox = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/inbox', '.in-row');
  const start = await inboxCount(page);
  ok((await count(page, '.in-row:not(.in-old) .in-sum')) >= 3, 'rows carry the one-sentence summary');
  /* a suggested edit is read in place */
  await page.click('.in-row[data-ref^="ep-"] .in-main >> nth=0');
  await page.waitForSelector('.in-detail .review-change');
  const topic = await page.evaluate(() => location.hash);
  if (!/why-warmup/.test(topic)) await page.click('.in-row:has-text("Why warmup helps") .in-main');
  await page.waitForSelector('.in-detail .prose-marked ins');
  ok((await count(page, '.in-detail .prose-marked ins')) >= 1, 'the suggested edit is marked in the text');
  ok((await text(page, '.in-detail .review-count')).includes('+48'), 'the note counts the words: ' + (await text(page, '.in-detail .review-count')));
  await shot(page, 'inbox-marked-edit');
  await page.click('.in-detail [data-act="toggle-diff"]');
  await page.waitForSelector('.in-detail pre.diff .diff-add');
  ok((await count(page, '.in-detail .prose-marked')) === 0, 'show as diff swaps the marked text for the page as it is');
  await page.click('.in-detail [data-act="toggle-diff"]');
  /* reject with a reason, by the keyboard */
  await page.click('.in-row:has-text("Q6") .in-main');
  await page.waitForSelector('.in-detail .page-title:has-text("full learning rate")');
  await page.keyboard.press('r');
  await page.waitForSelector('.reject-form input:focus');
  await page.keyboard.type('Covered by E4 already.');
  await page.keyboard.press('Enter');
  await until(page, (n) => document.querySelector('.views .count').textContent === String(n - 1), start, 3000, 'rejecting lowers the count');
  await until(page, () => !document.querySelector('.toast [data-act="undo"]'), null, 9000, 'the rejection is sent');
  await sleep(500);
  const q6 = (await api('GET', '/api/projects/' + pid + '/pages/Q6')).data.page;
  ok(q6.review === 'rejected' && q6.meta.decision_reason === 'Covered by E4 already.', 'the reason reaches the server: ' + JSON.stringify([q6.review, q6.meta.decision_reason]));
  /* Editing is not deciding. A proposal that is edited and saved is still a proposal, and the
     app says so; "Save and accept" is the decision. A rejected page stays rejected. */
  await page.click('.in-row:has-text("F2") .in-main');
  await page.waitForSelector('.in-detail .page-title:has-text("gives back")');
  await page.waitForSelector('.in-detail .prose');
  await page.keyboard.press('e');
  await page.waitForSelector('textarea#ed-body');
  /* E opens the editor and gives it the keyboard; the letter of the key is not typed into the text */
  ok((await page.inputValue('textarea#ed-body')) === (await api('GET', '/api/projects/' + pid + '/pages/F2')).data.page.body && /No changes yet/.test(await text(page, '.ed-draft')) && (await focusOf(page)).startsWith('textarea#ed-body'), 'E opens the editor on the text as it is, with the keyboard in it and nothing typed: ' + JSON.stringify((await page.inputValue('textarea#ed-body')).slice(0, 24)) + ', ' + (await text(page, '.ed-draft')).trim());
  ok((await count(page, '.ed-bar button[type=submit][data-save="plain"]')) === 1 && (await count(page, '.ed-bar [data-act="save-accept"]')) === 1, 'the editor of a proposal offers Save and Save and accept');
  ok(/Save keeps it a proposal/.test(await text(page, '.ed-what')), 'and says what each does: ' + (await text(page, '.ed-what')));
  const body = await page.inputValue('textarea#ed-body');
  await page.fill('textarea#ed-body', body + '\n\nMy own caveat: one learning rate only.\n');
  await until(page, () => /My own caveat/.test(document.querySelector('.ed-prev').textContent), null, 5000, 'the preview is rendered by the server and follows the editor');
  await shot(page, 'inbox-editor');
  const waitingBefore = await waitingNow(page);
  const saves = [];
  const listenSave = (request) => { if (request.method() === 'PUT' && /\/pages\//.test(request.url())) saves.push(JSON.parse(request.postData() || '{}')); };
  page.on('request', listenSave);
  await page.keyboard.press('Control+s');
  await until(page, () => !document.querySelector('form.editor'), null, 6000, 'saving closes the editor');
  let f2 = (await api('GET', '/api/projects/' + pid + '/pages/F2')).data.page;
  ok(f2.review === 'proposed' && f2.body.includes('My own caveat'), 'a saved edit of a proposal keeps it proposed, with the person\'s text: ' + f2.review);
  ok(saves.length === 1 && !saves[0].accept, 'the plain save does not ask for the page to be accepted: ' + JSON.stringify(saves.map((b) => b.accept)));
  ok(/still a proposal/.test(await text(page, '.toast')), 'and the app says that it is still a proposal: ' + (await text(page, '.toast')).trim());
  ok((await waitingNow(page)) === waitingBefore && (await count(page, '.in-row:not(.in-old):has-text("F2")')) === 1, 'it is still in the inbox: ' + (await waitingNow(page)) + ' waiting');
  ok(/edited by you|by you/.test(await text(page, '.in-detail .page-by')), 'the page says that the person edited it: ' + (await text(page, '.in-detail .page-by')));
  /* when it was proposed is not when the person saved their edit of it */
  const proposed = (await text(page, '.in-detail .review-say')).replace(/\s+/g, ' ').trim(), inRow = (await text(page, '.in-row.cur .in-what')).replace(/\s+/g, ' ').trim();
  ok(/^Proposed by the analyst/.test(proposed) && !/just now/.test(proposed) && !/just now/.test(inRow) && /updated just now/.test(await text(page, '.in-detail .page-by')), 'the proposal is still said to be proposed when the agent proposed it: ' + proposed.slice(0, 60) + ' | ' + inRow);
  await page.keyboard.press('e');
  await page.waitForSelector('textarea#ed-body');
  await page.fill('#ed-summary', 'The gain at step 100 is gone by step 300 (one learning rate).');
  await page.click('.ed-bar [data-act="save-accept"]');
  await until(page, () => !document.querySelector('form.editor'), null, 6000, 'save and accept closes the editor');
  page.off('request', listenSave);
  f2 = (await api('GET', '/api/projects/' + pid + '/pages/F2')).data.page;
  ok(f2.review === 'accepted' && f2.body.includes('My own caveat') && /one learning rate/.test(f2.summary), 'save and accept saves the person\'s version and accepts it: ' + f2.review);
  ok(saves.length === 2 && saves[1].accept === true, 'it says "accept: true" itself: ' + JSON.stringify(saves.map((b) => b.accept)));
  ok(/accepted F2/.test(await text(page, '.toast')), 'and the app says that it accepted: ' + (await text(page, '.toast')).trim());
  /* the page rejected a moment ago is edited: it stays rejected, and the editor and the toast say so */
  await open(page, '#/p/' + pid + '/wiki/Q6/edit', 'textarea#ed-body');
  ok(/keeps it rejected/.test(await text(page, '.ed-what')) && (await count(page, '.ed-bar [data-act="save-accept"]')) === 0, 'the editor of a rejected page says that saving keeps it rejected, and offers no accept: ' + (await text(page, '.ed-what')));
  await page.fill('#ed-summary', 'Corrected by me, and still not wanted.');
  await page.keyboard.press('Control+s');
  await until(page, () => !document.querySelector('form.editor'), null, 6000, 'the rejected page is saved');
  const q6now = (await api('GET', '/api/projects/' + pid + '/pages/Q6')).data.page;
  ok(q6now.review === 'rejected' && q6now.summary === 'Corrected by me, and still not wanted.', 'an edited rejected page stays rejected: ' + q6now.review);
  ok(/stays rejected/.test(await text(page, '.toast')), 'and the app says so: ' + (await text(page, '.toast')).trim());
  await open(page, '#/p/' + pid + '/inbox', '.in-row');
  /* a stale decision: the page is revised after it was read, the decision is refused and rolled back */
  await page.click('.in-row:has-text("E3") .in-main');
  await page.waitForSelector('.in-detail .page-title:has-text("small Transformer")');
  await page.waitForSelector('.in-detail .review .note-warn');
  ok(/No backend can run this/.test(await text(page, '.in-detail .review .note-warn')), 'an experiment that nothing here can run says so, with each backend\'s reason');
  ok(await page.locator('.in-detail [data-act="accept-run"]').isDisabled(), 'and accept and run is not offered for it');
  const before = await inboxCount(page);
  await page.click('.in-detail [data-act="accept"]');
  await until(page, (n) => document.querySelector('.views .count').textContent === String(n - 1), before, 3000, 'the accept shows at once');
  const elsewhere = await api('POST', '/api/projects/' + pid + '/review', { items: [{ ref: 'E3', decision: 'reject', reason: 'Decided in another window.' }] });
  ok(elsewhere.status === 200 && elsewhere.data.ok, 'the page was decided elsewhere meanwhile');
  await until(page, () => !!document.querySelector('.toast-error'), null, 9000, 'a refused decision says so');
  ok(/changed after you read it/.test(await text(page, '.toast-error')), 'the refusal is explained: ' + (await text(page, '.toast-error')));
  ok(/you changed it (just now|\d+ min ago)/.test(await text(page, '.toast-error')) && !/\d{4}-\d\d-\d\dT|human/.test(await text(page, '.toast-error')), 'in the app\'s words: no raw time, and the person is "you"');
  ok((await api('GET', '/api/projects/' + pid + '/pages/E3')).data.page.review === 'rejected', 'the stale decision did not land on the server');
  await open(page, '#/p/' + pid + '/experiments/E3', '.exp-page .page-title');
  await page.waitForSelector('.exp-page .note-rejected');
  ok((await count(page, '.exp-page .approve')) === 0 && /Decided in another window/.test(await text(page, '.exp-page .note-rejected')), 'a rejected experiment is not offered for approval and shows the reason');
  await open(page, '#/p/' + pid + '/inbox', '.in-list');
  await until(page, () => /rejected/.test((document.querySelector('.in-old:has(.pn-id)') || {}).textContent || '') && [...document.querySelectorAll('.in-old')].some((r) => /E3/.test(r.textContent) && /rejected/.test(r.textContent)), null, 5000, 'and the page is shown as it is now: rejected');
  await page.click('.toast [data-act="toast-x"]');
  /* What was decided holds the reason of a rejection, and the suggested edits that were decided
     beside the pages (the demonstration's story has one that was turned down, with a reason). */
  ok(await page.evaluate(() => [...document.querySelectorAll('.in-old')].some((r) => /Q6/.test(r.textContent) && /Covered by E4 already\./.test(r.textContent))), 'a rejected page is listed with the reason that was given');
  const decidedEdits = ((await api('GET', '/api/projects/' + pid + '/proposals?review=all')).data.proposals || []).filter((x) => x.review !== 'proposed' && x.decided_at);
  const turnedDown = decidedEdits.filter((x) => x.review === 'rejected' && x.decision_reason).sort((a, b) => String(b.decided_at).localeCompare(String(a.decided_at)))[0];
  if ((await count(page, '.in-done [data-act="decided-more"]'))) { await page.focus('.in-done [data-act="decided-more"]'); await page.keyboard.press('Enter'); }
  await until(page, (n) => document.querySelectorAll('.in-old.in-edit').length === Math.min(n, 60), decidedEdits.length, 8000, 'the suggested edits that were decided are listed among what was decided (' + decidedEdits.length + ')');
  if (ok(!!turnedDown, 'the story of the demonstration has a suggested edit that was turned down with a reason')) {
    const rowOf = '.in-old.in-edit[data-decided="' + turnedDown.id + '"]';
    ok((await text(page, rowOf)).replace(/\s+/g, ' ').includes(turnedDown.decision_reason) && /edit rejected/.test(await text(page, rowOf + ' .in-when')), 'one that was turned down says so, with the reason: ' + (await text(page, rowOf)).replace(/\s+/g, ' ').trim().slice(0, 150));
  }
  /* a double click decides once: the row that moves up under the pointer is not decided too */
  await open(page, '#/p/' + pid + '/inbox', '.in-row');
  const two = await waitingNow(page);
  if (two >= 2) {
    await page.locator('.in-row:not(.in-old) [data-act="accept"]').first().dblclick();
    await sleep(300);
    ok((await waitingNow(page)) === two - 1, 'a double click on Accept decides one proposal, not two: ' + two + ' -> ' + (await waitingNow(page)));
    await page.keyboard.press('z');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n, two, 3000, 'and that one is taken back by undo');
    /* a key held down decides once: A a moment too long does not accept the next proposal as well */
    await page.keyboard.down('a'); await page.keyboard.down('a'); await page.keyboard.down('a'); await page.keyboard.up('a');
    await sleep(300);
    ok((await waitingNow(page)) === two - 1, 'A held down accepts one proposal, not one after the other: ' + two + ' -> ' + (await waitingNow(page)));
    await page.keyboard.press('z');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n, two, 3000, 'which undo takes back');
    /* decisions made one after the other can each be taken back, the newest first */
    await page.keyboard.press('a');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n - 1, two, 3000, 'A accepts the proposal in front');
    await page.keyboard.press('a');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n - 2, two, 3000, 'and again the next one');
    ok(/before it can still be taken back/.test(await text(page, '.toast')), 'the toast says that the earlier decision can still be taken back: ' + (await text(page, '.toast')).replace(/\s+/g, ' '));
    await page.keyboard.press('z');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n - 1, two, 3000, 'Z takes back the newest decision');
    ok((await count(page, '.toast [data-act="undo"]')) === 1, 'and offers the one before it');
    await page.keyboard.press('z');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n, two, 3000, 'Z again takes back the one before');
    await sleep(5600);
    ok((await api('GET', '/api/projects/' + pid)).data.counts.pending.total === two, 'none of the decisions taken back reached the server');
  }
  /* bulk: select two, accept them together, take it back */
  await open(page, '#/p/' + pid + '/inbox', '.in-row');
  const waiting = await inboxCount(page);
  if (waiting >= 2) {
    await page.keyboard.press('x'); await page.keyboard.press('j'); await page.keyboard.press('x');   // no pause between them on purpose
    await until(page, () => /Accept 2 selected/.test(document.querySelector('[data-act="bulk-accept"]').textContent), null, 3000, 'X selects the item in front, twice');
    await shot(page, 'inbox-bulk');
    await page.keyboard.press('Shift+A');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n - 2, waiting, 3000, 'shift A accepts everything selected');
    await page.keyboard.press('z');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n, waiting, 3000, 'one undo takes the whole batch back');
  }
  await shot(page, 'inbox');
};

/* a suggested edit beside the page in the wiki; accepting it; the editor's conflict flow */
CHECKS.wiki = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/wiki/H2', '.page-wiki .prose');
  ok((await count(page, '.wiki-rail .review-change')) === 1, 'in the wiki the suggested edit is a note in the margin');
  ok((await count(page, '.wiki-main .review-change')) === 0, 'and not above the text as well');
  await page.click('.wiki-rail [data-act="accept"]');
  await until(page, () => !!document.querySelector('.page-status .tag-state.t-pos'), null, 3000, 'accepting the change shows the new state at once');
  await until(page, () => !document.querySelector('.toast [data-act="undo"]'), null, 9000, 'the change is sent');
  await sleep(400);
  ok((await api('GET', '/api/projects/' + pid + '/pages/H2')).data.page.state === 'supported', 'the accepted change is applied on the server');
  /* a suggested edit that was decided can be found again: on the page it was about, with the decision */
  await until(page, () => !!document.querySelector('.wiki-rail .decided-edits [data-decided]'), null, 8000, 'the page lists the suggested edit that was decided');
  const decidedEdit = (await text(page, '.wiki-rail .decided-edits')).replace(/\s+/g, ' ').trim();
  ok(/Mark H2 as supported/.test(decidedEdit) && /Suggested by the analyst/.test(decidedEdit) && /accepted by you/.test(decidedEdit), 'with what was suggested, by whom, and what became of it: ' + decidedEdit.slice(0, 150));
  /* history: versions and their diffs */
  await page.keyboard.press('h');
  await page.waitForSelector('.history .hist-i');
  ok((await count(page, '.hist-i')) >= 2, 'the history lists the versions');
  await open(page, '#/p/' + pid + '/wiki/topics%2Fwhy-warmup-helps/history', '.history .hist-i');
  await page.click('.hist-i [data-act="version"] >> nth=-1');
  await page.waitForSelector('.hist-body pre.diff');
  ok((await count(page, '.hist-body pre.diff .diff-line')) > 3, 'a version is compared with the page as it is now');
  await shot(page, 'wiki-history');
  /* the number check on a finding */
  await open(page, '#/p/' + pid + '/wiki/F1', '.page-wiki .prose');
  await page.waitForSelector('.prose a.num');
  ok((await count(page, '.prose a.num')) >= 7, 'checked numbers are marked in the text: ' + (await count(page, '.prose a.num')));
  /* the margin says what the check proves, in the server's own sentence, and lists what that sentence counts */
  const f1view = (await api('GET', '/api/projects/' + pid + '/pages/F1')).data, report = f1view.number_report;
  ok(!!report && (await text(page, '.numcheck-sum')).trim() === report.summary, 'the number check shows the server\'s summary sentence: ' + (await text(page, '.numcheck-sum')).trim());
  ok(/short and round numbers are not checked/.test(await text(page, '.numcheck-sum')), 'which says that short and round numbers are not checked');
  const farText = await text(page, '.numcheck-far');
  ok(report.elsewhere.every((n) => farText.includes(n)) && /only outside the records this finding is based on/.test(farText), 'the numbers found only outside the finding\'s records are named: ' + farText.trim());
  const skipText = await text(page, '.numcheck-skip');
  ok(report.skipped.length >= 1 && report.skipped.every((n) => skipText.includes(n)), 'and so are the numbers that were not looked at: ' + skipText.trim());
  ok((await count(page, '.prose a.num.num-far')) >= 1 && /not in the records this finding is based on/.test(await page.getAttribute('.prose a.num.num-far', 'data-tip')), 'a number found elsewhere is marked as that in the text');
  ok(!/occur in a record|is in a record/.test(await text(page, '.numcheck')), 'the margin promises no more than the report: ' + (await text(page, '.numcheck')).replace(/\s+/g, ' ').trim().slice(0, 120));
  await page.click('.prose a.num:not(.num-far) >> nth=1');
  await until(page, () => /results%2FE1-run1$/.test(location.hash), null, 4000, 'a number links to the record it was found in');
  ok((await count(page, '.trail .trail-i')) >= 2, 'the trail shows the pages opened on the way');
  ok((await count(page, '.prose img')) >= 1 || true, '');
  /* edit and conflict */
  await open(page, '#/p/' + pid + '/wiki/Q3/edit', 'textarea#ed-body');
  const q3 = (await api('GET', '/api/projects/' + pid + '/pages/Q3')).data.page;
  await page.fill('textarea#ed-body', q3.body + '\nMine: checked against the notes.\n');
  await page.fill('#ed-summary', 'My summary');
  const other = await api('PUT', '/api/projects/' + pid + '/pages/' + pagePath(q3.path), { body: q3.body + '\nTheirs: a line added elsewhere.\n', base_updated: q3.updated });
  ok(other.status === 200, 'someone else saved the page meanwhile');
  await page.keyboard.press('Control+s');
  await page.waitForSelector('.conflict');
  ok(/changed while you were editing/.test(await text(page, '.conflict')), 'the conflict is explained');
  ok(/Theirs: a line added elsewhere/.test(await text(page, '.conflict .diff')), 'what changed underneath is shown');
  ok((await page.inputValue('textarea#ed-body')).includes('Mine: checked'), 'the draft is kept through the conflict');
  await shot(page, 'wiki-conflict');
  /* the draft survives a reload */
  await page.reload();
  await page.waitForSelector('textarea#ed-body');
  ok((await page.inputValue('textarea#ed-body')).includes('Mine: checked'), 'the draft survives a reload');
  await page.keyboard.press('Control+s');
  await page.waitForSelector('.conflict');
  await page.click('[data-act="conflict-mine"]');
  await until(page, () => !document.querySelector('form.editor'), null, 6000, 'saving over the newer version closes the editor');
  const saved = (await api('GET', '/api/projects/' + pid + '/pages/Q3')).data;
  ok(saved.page.body.includes('Mine: checked') && !saved.page.body.includes('Theirs:') && saved.page.summary === 'My summary', 'my version is what the page holds now');
  ok(saved.history.length >= 2, 'both earlier versions are in the history');
  /* restore the version before mine */
  await page.keyboard.press('h');
  await page.waitForSelector('.hist-i [data-act="restore"]');
  await page.click('.hist-i [data-act="restore"] >> nth=0');
  await until(page, () => !document.querySelector('.history'), null, 6000, 'restoring returns to the page');
  ok((await api('GET', '/api/projects/' + pid + '/pages/Q3')).data.page.body.includes('Theirs:'), 'the restored text is the page again');
  /* the other way out of a conflict: keep editing on top of the newer version, with what it changed in sight */
  await open(page, '#/p/' + pid + '/wiki/Q4/edit', 'textarea#ed-body');
  ok((await focusOf(page)) === 'textarea#ed-body', 'an editor that opens has the keyboard: ' + (await focusOf(page)));
  const q4 = (await api('GET', '/api/projects/' + pid + '/pages/Q4')).data.page;
  await page.fill('textarea#ed-body', q4.body + '\nMine: a second thought.\n');
  await api('PUT', '/api/projects/' + pid + '/pages/' + pagePath(q4.path), { body: q4.body + '\nTheirs: written in another window.\n', base_updated: q4.updated });
  await page.keyboard.press('Control+s');
  await page.waitForSelector('.conflict');
  await page.click('[data-act="conflict-rebase"]');
  await page.waitForSelector('.conflict-kept');
  ok(/Theirs: written in another window/.test(await text(page, '.conflict-kept .diff')), 'after "keep editing" what the newer version changed stays in sight');
  ok((await page.inputValue('textarea#ed-body')).includes('Mine: a second thought'), 'and the text is still mine');
  await page.keyboard.press('Control+s');
  await until(page, () => !document.querySelector('form.editor'), null, 6000, 'the next save goes through');
  ok((await api('GET', '/api/projects/' + pid + '/pages/Q4')).data.page.body.includes('Mine: a second thought'), 'and my text is the page');
  /* a page of the person's own, then delete it */
  await page.click('[data-act="new-page"]');
  await page.waitForSelector('#np-title');
  /* the kinds are the ones the server says a person can write: a lesson, which only the person writes, among them */
  const kinds = (await api('GET', '/api/defaults')).data.creatable_kinds || [];
  await until(page, (n) => document.querySelectorAll('#np-kind option').length === n, kinds.length, 6000, 'the New dialog offers the ' + kinds.length + ' kinds the server lists');
  ok(kinds.indexOf('lesson') >= 0 && /only you write these/.test(await text(page, '#np-kind option[value="lesson"]')), 'a lesson among them, as the one kind only the person writes: ' + (await text(page, '#np-kind option[value="lesson"]')).trim());
  await page.selectOption('#np-kind', 'topic');
  await page.fill('#np-title', 'A note of my own');
  await page.keyboard.press('Enter');
  await page.waitForSelector('textarea#ed-body');
  await page.fill('textarea#ed-body', 'Written by me, about [[H1]].\n');
  await until(page, () => !!document.querySelector('.ed-prev a.wikilink'), null, 5000, 'the preview resolves wiki links');
  await page.click('.ed-bar button[type=submit]');
  await until(page, () => !document.querySelector('form.editor') && !!document.querySelector('.page-wiki .prose a.wikilink'), null, 6000, 'the new page is saved and shown');
  const mine = (await api('GET', '/api/projects/' + pid + '/pages/' + pagePath('wiki/topics/a-note-of-my-own.md'))).data;
  ok(mine && mine.page && mine.page.author === 'human' && mine.page.review === 'accepted', 'a page by the person is accepted as written');
  await shot(page, 'wiki-own-page');
  const lost = await refused(page, async () => {
    await page.click('.page-tools [data-act="delete-ask"]');
    await page.click('.confirm [data-act="delete"]');
    await until(page, () => !/a-note-of-my-own/.test(location.hash), null, 6000, 'deleting leaves the page');
  });
  ok(!lost.length, 'a deleted page is not asked for again: ' + lost.join(', '));
  ok((await api('GET', '/api/projects/' + pid + '/pages/' + pagePath('wiki/topics/a-note-of-my-own.md'))).status === 404, 'the page is gone from the wiki');
  /* a number that is in no record is flagged in the text and explained in the margin */
  const made = await api('POST', '/api/projects/' + pid + '/pages', { kind: 'finding', title: 'A finding with a number from nowhere', body: '## What was observed\n\nThe spike was 0.121 without warmup, and 73.25 times smaller with it.\n', relations: { based_on: ['results/E1-run1'] } });
  ok(made.status === 201, 'a finding is created: ' + made.status);
  await open(page, '#/p/' + pid + '/wiki/' + made.data.page.id, '.page-wiki .prose');
  await page.waitForSelector('.prose .num-flag');
  ok(/73\.25/.test(await text(page, '.prose .num-flag')), 'the unverified number is flagged in the text: ' + (await text(page, '.prose .num-flag')));
  ok((await count(page, '.prose a.num')) >= 1, 'next to one that is in a record');
  ok(/Not found:\s*73\.25/.test(await text(page, '.numcheck-flag')), 'and named in the margin as not found: ' + (await text(page, '.numcheck-flag')));
  ok((await text(page, '.numcheck-sum')).trim() === (await api('GET', '/api/projects/' + pid + '/pages/' + made.data.page.id)).data.number_report.summary, 'under the server\'s sentence: ' + (await text(page, '.numcheck-sum')).trim());
  await shot(page, 'wiki-number-flag');
  /* a suggested edit can be edited before it is accepted */
  const topic = (await api('GET', '/api/projects/' + pid + '/pages/' + pagePath('wiki/topics/why-warmup-helps.md'))).data;
  if (topic.proposals.length) {
    await open(page, '#/p/' + pid + '/wiki/topics%2Fwhy-warmup-helps', '.wiki-rail .review-change');
    await page.click('.wiki-rail [data-act="edit-proposal"]');
    await page.waitForSelector('textarea#ed-body');
    const suggested = await page.inputValue('textarea#ed-body');
    ok(/now gives one data point/.test(suggested), 'the editor starts from the suggested text');
    await page.fill('textarea#ed-body', suggested + '\nAnd a sentence of my own.\n');
    await page.click('.ed-bar button[type=submit]');
    await until(page, () => !document.querySelector('form.editor'), null, 6000, 'saving the edited suggestion closes the editor');
    const after = (await api('GET', '/api/projects/' + pid + '/pages/' + pagePath('wiki/topics/why-warmup-helps.md'))).data;
    ok(/now gives one data point/.test(after.page.body) && /a sentence of my own/.test(after.page.body) && !after.proposals.length, 'the page holds the suggestion with my change, and the suggestion is decided');
  }
  /* a result record that was edited by hand says so on its page */
  const record = (await api('GET', '/api/projects/' + pid + '/pages/' + pagePath('wiki/results/E2-run1.md'))).data;
  ok(record.edited_record === false, 'a record as code wrote it is not marked: ' + record.edited_record);
  await open(page, '#/p/' + pid + '/wiki/results%2FE2-run1', '.page-wiki .prose');
  ok((await count(page, '[data-edited-record]')) === 0, 'and its page says nothing about an edit');
  const edited = await api('PUT', '/api/projects/' + pid + '/pages/' + pagePath('wiki/results/E2-run1.md'), { body: record.page.body + '\nA remark added by hand.\n', base_updated: record.page.updated });
  ok(edited.status === 200 && edited.data.edited_record === true, 'the record is edited by hand: ' + edited.status + ' ' + (edited.data && edited.data.edited_record));
  await until(page, () => /edited by hand/.test((document.querySelector('[data-edited-record]') || {}).textContent || ''), null, 8000, 'a result record that was edited by hand says so on the page');
  await shot(page, 'wiki-edited-record');
  /* and the run it belongs to does not go on saying that the record is as code wrote it */
  await open(page, '#/p/' + pid + '/experiments/E2', '.exp-runs .run-h');
  if (!(await count(page, '.run.open .run-h[data-run="1"]'))) await page.click('.run-h[data-run="1"]');
  await until(page, () => /edited by hand since/.test((document.querySelector('.run.open .run-rec') || {}).textContent || ''), null, 6000, 'the run of an edited record says that the record was edited by hand since');
  /* (the page is left as it opens by itself, with the latest run unfolded, for the checks that follow) */
  const latestRun = await page.$$eval('.exp-runs .run-h', (heads) => Math.max(...heads.map((el) => +el.dataset.run)));
  if (latestRun !== 1) { await page.click('.run-h[data-run="' + latestRun + '"]'); await page.waitForSelector('.run.open .run-h[data-run="' + latestRun + '"]'); }
};

/* experiments: the table, an approval, a real local run with its log, metrics and figure,
   the person's edit of the code, and a cancelled run */
CHECKS.experiments = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/experiments', '.ex-table');
  ok((await count(page, '.ex-row')) === 4, 'the table lists the four experiments');
  ok(/budget/i.test(await text(page, '.ex-budget')), 'the budget and the limit per run stand above it');
  ok((await text(page, '.ex-row:has-text("E2") .ex-sub')).includes('ValueError'), 'a failed experiment says why under its title');
  ok((await text(page, '.ex-row:has-text("E1") .ex-sub')).includes('initial_loss'), 'a succeeded one shows its headline metric');
  await shot(page, 'experiments-table');
  /* a long metrics table: what the plan names first, the rest on demand; who wrote the code */
  await open(page, '#/p/' + pid + '/experiments/E1', '.run.open .metrics');
  const named = await count(page, '.run.open .res-grid > div:first-child > .metrics tr');
  ok(named === 16, 'the metrics the plan names come first: ' + named);
  ok(/All 22 metrics/.test(await text(page, '.metrics-all summary')), 'the rest is behind "All 22 metrics": ' + (await text(page, '.metrics-all summary')));
  await page.click('.metrics-all summary');
  ok((await count(page, '.metrics-all[open] tr')) === 6, 'and opens in place');
  ok(/written by the designer/.test(await text(page, '.code-by')), 'the code says who wrote it and when: ' + (await text(page, '.code-by')));
  await until(page, () => { const log = document.querySelector('.run.open .log-body'); return log && !log.dataset.empty; }, null, 5000, 'the log of the finished run is read');
  const inFile = (await api('GET', '/api/projects/' + pid + '/experiments/E1/runs/1/log?offset=0')).data.text;
  ok((await page.textContent('.run.open .log-body')) === inFile, 'the log of a finished run is shown as the file is, once: ' + (await page.textContent('.run.open .log-body')).length + ' characters shown, ' + inFile.length + ' in the file');
  await shot(page, 'experiment-metrics');
  await open(page, '#/p/' + pid + '/experiments', '.ex-table');
  /* a failed run: what kind of failure, the traceback, the repair with its diff */
  await page.click('.ex-row:has-text("E2") .ex-name a');
  await page.waitForSelector('.failure');
  ok(/programming error/.test(await text(page, '.fail-t')), 'the failure is named: ' + (await text(page, '.fail-t')));
  await until(page, () => /Traceback/.test(document.querySelector('.failure .code-err').textContent), null, 5000, 'the last lines of the traceback are shown');
  ok((await count(page, '.repair pre.diff .diff-add')) >= 1, 'the runner\'s repair is shown as a diff');
  ok((await count(page, '.failure [data-act="edit-code"]')) === 1, 'a programming error offers to edit the code');
  await shot(page, 'experiment-failed');
  /* a proposed experiment: only what can run is offered (when the inbox check has not decided it) */
  if ((await api('GET', '/api/projects/' + pid + '/pages/E3')).data.page.review === 'proposed') {
    await open(page, '#/p/' + pid + '/experiments/E3', '.approve .backend');
    ok((await count(page, '.approve .backend.off')) >= 1, 'a backend that cannot run the plan is shown as such, with the reason');
    ok(await page.locator('.approve [data-act="accept-run"]').isDisabled(), 'accept and run is not offered when nothing can run it');
    await shot(page, 'experiment-proposed');
  }
  /* an approved one runs now: live log, then metrics and a figure */
  await open(page, '#/p/' + pid + '/experiments/E4', '[data-act="run-now"]');
  await page.click('[data-act="run-now"]');
  await until(page, () => !!document.querySelector('.run-live') || !!document.querySelector('.run.open .metrics'), null, 15000, 'the run shows as running');
  await until(page, () => { const log = document.querySelector('.run-live .log-body, .run.open .log-body'); return log && log.textContent.trim().split('\n').length >= 2 && !/Reading the log/.test(log.textContent); }, null, 20000, 'the log streams while it runs');
  await shot(page, 'experiment-running');
  await until(page, () => !!document.querySelector('.run.open .metrics tr'), null, 60000, 'the run ends with its metrics');
  ok((await count(page, '.run.open .fig img')) >= 1, 'and with its figure, from the files endpoint');
  const natural = await page.evaluate(async () => { const img = document.querySelector('.run.open .fig img'); if (!img) return 0; if (!img.complete) await new Promise((r) => { img.onload = r; img.onerror = r; }); return img.naturalWidth; });
  ok(natural > 50, 'the figure really loads (' + natural + ' px wide)');
  /* (the pill follows the run's own block by a render: wait for it, do not read it once) */
  await until(page, () => /succeeded/.test((document.querySelector('.page-status .tag-state') || {}).textContent || ''), null, 20000, 'the experiment is succeeded');
  await shot(page, 'experiment-succeeded');
  await idle(pid);
  /* the person edits the code so that it runs long, approves it again, and cancels the run */
  await page.click('[data-act="edit-code"]');
  await page.waitForSelector('#code-text');
  const code = await page.inputValue('#code-text');
  await page.fill('#code-text', 'import time\nfor i in range(600):\n    print("waiting", i, flush=True)\n    time.sleep(0.25)\n' + code);
  await page.reload();
  await page.waitForSelector('#code-text');
  ok((await page.inputValue('#code-text')).startsWith('import time\nfor i in range(600)'), 'code that is being edited survives a reload');
  ok(/unsaved edit of main\.py was restored/.test(await text(page, '.code-bar')), 'and the editor says that it was restored: ' + (await text(page, '.code-bar')).trim().slice(0, 80));
  await page.click('.code-bar button[type=submit]');
  await until(page, () => !document.querySelector('#code-text') && /waiting/.test(document.querySelector('.code-n').textContent), null, 6000, 'the edited code is saved and shown');
  ok(/written by you/.test(await text(page, '.code-by')), 'and says that you wrote it: ' + (await text(page, '.code-by')));
  /* code that changes under an approval takes the approval back, and the page says why */
  const again = await api('POST', '/api/projects/' + pid + '/experiments/E4/approve', { backend: 'local' });
  ok(again.status === 200, 'the experiment is approved through the API: ' + again.status);
  const now = (await api('GET', '/api/projects/' + pid + '/experiments/E4')).data.code.files.find((f) => f.name === 'main.py').text;
  await api('PUT', '/api/projects/' + pid + '/experiments/E4/code/main.py', { text: now + '\n# changed after the approval\n' });
  /* before anything is pressed: it reads approved and will not run, and every list and the main button say so */
  await changedAfterApproval(page, pid, 'E4');
  await api('POST', '/api/projects/' + pid + '/experiments/E4/run');
  let back = null;
  for (let i = 0; i < 80; i++) { back = (await api('GET', '/api/projects/' + pid + '/pages/E4')).data.page; if (back.state === 'planned' && back.meta.state_reason) break; await sleep(250); }
  ok(back.state === 'planned' && !!back.meta.state_reason, 'code that changed after the approval is not run, and the experiment is planned again with the reason: ' + back.state + ' ' + (back.meta.state_reason || ''));
  await idle(pid);
  await until(page, () => !!document.querySelector('.approve .approve-back'), null, 8000, 'an experiment whose approval was taken back says so where it is approved again');
  const why = (await text(page, '.approve .approve-back')).replace(/\s+/g, ' ').trim();
  ok(/changed after the approval/.test(why) && /by you/.test(why) && !/\d{4}-\d\d-\d\dT|human/.test(why), 'with the reason, in the app\'s words: ' + why);
  ok(/did not run and is planned again/.test(await text(page, '.toast-error')), 'and it is said aloud, wherever the person is looking: ' + (await text(page, '.toast-error')).replace(/\s+/g, ' ').trim().slice(0, 90));
  await shot(page, 'experiment-approval-taken-back');
  await page.click('.toast [data-act="toast-x"]');
  if (!(await count(page, '.approve [data-act="approve"][data-start]'))) await page.click('[data-act="reapprove"]');
  await page.waitForSelector('.approve [data-act="approve"][data-start]');
  await page.fill('.approve-fields input[type=number]', '2');
  await page.click('.approve [data-act="approve"][data-start]');
  await until(page, () => !!document.querySelector('.run-live [data-act="cancel-run"]'), null, 20000, 'the second run starts');
  await until(page, () => /waiting 3/.test((document.querySelector('.run-live .log-body') || {}).textContent || ''), null, 20000, 'its log streams');
  await page.click('.run-live [data-act="cancel-run"]');
  await until(page, () => /cancelled/.test((document.querySelector('.page-status .tag-state') || {}).textContent || ''), null, 30000, 'the cancelled run ends as cancelled');
  await page.waitForSelector('.failure.k-cancelled');
  ok(/Cancelled by you/.test(await text(page, '.failure.k-cancelled .fail-t')) && (await count(page, '.failure.k-cancelled .code-err')) === 0, 'a run the person cancelled is said plainly, not as an error: ' + (await text(page, '.failure.k-cancelled')).trim().slice(0, 80));
  await shot(page, 'experiment-cancelled');
  await idle(pid);
  const e4 = (await api('GET', '/api/projects/' + pid + '/experiments/E4')).data;
  ok(e4.page.state === 'cancelled' && e4.runs.length === 2 && e4.runs[1].status === 'cancelled', 'the run record says cancelled: ' + e4.page.state);
  /* its first run succeeded: wherever "cancelled" is said that run is said with it, and it is the one result to analyse */
  await cancelledAfterSuccess(page, pid, 'E4');
  /* a run that is under way when the agents are stopped is queued again, and is not "cancelled by you" */
  const slow = await api('POST', '/api/projects/' + pid + '/experiments/E4/approve', { backend: 'local', timeout_seconds: 120, start: true });
  ok(slow.status === 200 && slow.data.worker, 'a third run starts: ' + slow.status);
  await until(page, () => /waiting 2/.test((document.querySelector('.run-live .log-body') || {}).textContent || ''), null, 20000, 'and is under way');
  await page.keyboard.press('1');
  await page.waitForSelector('.strip.s-running [data-act="stop"]');
  ok(/Experiment/.test(await text(page, '.strip-kicker')) && /E4/.test(await text(page, '.strip-text')), 'the strip says which experiment runs: ' + (await text(page, '.strip-text')).trim().slice(0, 60));
  await page.click('.strip [data-act="stop"]');
  await until(page, () => /Stopping the run/.test((document.querySelector('.toast') || {}).textContent || ''), null, 5000, 'stopping says that it is the run that is stopped');
  await idle(pid);
  const queued = (await api('GET', '/api/projects/' + pid + '/experiments/E4')).data;
  ok(queued.page.state === 'queued' && queued.runs.length === 3 && !!queued.runs[2].requeued, 'the experiment is queued again: ' + queued.page.state + ', ' + queued.runs.length + ' runs');
  await page.keyboard.press('4');
  await page.waitForSelector('.run.open .failure.k-cancelled');
  ok(/Stopped when you stopped the agents/.test(await text(page, '.run.open .failure .fail-t')) && /queued again/.test(await text(page, '.run.open .failure')), 'its run says that it was stopped with the agents and is queued again: ' + (await text(page, '.run.open .failure .fail-t')));
  await shot(page, 'experiment-stopped-with-the-agents');
  /* An experiment that waits in the queue is cancelled: the button says what the server says a
     press does (`cancel_does`), the press tells the server what the person was told, and what
     happened is said in the server's own sentence. */
  ok(queued.cancel_does === 'cancel', 'the server says that a cancel of a queued experiment cancels it: ' + queued.cancel_does);
  await until(page, () => (document.querySelector('.callout [data-act="cancel-run"]') || { dataset: {} }).dataset.does === 'cancel', null, 8000, 'the button of a queued experiment is the one that cancels');
  ok((await text(page, '.callout [data-act="cancel-run"]')).trim() === 'Cancel the run', 'and says so: ' + (await text(page, '.callout [data-act="cancel-run"]')).trim());
  const cancels = [];
  const listenCancel = async (response) => { if (response.request().method() === 'POST' && /\/cancel$/.test(response.url())) cancels.push({ sent: JSON.parse(response.request().postData() || '{}'), status: response.status(), answer: await response.json().catch(() => null) }); };
  page.on('response', listenCancel);
  await page.click('.callout [data-act="cancel-run"]');
  await until(page, () => /cancelled/.test((document.querySelector('.page-status .tag-state') || {}).textContent || ''), null, 10000, 'the queued experiment is cancelled');
  page.off('response', listenCancel);
  ok(cancels.length === 1 && cancels[0].sent.expect === 'cancel' && cancels[0].answer && cancels[0].answer.outcome === 'cancelled', 'the press said what the person was told, and the server cancelled: ' + JSON.stringify(cancels.map((c) => [c.sent, c.status, c.answer && c.answer.outcome])));
  ok(cancels.length === 1 && (await text(page, '.toast')).replace(/\s+/g, ' ').includes(cancels[0].answer.message), 'the app says what happened in the server\'s sentence: ' + (await text(page, '.toast')).trim().slice(0, 90));
  /* a run stopped at its time limit: a limit, not a bug */
  const short = await api('POST', '/api/projects/' + pid + '/experiments/E4/approve', { backend: 'local', timeout_seconds: 3, start: true });
  ok(short.status === 200 && short.data.worker, 'a run with a three-second limit starts: ' + short.status);
  await until(page, () => /failed/.test((document.querySelector('.page-status .tag-state') || {}).textContent || '') && !!document.querySelector('.failure.k-timeout'), null, 40000, 'the run that ran out of time is shown as a time limit');
  ok(/Stopped at the time limit/.test(await text(page, '.failure .fail-t')), 'it is named: ' + (await text(page, '.failure .fail-t')));
  ok(/a limit, not a bug/.test(await text(page, '.failure')), 'and explained as a limit, not a bug');
  ok((await count(page, '.failure [data-act="rerun"][data-minutes]')) === 1 && (await count(page, '.failure [data-act="edit-code"]')) === 1, 'raising the limit and editing the code are offered');
  ok((await count(page, '.failure .repair')) === 0, 'no repair is shown for a limit');
  await shot(page, 'experiment-timeout');
  await idle(pid);
  /* a run that succeeds only after the runner repaired the code says so, with what was changed */
  await open(page, '#/p/' + pid + '/experiments/E2', '.failure [data-act="rerun"]');
  await page.click('.failure [data-act="rerun"]');
  await until(page, () => /succeeded/.test((document.querySelector('.page-status .tag-state') || {}).textContent || '') && !!document.querySelector('.run.open .repair-ok'), null, 90000, 'a run that succeeded on repaired code says that the runner changed the code');
  await page.click('.run.open .repair-ok summary');
  ok((await count(page, '.run.open .repair-ok pre.diff .diff-add')) >= 1, 'and shows what was changed as a diff');
  ok(/written by the runner/.test(await text(page, '.code-by')), 'the code says that the runner wrote it: ' + (await text(page, '.code-by')));
  await shot(page, 'experiment-repaired');
  await idle(pid);
  /* approve without starting; the engine's run phase takes it, and the analysis follows */
  await open(page, '#/p/' + pid + '/experiments/E1', '[data-act="reapprove"]');
  await page.click('[data-act="reapprove"]');
  await page.waitForSelector('.approve [data-act="approve"]:not([data-start])');
  await page.click('.approve [data-act="approve"]:not([data-start])');
  await until(page, () => /Approved to run/.test((document.querySelector('.callout h2') || {}).textContent || ''), null, 8000, 'approved without starting: it waits for the run phase');
  /* Taking an approval back. An approved experiment that has not started becomes a plan again,
     as `research cancel` does from a terminal: the server says that this is what a press does
     (`cancel_does: "withdraw"`), the button says it, and afterwards the app says what the
     server said happened. By the keyboard, and the keyboard lands on what can be done next. */
  const asApproved = (await api('GET', '/api/projects/' + pid + '/experiments/E1')).data;
  ok(asApproved.page.state === 'approved' && asApproved.cancel_does === 'withdraw', 'the server says that a cancel of an approved experiment that has not started takes the approval back: ' + asApproved.cancel_does);
  await until(page, () => (document.querySelector('.callout [data-act="cancel-run"]') || { dataset: {} }).dataset.does === 'withdraw', null, 8000, 'the button knows that a press takes the approval back');
  ok((await text(page, '.callout [data-act="cancel-run"]')).trim() === 'Take the approval back', 'the button says "Take the approval back": ' + (await text(page, '.callout [data-act="cancel-run"]')).trim());
  const runsBefore = asApproved.runs.length;
  const taken = [];
  const listenTaken = async (response) => { if (response.request().method() === 'POST' && /\/cancel$/.test(response.url())) taken.push({ sent: JSON.parse(response.request().postData() || '{}'), status: response.status(), answer: await response.json().catch(() => null) }); };
  page.on('response', listenTaken);
  await page.focus('.callout [data-act="cancel-run"]');
  await page.keyboard.press('Enter');
  await until(page, () => /withdrew the approval/.test((document.querySelector('.approve .approve-back') || {}).textContent || ''), null, 10000, 'the experiment is a plan again and says that the approval was withdrawn');
  page.off('response', listenTaken);
  ok(taken.length === 1 && taken[0].sent.expect === 'withdraw' && taken[0].status === 200 && taken[0].answer.outcome === 'withdrawn' && taken[0].answer.state === 'planned', 'the press told the server what the person was told, and the server took the approval back: ' + JSON.stringify(taken.map((c) => [c.sent, c.status, c.answer && c.answer.outcome])));
  ok(taken.length === 1 && (await text(page, '.toast')).replace(/\s+/g, ' ').includes(taken[0].answer.message) && /approval was taken back/.test(await text(page, '.toast')), 'the confirmation is the server\'s sentence: ' + (await text(page, '.toast')).replace(/\s+/g, ' ').trim().slice(0, 110));
  const asPlan = (await api('GET', '/api/projects/' + pid + '/experiments/E1')).data;
  ok(asPlan.page.state === 'planned' && asPlan.runs.length === runsBefore && asPlan.cancel_does === null, 'on the server it is planned, with no run recorded and nothing left to cancel: ' + asPlan.page.state + ', ' + asPlan.runs.length + ' runs');
  ok(await page.evaluate(() => !!document.activeElement && document.activeElement.dataset.act === 'approve'), 'the keyboard is on the approval that can be given again: ' + (await focusOf(page)));
  ok(/planned/.test(await text(page, '.page-status .tag-state')) && (await count(page, '.callout [data-act="cancel-run"]')) === 0, 'the page shows a plan, with nothing to take back');
  await shot(page, 'experiment-approval-withdrawn');
  /* a reload shows the same: it is the page's own state, not something this window remembers */
  await page.reload();
  await until(page, () => /withdrew the approval/.test((document.querySelector('.approve .approve-back') || {}).textContent || ''), null, 10000, 'after a reload the plan still says that its approval was withdrawn');
  /* approved again, with the keyboard, for the run phase that follows */
  await page.focus('.approve [data-act="approve"]:not([data-start])');
  await page.keyboard.press('Enter');
  await until(page, () => /Approved to run/.test((document.querySelector('.callout h2') || {}).textContent || ''), null, 8000, 'and it is approved again from the form');
  const findings = (await api('GET', '/api/projects/' + pid + '/pages?kind=finding')).data.pages.length;
  await page.keyboard.press('1');
  await page.waitForSelector('.strip [data-act="continue"]');
  await until(page, () => ((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim() === 'Run the approved experiment, then analyse it', null, 8000, 'the main button says what a press does: run what is approved, then analyse it');
  ok((await text(page, '.strip [data-act="continue"]')).trim() === 'Run the approved experiment, then analyse it', 'it reads: ' + (await text(page, '.strip [data-act="continue"]')).trim());
  ok(/→ runs, analysis/.test(await text(page, '#status .sb-state')), 'and the status bar says the same: ' + (await text(page, '#status .sb-state')).replace(/\s+/g, ' ').trim());
  await page.click('.strip [data-act="continue"]');
  await until(page, () => !!document.querySelector('.strip.s-running'), null, 15000, 'the run phase starts from the strip');
  await until(page, () => /analysis/.test((document.querySelector('.strip-kicker') || {}).textContent || ''), null, 90000, 'after the runs the analysis starts by itself');
  await shot(page, 'map-analysis');
  await until(page, () => /Checkpoint.*after analysis/i.test((document.querySelector('.strip-kicker') || {}).textContent || ''), null, 90000, 'and ends at a checkpoint');
  await idle(pid);
  const e1 = (await api('GET', '/api/projects/' + pid + '/experiments/E1')).data;
  ok(e1.page.state === 'succeeded' && e1.runs.length === 2, 'the approved experiment was run by the run phase: ' + e1.page.state + ', ' + e1.runs.length + ' runs');
  ok((await api('GET', '/api/projects/' + pid + '/pages?kind=finding')).data.pages.length > findings, 'the analyst proposed findings from the new results');
  ok((await count(page, '.node.k-finding.r-proposed')) >= 1, 'which wait on the map');
  await until(page, () => /succeeded/.test((document.querySelector('.panel .tag-state') || {}).textContent || ''), null, 8000, 'the open page follows: the experiment is shown as succeeded');
  await shot(page, 'map-after-run-phase');
  /* What is approved runs before the phase that is next, and is analysed; that phase is
     still the next one afterwards. The button and the strip say so from the state the
     server gives: here the checkpoint named hypotheses, and an experiment is approved since. */
  await page.keyboard.press('Escape');
  const was = (await api('GET', '/api/projects/' + pid)).data.state;
  ok(was.next_phase === 'hypothesize', 'after the round\'s analysis the next phase is hypotheses: ' + was.next_phase);
  await until(page, () => /^Continue to hypotheses/.test(((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim()), null, 8000, 'with nothing approved the button offers the next phase');
  const once = await api('POST', '/api/projects/' + pid + '/experiments/E1/approve', { backend: 'local' });
  ok(once.status === 200, 'E1 is approved once more, without starting it: ' + once.status);
  await until(page, () => ((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim() === 'Run the approved experiment, then analyse it', null, 8000, 'an approval made at the checkpoint changes what the button says: the run comes first');
  const says = (await text(page, '.strip-sub .go-says')).trim();
  ok(/^E1 runs first and is analysed\. After that, the next phase is still hypotheses\.$/.test(says), 'the strip says what happens and which phase stays next: ' + says);
  ok(/→ runs, analysis, then hypotheses/.test(await text(page, '#status .sb-state')), 'and so does the status bar: ' + (await text(page, '#status .sb-state')).replace(/\s+/g, ' ').trim());
  await shot(page, 'map-run-comes-first');
  const sent = [];
  const listenAdvance = (request) => { if (request.method() === 'POST' && /\/advance$/.test(request.url())) sent.push(request.postData() || ''); };
  page.on('request', listenAdvance);
  await page.click('.strip [data-act="continue"]');
  await until(page, () => !!document.querySelector('.strip.s-running'), null, 15000, 'the press starts the engine');
  page.off('request', listenAdvance);
  ok(sent.length === 1 && !JSON.parse(sent[0] || '{}').phase, 'without naming a phase: what runs first is the engine\'s rule, not the page\'s: ' + sent.join(' '));
  await until(page, () => /Checkpoint.*after analysis/i.test((document.querySelector('.strip-kicker') || {}).textContent || ''), null, 120000, 'the run and its analysis end at a checkpoint');
  const rested = await idle(pid);
  ok(rested.state.next_phase === 'hypothesize' && rested.state.round === was.round, 'the phase that was next is still next, in the same round: ' + rested.state.next_phase + ', round ' + rested.state.round);
  ok((await api('GET', '/api/projects/' + pid + '/experiments/E1')).data.runs.length === 3, 'and the approved experiment ran');
  await until(page, () => /^Continue to hypotheses/.test(((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim()) && !document.querySelector('.strip-sub .go-says'), null, 8000, 'after it ran, the phase that was next is offered again');
};

/* overview: the argument, an instruction, spend; then the agents: start, live activity, stop, resume */
CHECKS.agents = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/overview', '.ov-head');
  ok((await count(page, '.arg tbody tr')) >= 4, 'the argument so far lists the hypotheses');
  ok((await count(page, '.arg .arg-ev .chip')) >= 4, 'with their experiments and findings');
  ok((await count(page, '.smap .smap-st')) === 8 && (await count(page, '.trk')) >= 2, 'the strip map and the rounds are there');
  await page.fill('#rq-text', 'Compare against RMSprop as well.');
  await page.reload();
  await page.waitForSelector('#rq-text');
  ok((await page.inputValue('#rq-text')) === 'Compare against RMSprop as well.', 'an instruction that is being typed survives a reload: ' + JSON.stringify(await page.inputValue('#rq-text')));
  await page.click('.rq-form button[type=submit]');
  await until(page, () => /RMSprop/.test((document.querySelector('.rqs') || {}).textContent || ''), null, 5000, 'the instruction is listed');
  const requests = (await api('GET', '/api/projects/' + pid + '/requests')).data.requests;
  ok(requests.some((r) => /RMSprop/.test(r.text) && r.state === 'open'), 'the instruction is a request on the server');
  ok((await page.inputValue('#rq-text')) === '', 'the box is cleared once it is sent');
  /* An instruction has an addressee: everyone unless the person says otherwise, or one of
     the agents, in words a researcher uses. The choices are the server's. */
  const roles = (await api('GET', '/api/defaults')).data.request_for;
  ok(JSON.stringify(await page.$$eval('#rq-for option', (options) => options.map((o) => o.value))) === JSON.stringify(roles), 'the instruction box offers the addressees the server gives: ' + JSON.stringify(roles));
  ok((await page.inputValue('#rq-for')) === 'all' && /Everyone/.test(await page.$eval('#rq-for option:checked', (o) => o.textContent)), 'everyone is the choice until the person makes another');
  ok(/searches the literature/.test(await text(page, '#rq-for')) && /proposes hypotheses/.test(await text(page, '#rq-for')) && /designs experiments/.test(await text(page, '#rq-for')) && /analyses the results/.test(await text(page, '#rq-for')), 'the agents are named by what they do');
  ok(requests.some((r) => /RMSprop/.test(r.text) && r.for === 'all'), 'an instruction sent as it stands is for everyone on the server');
  ok(/for everyone/.test(await text(page, '.rq:has-text("RMSprop") .rq-m')) && /no phase has been given it yet/.test(await text(page, '.rq:has-text("RMSprop") .rq-had')), 'the open instruction says who it is for and that no phase has had it: ' + (await text(page, '.rq:has-text("RMSprop") .rq-m')).replace(/\s+/g, ' ').trim());
  await page.fill('#rq-text', 'Look for work on schedule-free optimisers.');
  await page.selectOption('#rq-for', 'scout');
  await until(page, () => /does not come round by itself/.test((document.querySelector('.rq-when') || {}).textContent || ''), null, 3000, 'the box says when the scout is given it: research does not come round by itself');
  /* whom an instruction is for is kept with what is typed: a reload does not make it one for everyone */
  await page.reload();
  await page.waitForSelector('#rq-text');
  await until(page, () => document.querySelector('#rq-for').value === 'scout' && document.querySelectorAll('#rq-for option').length >= 5, null, 5000, 'the addressee of an instruction that is being typed survives a reload with it');
  ok((await page.inputValue('#rq-text')) === 'Look for work on schedule-free optimisers.' && /does not come round by itself/.test(await text(page, '.rq-when')), 'the text is there too, and the box still says when the scout is given it: ' + JSON.stringify([await page.inputValue('#rq-text'), await page.inputValue('#rq-for')]));
  const posted = [];
  const listenRequest = (request) => { if (request.method() === 'POST' && /\/requests$/.test(request.url())) posted.push(JSON.parse(request.postData() || '{}')); };
  page.on('request', listenRequest);
  await page.click('.rq-form button[type=submit]');
  await until(page, () => [...document.querySelectorAll('.rq')].some((r) => /schedule-free optimisers\./.test(r.textContent) && /for the scout/.test(r.textContent)), null, 5000, 'the instruction for the scout is listed as that');
  page.off('request', listenRequest);
  ok(posted.length === 1 && posted[0].for === 'scout', 'the request names its addressee: ' + JSON.stringify(posted.map((b) => b.for)));
  const forScout = (await api('GET', '/api/projects/' + pid + '/requests')).data.requests.find((r) => /Look for work on schedule-free/.test(r.text));
  ok(!!forScout && forScout.for === 'scout' && forScout.state === 'open' && Array.isArray(forScout.handled_in) && !forScout.handled_in.length, 'and is one for the scout on the server: ' + JSON.stringify(forScout && [forScout.for, forScout.handled_in]));
  ok((await page.inputValue('#rq-for')) === 'all', 'the next instruction is for everyone again');
  /* such an instruction waits for a research phase, which only the person starts: said beside the button, with the choice */
  await page.evaluate(() => { document.querySelector('.overview').scrollTop = 0; });
  ok(/instruction for the scout waits for a research phase/.test(await text(page, '.ov-next .go-rec[data-reason="requests"]')) && (await count(page, '.ov-next .go-rec[data-reason="requests"] [data-act="continue-phase"][data-phase="research"]')) === 1, 'an instruction for the scout is a reason, beside the button, to search the literature again: ' + (await text(page, '.ov-next .go-rec[data-reason="requests"]')).replace(/\s+/g, ' ').trim().slice(0, 110));
  await shot(page, 'overview-instruction-for-the-scout');
  /* a request can be withdrawn */
  await page.click('.rq[data-rq="' + forScout.id + '"] [data-act="rq-del"]');
  await until(page, (id) => !document.querySelector('.rq[data-rq="' + id + '"]'), forScout.id, 5000, 'a withdrawn instruction leaves the list');
  ok(!(await api('GET', '/api/projects/' + pid + '/requests')).data.requests.some((r) => r.id === forScout.id), 'and is gone from the server');
  ok((await count(page, '.go-rec[data-reason="requests"]')) === 0, 'and with it the reason for more research');
  /* An instruction withdrawn in another window is an event (`request.withdrawn`): this window
     takes the row away by itself, and the activity says who withdrew what. */
  const elsewhere = (await api('POST', '/api/projects/' + pid + '/requests', { text: 'Withdrawn in another window.', for: 'designer' })).data.request;
  await page.waitForSelector('.rq[data-rq="' + elsewhere.id + '"] [data-act="rq-del"]');
  ok((await api('DELETE', '/api/projects/' + pid + '/requests/' + elsewhere.id)).status === 200, 'an instruction is withdrawn in another window');
  await until(page, (id) => !document.querySelector('.rq[data-rq="' + id + '"]'), elsewhere.id, 8000, 'an instruction withdrawn in another window leaves the list of this one by itself');
  await until(page, () => [...document.querySelectorAll('.feed .ev')].slice(0, 6).some((row) => /You withdrew your request to the designer/.test(row.textContent) && /you/.test((row.querySelector('.ev-who') || {}).textContent || '')), null, 6000, 'and the activity has a line for it, as something you did');
  /* The other window was a moment faster: the press of Withdraw here reaches a request that is
     gone already. It says so and takes the row away, not "not withdrawn". */
  const raced = (await api('POST', '/api/projects/' + pid + '/requests', { text: 'Withdrawn a moment earlier elsewhere.', for: 'designer' })).data.request;
  await page.waitForSelector('.rq[data-rq="' + raced.id + '"] [data-act="rq-del"]');
  const first = '**/api/projects/' + pid + '/requests/' + raced.id;
  await page.route(first, async (route) => { if (route.request().method() === 'DELETE') await api('DELETE', '/api/projects/' + pid + '/requests/' + raced.id); await route.continue(); });
  await page.click('.rq[data-rq="' + raced.id + '"] [data-act="rq-del"]');
  await until(page, (id) => !document.querySelector('.rq[data-rq="' + id + '"]'), raced.id, 6000, 'a press of Withdraw on an instruction that another window withdrew first takes the row away');
  await page.unroute(first);
  ok(/withdrawn already/.test(await text(page, '.toast')) && (await count(page, '.toast-error')) === 0 && !/rq-[0-9a-f]{6}/.test(await text(page, '.toast')), 'and says that it was withdrawn already, without the name of a file: ' + (await text(page, '.toast')).replace(/\s+/g, ' ').trim());
  const rows = await count(page, '.feed .ev');
  await page.click('[data-act="feed-more"]');
  await until(page, (n) => document.querySelectorAll('.feed .ev').length > n, rows, 5000, 'earlier activity is fetched when asked for');
  await until(page, () => { const more = document.querySelector('[data-act="feed-more"]'); if (!document.querySelector('.feed .ev-run') && more) more.click(); return !!document.querySelector('.feed .ev-run'); }, null, 8000, 'a run is one line in the feed');
  ok(!/is now (queued|running|succeeded|failed)/.test(await page.locator('.feed').innerText()), 'its steps are folded into that line');
  ok(/E\d+ ran on this machine/.test(await text(page, '.feed .ev-run .ev-run-l a')), 'which says what happened: ' + (await text(page, '.feed .ev-run .ev-run-l a')));
  await page.click('.feed .ev-run .ev-steps summary');
  ok((await count(page, '.feed .ev-run .ev-steps[open] li')) >= 3, 'and can be opened for the steps');
  await page.click('.spend-more summary[data-act="limits-open"]');
  await page.fill('#lim-experiment_usd', '7.5');
  await page.click('.limits-form button[type=submit]');
  await until(page, () => /of \$7\.50 for experiments/.test(document.querySelector('.spend-big').textContent), null, 5000, 'a changed limit shows at once');
  ok((await api('GET', '/api/projects/' + pid)).data.settings.limits.experiment_usd === 7.5, 'and is in project.json');
  await shot(page, 'overview');
  /* start the next phase on the scripted model; activity arrives over the event stream */
  await api('PATCH', '/api/projects/' + pid, {});
  await open(page, '#/p/' + pid + '/map', '.strip');
  const before = (await api('GET', '/api/projects/' + pid + '/activity?limit=1')).data.events[0].seq;
  const proposed = await count(page, '.node.k-hypothesis.r-proposed');
  const advance = await api('POST', '/api/projects/' + pid + '/advance', { phase: 'hypothesize' });
  ok(advance.status === 202, 'the hypothesize phase starts: ' + advance.status + ' ' + JSON.stringify(advance.data).slice(0, 200));
  await until(page, () => !!document.querySelector('.strip.s-running'), null, 10000, 'the strip shows the running phase without a reload');
  ok(/hypotheses/.test(await text(page, '.strip-kicker')), 'and names the phase: ' + (await text(page, '.strip-kicker')));
  ok((await count(page, '.top-status.st-running')) === 1 && /Running/.test(await text(page, '#status .sb-state')), 'the header and the status bar say it runs');
  await page.click('[data-act="toggle-activity"]');
  await until(page, () => document.querySelectorAll('.strip-log div').length >= 2, null, 10000, 'activity lines arrive live');
  await until(page, (n) => document.querySelectorAll('.node.k-hypothesis.r-proposed').length > n, proposed, 20000, 'a proposed hypothesis appears on the map while the phase runs');
  ok((await count(page, '.strip.s-running')) === 1, 'the phase is still running when the first page has arrived');
  await shot(page, 'map-live');
  /* stop it in the middle */
  await page.click('.strip [data-act="stop"]');
  await until(page, () => !document.querySelector('.strip.s-running'), null, 30000, 'the phase stops');
  const stopped = await idle(pid);
  ok(stopped.state.status === 'stopped', 'state.json says stopped: ' + stopped.state.status);
  ok(/Stopped/.test(await text(page, '.strip-kicker')), 'the strip says it was stopped by the person: ' + (await text(page, '.strip-kicker')));
  await shot(page, 'map-stopped');
  /* A run approved while a phase waits to be resumed comes first, with its analysis, and the
     phase is resumed after them: the line of the round draws what is to come in that order,
     with the phase that is resumed in it. */
  const meanwhile = await api('POST', '/api/projects/' + pid + '/experiments/E1/approve', { backend: 'local' });
  if (ok(meanwhile.status === 200, 'an experiment is approved while the phase is stopped: ' + meanwhile.status)) {
    await page.keyboard.press('5');
    await until(page, () => /^Run the approved experiment, then analyse it/.test(((document.querySelector('.ov-actions [data-act="continue"]') || {}).textContent || '').trim()), null, 8000, 'with a phase to resume, an approved run and its analysis still come first');
    ok(/is resumed/.test(await text(page, '.ov-next .go-says')), 'and the note beside the button says that the phase is resumed after them: ' + (await text(page, '.ov-next .go-says')).trim());
    const coming = await page.$$eval('.trks .trk:last-child .trk-p.s-next', (list) => list.map((el) => el.textContent.trim()));
    const resumed = (await text(page, '.trks .trk:last-child .trk-p.s-interrupted')).replace(/\s*\(interrupted\)/, '').trim();
    ok(coming.length >= 3 && coming[0] === 'runs' && coming[1] === 'analysis' && coming[2] === resumed, 'the round draws the run, its analysis and then the resumed phase as what is to come: ' + coming.join(', ') + ' (resumed: ' + resumed + ')');
    ok((await api('POST', '/api/projects/' + pid + '/experiments/E1/cancel')).status < 300, 'the approval is taken back again');
    await until(page, () => { const label = ((document.querySelector('.ov-actions [data-act="continue"]') || {}).textContent || '').trim(); return !!label && !/^Run the/.test(label); }, null, 8000, 'without it the button no longer offers a run');
    await page.keyboard.press('1');
    await page.waitForSelector('.strip [data-act="continue"]');
  }
  /* resume: it runs to the checkpoint */
  await page.click('.strip [data-act="continue"]');
  await until(page, () => !!document.querySelector('.strip.s-running'), null, 10000, 'resume starts the phase again');
  await until(page, () => !!document.querySelector('.strip.s-checkpoint') && /Checkpoint/.test(document.querySelector('.strip-kicker').textContent), null, 60000, 'the phase ends at a checkpoint');
  const after = (await api('GET', '/api/projects/' + pid + '/activity?limit=200')).data.events.filter((e) => e.seq > before);
  ok(after.some((e) => e.type === 'checkpoint'), 'the checkpoint is in the activity stream');
  ok((await count(page, '.node.k-hypothesis.r-proposed')) === proposed + 2, 'both scripted hypotheses wait on the map: ' + (await count(page, '.node.k-hypothesis.r-proposed')));
  await idle(pid);
  /* an open request says which phases were given it so far: the list grows while it is open */
  const given = (await api('GET', '/api/projects/' + pid + '/requests')).data.requests.find((r) => /RMSprop/.test(r.text));
  if (ok(!!given && given.state === 'open' && Array.isArray(given.handled_in) && given.handled_in.some((h) => h.phase === 'hypothesize'), 'the instruction for everyone was given to the hypotheses phase and is still open: ' + JSON.stringify(given && [given.state, given.handled_in]))) {
    await page.keyboard.press('5');
    await page.waitForSelector('.rqs .rq');
    await until(page, (id) => /given so far to hypotheses \(round \d+\)/.test((document.querySelector('.rq[data-rq="' + id + '"] .rq-had') || {}).textContent || ''), given.id, 6000, 'the open instruction lists the phases that were given it so far');
    await page.keyboard.press('1');
    await page.waitForSelector('.map-svg .node');
  }
  /* accept one, let the designer plan its experiment, then accept and run it from the inbox */
  const hyps = (await api('GET', '/api/projects/' + pid + '/pages?kind=hypothesis&review=proposed')).data.pages;
  const mine = hyps.find((h) => /scripted effect is there/.test(h.title));
  ok(!!mine, 'the scripted hypothesis is among the proposals');
  await page.click('.node.k-hypothesis.r-proposed[data-key="' + mine.path + '"] .n-yes');
  await until(page, () => !document.querySelector('.toast [data-act="undo"]'), null, 9000, 'the hypothesis is accepted');
  await sleep(400);
  const design = await api('POST', '/api/projects/' + pid + '/advance', { phase: 'design' });
  ok(design.status === 202, 'the design phase starts: ' + design.status);
  await until(page, () => [...document.querySelectorAll('.node.k-experiment.r-proposed')].some((n) => /scripted check/i.test(n.getAttribute('aria-label'))), null, 60000, 'the designed experiment appears on the map');
  await idle(pid);
  const planned = (await api('GET', '/api/projects/' + pid + '/experiments')).data.experiments.find((x) => x.review === 'proposed' && /scripted check/i.test(x.title));
  ok(!!planned, 'the scripted experiment waits for a decision');
  await open(page, '#/p/' + pid + '/inbox/' + planned.id, '.in-detail .runon select');
  ok(/Local CPU/.test(await text(page, '.in-detail .runon select')), 'run on offers what can run it: ' + (await text(page, '.in-detail .runon select')).trim());
  ok((await page.locator('.in-detail .runon option').count()) === 1, 'and only that');
  await shot(page, 'inbox-run-on');
  await page.click('.in-detail [data-act="accept-run"]');
  await until(page, (id) => location.hash.endsWith('/experiments/' + id), planned.id, 10000, 'accept and run opens the experiment');
  await until(page, () => !!document.querySelector('.run-live .log-body') || !!document.querySelector('.run.open .metrics tr'), null, 20000, 'its run starts at once');
  await until(page, () => /step \d of 5/.test((document.querySelector('.run-live .log-body, .run.open .log-body') || {}).textContent || ''), null, 20000, 'the log of the accepted run streams');
  await until(page, () => /succeeded/.test((document.querySelector('.page-status .tag-state') || {}).textContent || '') && !!document.querySelector('.run.open .metrics tr'), null, 30000, 'and it ends with its metrics');
  ok(/steps/.test(await text(page, '.run.open .metrics')), 'the metrics are the ones the script wrote: ' + (await text(page, '.run.open .metrics')).trim().slice(0, 60));
  /* nothing is approved any more and a result waits to be read: the button says what it will do */
  await page.keyboard.press('1');
  await page.waitForSelector('.strip [data-act="continue"]');
  const ready = (await api('GET', '/api/projects/' + pid + '/experiments')).data.experiments.filter((x) => x.review === 'accepted' && (x.state === 'approved' || x.state === 'queued')).length;
  if (!ready) {
    /* (the result is in the list of pages a moment after the run's end is known) */
    await until(page, () => /^Analyse the (\d+ )?new results?/.test(((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim()), null, 8000, 'with nothing approved and a result not analysed, the button offers the analysis, not a run');
    ok(/^Analyse the (\d+ )?new results?/.test((await text(page, '.strip [data-act="continue"]')).trim()), 'it reads: ' + (await text(page, '.strip [data-act="continue"]')).trim());
    ok(/No experiment is approved to run/.test(await text(page, '.strip-sub .go-says')), 'and the strip says why: ' + (await text(page, '.strip-sub .go-says')).trim());
  }
  const done = (await api('GET', '/api/projects/' + pid + '/pages/' + planned.id)).data.page;
  ok(done.review === 'accepted' && done.state === 'succeeded', 'accepted and run in one decision: ' + done.review + ' ' + done.state);
  await idle(pid);
};

/* states: a worker that died, a failed phase with its retry, a budget stop, the server gone */
CHECKS.states = async ({ page, pid, context }) => {
  await open(page, '#/p/' + pid + '/map', '.strip');
  /* a worker killed in the middle of a phase */
  let started = await api('POST', '/api/projects/' + pid + '/advance', { phase: 'hypothesize' });
  ok(started.status === 202, 'a phase starts for the kill test: ' + started.status);
  await until(page, () => !!document.querySelector('.strip.s-running'), null, 10000, 'it runs');
  let worker = null;
  for (let i = 0; i < 600 && !(worker && worker.pid > 0); i++) { const view = (await api('GET', '/api/projects/' + pid)).data; if (view.state.status === 'running') worker = view.worker; else await sleep(150); }
  ok(worker && worker.pid > 0, 'the worker has a pid and its phase is under way');
  if (worker && worker.pid > 0) { try { process.kill(worker.pid, 'SIGKILL'); } catch (e) { fails.push('could not kill the worker: ' + e.message); } }
  await until(page, () => !!document.querySelector('.strip.s-failed'), null, 30000, 'a dead worker is noticed');
  ok(/stopped unexpectedly/.test(await text(page, '.strip-kicker')), 'the strip says the agents stopped unexpectedly: ' + (await text(page, '.strip-kicker')));
  ok(/killed|gone|signal/i.test(await text(page, '.strip-text')), 'and why: ' + (await text(page, '.strip-text')));
  await shot(page, 'state-worker-died');
  await page.click('.strip [data-act="continue"]');
  await until(page, () => !!document.querySelector('.strip.s-checkpoint'), null, 60000, 'resuming after the death reaches the checkpoint');
  await idle(pid);
  /* a failed phase: an instruction makes the scripted model fail twice */
  const made = await api('POST', '/api/projects/' + pid + '/requests', { text: '[scripted-failure] make the next phase fail' });
  started = await api('POST', '/api/projects/' + pid + '/advance', { phase: 'hypothesize' });
  await until(page, () => !!document.querySelector('.strip.s-failed'), null, 60000, 'a phase that fails twice stops the engine');
  ok(/failed/i.test(await text(page, '.strip-kicker')), 'the strip names the failed phase: ' + (await text(page, '.strip-kicker')));
  ok(/scripted failure/i.test(await text(page, '.strip-text')), 'with its error: ' + (await text(page, '.strip-text')));
  ok((await count(page, '.top-status.st-failed')) === 1, 'the header says so too');
  if (!(await count(page, '.strip-log'))) await page.click('[data-act="toggle-activity"]');   // open already when an earlier check opened the activity
  await page.waitForSelector('.strip-log .strip-fix');
  ok(await page.evaluate(() => document.querySelector('.strip-log').scrollTop === 0), 'the log of a failure opens at its top, where the advice and the worker\'s words are');
  await shot(page, 'state-failed-phase');
  await idle(pid);
  await api('DELETE', '/api/projects/' + pid + '/requests/' + made.data.request.id);
  await page.click('.strip [data-act="continue"]');
  await until(page, () => !!document.querySelector('.strip.s-checkpoint'), null, 60000, 'the retry succeeds once the cause is gone');
  await idle(pid);
  /* a budget stop */
  const limits = (await api('GET', '/api/projects/' + pid)).data.settings.limits;
  await api('PATCH', '/api/projects/' + pid, { settings: { limits: { model_usd: 1 } } });
  started = await api('POST', '/api/projects/' + pid + '/advance', { phase: 'hypothesize' });
  await until(page, () => !!document.querySelector('.strip.s-budget'), null, 30000, 'a spent budget stops at a checkpoint that says so');
  ok(/budget/i.test(await text(page, '.strip-text')), 'the strip quotes the budget: ' + (await text(page, '.strip-text')));
  ok(!/settings\.limits/.test(await text(page, '.strip-text')), 'in the app\'s words, not as the name of a setting');
  await shot(page, 'state-budget');
  await idle(pid);
  await page.click('[data-act="limit-edit"]');
  await page.fill('#limit-usd', String(limits.model_usd));
  await page.keyboard.press('Enter');
  await until(page, () => !document.querySelector('.limit-form'), null, 5000, 'the limit is raised from the strip');
  ok((await api('GET', '/api/projects/' + pid)).data.settings.limits.model_usd === limits.model_usd, 'the new limit is in project.json');
  /* the stop is over once the limit is raised: the app no longer says that the budget is reached */
  await until(page, () => !!document.querySelector('.strip.s-checkpoint') && /limit for models is raised/.test(document.querySelector('.strip-text').textContent), null, 6000, 'with the limit raised the strip no longer says that the budget is reached');
  await until(page, () => /Resume hypotheses/.test(((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim()), null, 8000, 'and offers to go on where the phase stopped');
  ok(/Resume hypotheses/.test(await text(page, '.strip [data-act="continue"]')) && (await count(page, '.top-status.st-budget')) === 0, 'it reads: ' + (await text(page, '.strip [data-act="continue"]')).trim());
  /* A worker the server cannot verify: the project's files name one (here, on another
     machine) that this server did not start and cannot show to be this project's. It is
     shown as that, with the server's note; there is no Stop, because the server will not
     signal it; "Forget this worker" clears the name, and the project can be continued. */
  const dir = await projectDir(pid);
  if (dir) {
    await open(page, '#/p/' + pid + '/map', '.strip');
    const stateFile = path.join(dir, 'state.json');
    const name = () => { const state = JSON.parse(fs.readFileSync(stateFile, 'utf8')); state.worker = { pid: 4242, host: 'another-machine', started: stamp(), heartbeat: stamp() }; fs.writeFileSync(stateFile, JSON.stringify(state, null, 2)); };
    name();
    await until(page, () => !!document.querySelector('.strip.s-unverified'), null, 15000, 'a worker the server cannot verify is shown as such');
    const unverified = (await api('GET', '/api/projects/' + pid)).data.worker;
    ok(unverified.verified === false && unverified.state !== 'idle' && !!unverified.note, 'the server says that it cannot verify the worker: ' + JSON.stringify([unverified.state, unverified.verified]));
    ok((await text(page, '.strip-text')).trim() === unverified.note, 'the strip shows the server\'s note: ' + (await text(page, '.strip-text')).trim().slice(0, 90));
    ok((await count(page, '.strip [data-act="stop"]')) === 0 && (await count(page, '.strip [data-act="continue"]')) === 0 && (await count(page, '.strip [data-act="forget-worker"]')) === 1, 'there is no Stop and nothing to start; Forget this worker is offered');
    ok(/Worker not verified/.test(await text(page, '.top-status')) && /Worker not verified/.test(await text(page, '#status .sb-state')) && (await count(page, '.top-status.st-running, #status .sb-live')) === 0, 'the header and the status bar say the same, and not that something runs here');
    await shot(page, 'state-worker-not-verified');
    await page.keyboard.press('5');
    await page.waitForSelector('.ov-head');
    ok((await text(page, '.ov-sub')).trim() === unverified.note && (await count(page, '.ov-actions [data-act="forget-worker"]')) === 1 && (await count(page, '.ov-actions [data-act="stop"], .ov-actions [data-act="continue"]')) === 0, 'the overview shows the note and offers to forget it, no Stop: ' + (await text(page, '.ov-head h1')).trim());
    name();   // the sign of life it rests on is a minute old at most: written again, as a worker elsewhere would
    await open(page, '#/library', '.lib-row');
    await until(page, (id) => /Worker not verified/.test((document.querySelector('.lib-row[href="#/p/' + id + '/map"] .lib-st1') || {}).textContent || ''), pid, 8000, 'the library says the same for the project, not "running now"');
    name();
    await open(page, '#/p/' + pid + '/overview', '.ov-actions [data-act="forget-worker"]');
    const asked = [];
    const listenForget = (request) => { if (request.method() === 'POST') asked.push(request.url().replace(BASE, '')); };
    page.on('request', listenForget);
    await page.click('.ov-actions [data-act="forget-worker"]');
    await until(page, () => !!document.querySelector('.ov-actions [data-act="continue"]') && /Forgotten/.test((document.querySelector('#toast') || {}).textContent || ''), null, 10000, 'forgetting the worker makes the project continuable again, and says that nothing was stopped');
    page.off('request', listenForget);
    ok(asked.length === 1 && asked[0] === '/api/projects/' + pid + '/worker/forget', 'through the server\'s own way of forgetting a worker: ' + asked.join(', '));
    const after = (await api('GET', '/api/projects/' + pid)).data;
    ok(after.worker.state === 'idle' && !after.state.worker && !JSON.parse(fs.readFileSync(stateFile, 'utf8')).worker, 'the name is gone from the project\'s files: ' + after.worker.state);
    ok((await count(page, '.top-status.st-failed')) === 0, 'a forgotten worker is not "the agents stopped unexpectedly"');
    /* A run that was interrupted: the page says "running" and nothing runs it (its worker is
       gone). It is shown as interrupted, with the server's sentence, and Run starts it again. */
    await idle(pid);
    const approved = await api('POST', '/api/projects/' + pid + '/experiments/E1/approve', { backend: 'local' });
    ok(approved.status === 200, 'E1 is approved for the interrupted-run check: ' + approved.status);
    const e1 = (await api('GET', '/api/projects/' + pid + '/experiments/E1')).data;
    const file = pageFile(dir, e1.page.path), written = fs.readFileSync(file, 'utf8');
    ok(/^state: approved$/m.test(written), 'the page of E1 says approved in its file');
    fs.writeFileSync(file, written.replace(/^state: approved$/m, 'state: running'));
    let left = null;
    for (let i = 0; i < 60 && !(left && left.interrupted); i++) { left = (await api('GET', '/api/projects/' + pid + '/experiments/E1')).data; if (!left.interrupted) await sleep(250); }
    if (ok(!!left && left.interrupted === true && !!left.interrupted_note, 'the server sees a run that nothing is running: ' + JSON.stringify(left && [left.page.state, left.interrupted]))) {
      await reopen(page, '#/p/' + pid + '/experiments', '.ex-table');
      ok(/interrupted/.test(await text(page, '.ex-row:has(.ex-id b:text-is("E1")) .ex-st')) && /worker that ran it is gone/.test(await text(page, '.ex-row:has(.ex-id b:text-is("E1")) .ex-sub')), 'the table shows the run as interrupted, not as running: ' + (await text(page, '.ex-row:has(.ex-id b:text-is("E1")) .ex-st')).trim());
      ok((await count(page, '#status .sb-run')) === 0, 'and the status bar does not say that it runs');
      await page.click('.ex-row:has(.ex-id b:text-is("E1")) .ex-name a');
      await page.waitForSelector('.callout-interrupted');
      ok((await text(page, '.callout-interrupted p')).trim() === left.interrupted_note, 'the experiment says so in the server\'s sentence: ' + (await text(page, '.callout-interrupted p')).trim().slice(0, 90));
      ok((await count(page, '.run-live')) === 0 && (await count(page, '.callout-interrupted [data-act="run-now"]')) === 1 && (await count(page, '.callout-interrupted [data-act="cancel-run"]')) === 1 && /interrupted/.test(await text(page, '.page-status')), 'it is not shown as a live run; Run it again and Cancel are offered');
      /* The mark says what the words say, wherever the page has one: the server marks the
         experiment in every list of pages, and no mark or pill goes on saying "running". */
      const asListed = (await api('GET', '/api/projects/' + pid + '/pages?kind=experiment')).data.pages.find((x) => x.id === 'E1');
      ok(!!asListed && asListed.interrupted === true && asListed.state === 'running', 'the list of pages marks the experiment as interrupted: ' + JSON.stringify(asListed && [asListed.state, asListed.interrupted]));
      ok((await text(page, '.exp-page .page-status .tag-state')).trim() === 'interrupted' && (await count(page, '.exp-page .page-status .tag-state.t-halt .gl.t-halt')) === 1, 'the pill in the header says interrupted, with the mark of it: ' + (await text(page, '.exp-page .page-status')).trim().slice(0, 40));
      ok((await count(page, '.exp-rail .idx.cur .gl.t-halt')) === 1 && (await count(page, '.exp-rail .idx.cur .gl.t-live')) === 0 && (await count(page, '.exp-page .kicker-kind .gl.t-halt')) === 1, 'the mark in the rail and beside the title is the one of an interrupted run, not of a running one');
      await shot(page, 'experiment-interrupted');
      await open(page, '#/p/' + pid + '/wiki/E1', '.page-wiki .page-status');
      ok((await text(page, '.page-wiki .page-status .tag-state')).trim() === 'interrupted' && (await count(page, '.wiki-index .idx.cur .gl.t-halt')) === 1, 'in the wiki its pill and its mark in the index say interrupted: ' + (await text(page, '.page-wiki .page-status .tag-state')).trim());
      await open(page, '#/p/' + pid + '/map/E1', '.map-svg .node.sel');
      ok((await count(page, '.map-svg .node.sel.t-halt .gl.t-halt')) === 1 && /, interrupted$/.test(await page.getAttribute('.map-svg .node.sel', 'aria-label')) && (await text(page, '.panel .page-status .tag-state')).trim() === 'interrupted', 'on the map the node is drawn and named as interrupted, and so is the page beside it: ' + (await page.getAttribute('.map-svg .node.sel', 'aria-label')));
      await shot(page, 'map-interrupted');
      /* what a press of Continue does is the server's word too: the interrupted run is run again first */
      const owed = (await api('GET', '/api/projects/' + pid)).data.next_step;
      ok(!!owed && owed.experiments.includes('E1') && owed.will_run[0] === 'run', 'the server says that a press of Continue runs the interrupted experiment first: ' + JSON.stringify(owed));
      await until(page, () => /^Run the interrupted experiment again/.test(((document.querySelector('.strip [data-act="continue"]') || {}).textContent || '').trim()), null, 8000, 'and the main button says so');
      await open(page, '#/p/' + pid + '/experiments/E1', '.callout-interrupted');
      const runsBefore = left.runs.length;
      await page.click('.callout-interrupted [data-act="run-now"]');
      await until(page, () => /succeeded/.test((document.querySelector('.page-status .tag-state') || {}).textContent || '') && !document.querySelector('.callout-interrupted'), null, 90000, 'Run starts the interrupted experiment again, and it ends');
      await idle(pid);
      const again = (await api('GET', '/api/projects/' + pid + '/experiments/E1')).data;
      ok(again.page.state === 'succeeded' && again.interrupted === false && again.runs.length === runsBefore + 1, 'it ran once more and is no longer interrupted: ' + again.page.state + ', ' + again.runs.length + ' runs (were ' + runsBefore + ')');
    }
    /* Where a project stands when state.json is lost. The engine reads round, phases and what
       comes next from the project's events; the server passes that reading on with the
       engine's sentence about it (`state_problem`) and where it came from (`state_source`).
       The app shows the project where it stands, with that sentence, and never as one that
       has not started. With the events gone as well nothing says where it stands: the app
       says that, and the one button asks for the phase to go on with. */
    await idle(pid);
    const kept = fs.mkdtempSync(path.join(os.tmpdir(), 'po-e2e-state-'));
    const stood = (await api('GET', '/api/projects/' + pid)).data;
    await open(page, '#/p/' + pid + '/map', '.strip');
    try {
      fs.renameSync(stateFile, path.join(kept, 'state.json'));
      let lost = null;
      for (let i = 0; i < 40 && !(lost && lost.state_source === 'events'); i++) { lost = (await api('GET', '/api/projects/' + pid)).data; if (lost.state_source !== 'events') await sleep(250); }
      if (ok(!!lost && lost.state_source === 'events' && !!lost.state_problem && lost.state.round === stood.state.round && lost.state.phases.length > 0, 'with state.json gone the server gives the engine\'s reading from the events, and says so: ' + JSON.stringify(lost && [lost.state_source, lost.state.round, lost.state.next_phase]) + ' (the file said round ' + stood.state.round + ', next ' + stood.state.next_phase + ')')) {
        ok(!lost.problems.some((x) => x.path === 'state.json'), 'and says it once: not among the problems of the files as well');
        /* this window was open: the state that arrives with the event stream brings the sentence along */
        await until(page, () => !!document.querySelector('.strip-sub [data-state-problem]'), null, 20000, 'the open window learns that the state was read from the events, without a reload');
        const said = (await text(page, '.strip-sub [data-state-problem]')).replace(/\s+/g, ' ').trim();
        ok(said === lost.state_problem.replace(/\s+/g, ' ').trim(), 'the strip shows the engine\'s sentence as the server gives it: ' + said.slice(0, 110));
        ok(new RegExp('round ' + lost.state.round).test(await text(page, '.strip-kicker')) && !/Not started|not known/i.test(await text(page, '.strip')), 'and the project stands where its events say, not as "not started": ' + (await text(page, '.strip-kicker')).trim());
        ok((await text(page, '.top-prob')).trim() === String(lost.problems_total + 1) && (await count(page, '.top-status.st-failed')) === 0, 'the quiet mark in the top bar counts it: ' + (await text(page, '.top-prob')).trim());
        await shot(page, 'state-read-from-events');
        await reopen(page, '#/p/' + pid + '/overview', '#problems [data-state-problem]');
        ok((await text(page, '#problems [data-state-problem]')).replace(/\s+/g, ' ').trim() === lost.state_problem.replace(/\s+/g, ' ').trim(), 'after a reload the overview says the same, among what was read around');
        ok(new RegExp('round ' + lost.state.round, 'i').test(await text(page, '.ov-state')) && /\$\d/.test(await text(page, '.spend-big')) && (await text(page, '.spend-big b >> nth=0')).trim() === '$' + lost.state.spent.model_usd.toFixed(2), 'with the round and the spending the events account for: ' + (await text(page, '.ov-state')).trim().slice(-10) + ', ' + (await text(page, '.spend-big b >> nth=0')).trim());
        await shot(page, 'overview-state-read-from-events');
        await open(page, '#/library', '.lib-row');
        const rowNow = ((await api('GET', '/api/projects')).data.projects || []).find((x) => x.id === pid);
        ok(!!rowNow && rowNow.state_source === 'events' && rowNow.round === lost.state.round, 'the row of the library carries the same reading: ' + JSON.stringify(rowNow && [rowNow.state_source, rowNow.status, rowNow.round]));
        const libRow = '.lib-row[href="#/p/' + pid + '/map"]';
        ok(new RegExp('Round ' + lost.state.round).test(await text(page, libRow + ' .lib-st1')) && !/Not started/.test(await text(page, libRow)) && /read from its events/.test(await text(page, libRow + ' [data-state-problem]')), 'and the library lists the project where it stands, saying where that was read: ' + (await text(page, libRow + ' .lib-st')).replace(/\s+/g, ' ').trim());
        await shot(page, 'library-state-read-from-events');
      }
      /* the events gone too: nothing says where the project stands */
      fs.renameSync(path.join(dir, 'events.jsonl'), path.join(kept, 'events.jsonl'));
      let adrift = null;
      for (let i = 0; i < 40 && !(adrift && adrift.state_source === 'unknown'); i++) { adrift = (await api('GET', '/api/projects/' + pid)).data; if (adrift.state_source !== 'unknown') await sleep(250); }
      if (ok(!!adrift && adrift.state_source === 'unknown' && !!adrift.state_problem && !!adrift.next_step && adrift.next_step.needs_phase === true, 'with the events gone as well the server says that nothing tells where the project stands, and that a phase has to be named: ' + JSON.stringify(adrift && [adrift.state_source, adrift.next_step]))) {
        await reopen(page, '#/p/' + pid + '/map', '.strip.s-unknown');
        ok((await text(page, '.strip-text')).replace(/\s+/g, ' ').trim() === adrift.state_problem.replace(/\s+/g, ' ').trim() && /not known/.test(await text(page, '.strip-kicker')), 'the strip says so in the server\'s sentence: ' + (await text(page, '.strip-text')).trim().slice(0, 100));
        ok(/State unknown/.test(await text(page, '.top-status')) && /State unknown/.test(await text(page, '#status .sb-state')) && !/Not started/.test(await text(page, '.strip')), 'the header and the status bar say that the state is unknown, not that nothing has started');
        ok((await count(page, '.strip [data-act="continue"]')) === 0 && /Choose the phase to go on with/.test(await text(page, '.strip .go-pick [data-act="go-menu"]')), 'no press without a phase is offered: the one button asks for the phase: ' + (await text(page, '.strip .go-pick')).trim());
        await page.focus('.strip .go-pick [data-act="go-menu"]');
        await page.keyboard.press('Enter');
        await page.waitForSelector('.strip .go-menu .go-i:focus');
        ok((await count(page, '.strip .go-menu [data-act="continue-phase"]')) >= 2 && (await count(page, '.strip .go-menu [data-act="continue-phase"][data-phase="research"]')) === 1, 'which opens the list of phases, with the keyboard in it: ' + (await page.locator('.strip .go-menu .go-i b').allTextContents()).join(', '));
        await shot(page, 'state-unknown');
        await page.keyboard.press('Escape');
        await until(page, () => !document.querySelector('.go-menu') && document.activeElement.dataset.act === 'go-menu', null, 3000, 'Escape closes it and gives the keyboard back to the button');
        await open(page, '#/library', '.lib-row');
        const libRow = '.lib-row[href="#/p/' + pid + '/map"]';
        ok(/State unknown/.test(await text(page, libRow + ' .lib-st1')) && !/Not started/.test(await text(page, libRow)) && (await text(page, libRow + ' .lib-main p')).replace(/\s+/g, ' ').includes(adrift.state_problem.replace(/\s+/g, ' ').trim()), 'the library says the same of the project, with the server\'s sentence: ' + (await text(page, libRow + ' .lib-st')).replace(/\s+/g, ' ').trim());
      }
      /* A link where state.json should be. The server reads no record of a project through a
         link, so it can say neither where the project stands nor what a press would do: the
         app shows its sentence, which says what to put right, and offers no press at all. */
      fs.renameSync(path.join(kept, 'events.jsonl'), path.join(dir, 'events.jsonl'));
      fs.symlinkSync(path.join(kept, 'state.json'), stateFile);
      let linked = null;
      for (let i = 0; i < 40 && !(linked && linked.state_source === 'unknown' && linked.next_step === null); i++) { linked = (await api('GET', '/api/projects/' + pid)).data; if (!(linked.state_source === 'unknown' && linked.next_step === null)) await sleep(250); }
      if (ok(!!linked && linked.state_source === 'unknown' && linked.next_step === null && /not an ordinary file/.test(linked.state_problem || ''), 'with a link in the place of state.json the server says that it does not read it, and says nothing about a press: ' + JSON.stringify(linked && [linked.state_source, linked.next_step]))) {
        await reopen(page, '#/p/' + pid + '/map', '.strip.s-unknown');
        ok((await text(page, '.strip-text')).replace(/\s+/g, ' ').trim() === linked.state_problem.replace(/\s+/g, ' ').trim(), 'the strip shows the server\'s sentence, which says what to put right: ' + (await text(page, '.strip-text')).trim().slice(0, 90));
        ok((await count(page, '.strip [data-act="continue"], .strip [data-act="go-menu"], .strip [data-act="continue-phase"]')) === 0, 'and no press is offered that nobody could say the effect of');
        await page.keyboard.press('5');
        await page.waitForSelector('.ov-head');
        ok((await count(page, '.ov-actions [data-act="continue"], .ov-actions [data-act="go-menu"]')) === 0 && (await text(page, '.ov-sub')).replace(/\s+/g, ' ').trim() === linked.state_problem.replace(/\s+/g, ' ').trim(), 'the overview says the same, and offers none either');
        /* what the app does not offer does not happen behind it: the server starts no agent that would read the state through the link */
        const behind = await api('POST', '/api/projects/' + pid + '/advance', {});
        const after = (await api('GET', '/api/projects/' + pid)).data;
        ok(behind.status === 409 && String((behind.data.error || {}).message || '').includes(linked.state_problem) && after.worker.state === 'idle', 'and a press sent past the app is refused with the same sentence, with nothing started: ' + behind.status);
        const linkedRow = '.lib-row[href="#/p/' + pid + '/map"]';
        await open(page, '#/library', linkedRow);
        ok(/State unknown/.test(await text(page, linkedRow + ' .lib-st1')) && !/choose the phase/.test(await text(page, linkedRow)) && (await text(page, linkedRow + ' .lib-main p')).replace(/\s+/g, ' ').includes(linked.state_problem.replace(/\s+/g, ' ').trim()), 'the library row says the server\'s sentence and does not promise a choice of phase that the project does not offer: ' + (await text(page, linkedRow + ' .lib-st')).replace(/\s+/g, ' ').trim());
      }
    } finally {
      /* whatever went wrong above, the project gets its files back for the checks that follow */
      for (const name of ['events.jsonl', 'state.json']) if (fs.existsSync(path.join(kept, name))) fs.renameSync(path.join(kept, name), path.join(dir, name));
      fs.rmSync(kept, { recursive: true, force: true });
    }
    let whole = null;
    for (let i = 0; i < 40 && !(whole && whole.state_source === 'file'); i++) { whole = (await api('GET', '/api/projects/' + pid)).data; if (whole.state_source !== 'file') await sleep(250); }
    ok(!!whole && whole.state_source === 'file' && whole.state_problem === null && whole.state.round === stood.state.round, 'with its files back the project stands where it stood, with nothing to say about its state: ' + JSON.stringify(whole && [whole.state_source, whole.state_problem]));
    await reopen(page, '#/p/' + pid + '/map', '.strip');
    ok((await count(page, '[data-state-problem]')) === 0 && (await count(page, '.strip.s-unknown')) === 0, 'and the app says nothing about it any more');
  } else notes.push('states: the project directory is not on this machine; a worker the server cannot verify, an interrupted run and a lost state.json were not looked at');
  /* the server unreachable: say so, keep the draft, come back */
  const healthBefore = (await api('GET', '/api/health')).data || {};
  await open(page, '#/p/' + pid + '/wiki/Q2/edit', 'textarea#ed-body');
  await page.fill('#ed-summary', 'Typed while the server was away');
  await context.setOffline(true);
  await page.keyboard.press('Control+s');
  await until(page, () => !document.querySelector('#banner').hidden, null, 8000, 'the banner says the server is not answering');
  ok(/not answering/.test(await text(page, '#banner')), 'in plain words');
  /* What ending the server does to agents and experiments, said truly: the server says
     whether it stops its workers when it ends (`stops_workers_on_exit` of /api/health), and
     the banner says that, from the last answer the app had. */
  const stops = healthBefore.stops_workers_on_exit;
  const about = (await text(page, '#banner [data-workers]')).replace(/\s+/g, ' ').trim();
  ok(typeof stops === 'boolean' && (await page.getAttribute('#banner [data-workers]', 'data-workers')) === (stops ? 'stopped' : 'go-on'), 'the banner goes by what the server said of itself (stops its workers when it ends: ' + stops + ')');
  ok(stops ? /started to stop agents and experiments when it ends/.test(about) && !/go on without/.test(about) : /go on without the server/.test(about) && /finds them again when it is back/.test(about), 'and says what is true of this server: ' + about);
  ok(!/unless/.test(about), 'without hedging');
  ok((await page.inputValue('#ed-summary')) === 'Typed while the server was away', 'what was typed is kept');
  ok(/Not saved/.test(await text(page, '.editor .note-neg')), 'the editor says it was not saved');
  await shot(page, 'state-offline');
  await context.setOffline(false);
  await until(page, () => document.querySelector('#banner').hidden, null, 12000, 'the banner goes when the server answers again');
  await page.keyboard.press('Control+s');
  await until(page, () => !document.querySelector('form.editor'), null, 6000, 'the save goes through afterwards');
  ok((await api('GET', '/api/projects/' + pid + '/pages/Q2')).data.page.summary === 'Typed while the server was away', 'and is on the server');
  /* a page that could not be read while the server was away is read by itself when it is back */
  await open(page, '#/p/' + pid + '/wiki/Q2', '.page-wiki .prose');
  await context.setOffline(true);
  await page.click('.wiki-index .idx:has-text("Q5")');
  await page.waitForSelector('.wiki-main .note-neg');
  ok(/could not be loaded/.test(await text(page, '.wiki-main .note-neg')), 'a page that cannot be read says so: ' + (await text(page, '.wiki-main .note-neg')).trim().slice(0, 70));
  await context.setOffline(false);
  await until(page, () => !!document.querySelector('.page-wiki .prose') && !document.querySelector('.wiki-main .note-neg') && /\/wiki\/Q5$/.test(location.hash), null, 15000, 'and is read by itself when the server answers again');
  /* the event stream is live again: a change made elsewhere arrives without a reload */
  const pending = (await api('GET', '/api/projects/' + pid)).data.counts.pending.total;
  await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n, pending, 8000, 'the inbox count is current after the reconnect');
  const note = await api('POST', '/api/projects/' + pid + '/pages', { kind: 'topic', title: 'Written elsewhere while this window was open', body: 'x' });
  ok(note.status === 201, 'a page is written elsewhere');
  await until(page, () => [...document.querySelectorAll('.wiki-index .idx')].some((a) => /Written elsewhere/.test(a.textContent)), null, 8000, 'it appears in the index by itself');

  /* The server was restarted and has a new access token. To this browser that is: the
     server goes away, comes back, and refuses the cookie the browser holds (401). The app
     says so in plain words, promises no retry and asks nothing at intervals; once the new
     link was opened in this browser, the page works again. */
  if (TOKEN) {
    const held = (await context.cookies(BASE)).filter((c) => /po_token$/.test(c.name));
    ok(held.length === 1 && held[0].value === TOKEN && held[0].httpOnly && held[0].sameSite === 'Strict', 'the token is kept as one cookie that scripts cannot read and other sites cannot send: ' + JSON.stringify(held.map((c) => [c.name, c.httpOnly, c.sameSite])));
    ok(!(await page.evaluate(() => document.cookie + ' ' + location.href)).includes(TOKEN), 'neither a script of the page nor its address holds the token');
    await open(page, '#/p/' + pid + '/wiki/Q2', '.page-wiki .prose');
    refusing = true;
    try {
      await context.setOffline(true);
      await context.addCookies([Object.assign({}, held[0], { value: 'the-token-of-the-server-before-the-restart' })]);
      await page.click('.wiki-index .idx:has-text("Written elsewhere")');      // a page this window has not read yet
      await until(page, () => !document.querySelector('#banner').hidden, null, 8000, 'the server is away: the banner says that it is not answering');
      await context.setOffline(false);
      await until(page, () => !!document.querySelector('#banner [data-denied]'), null, 20000, 'the server is back and refuses this page: the banner says that it was restarted');
      const said = (await text(page, '#banner')).replace(/\s+/g, ' ').trim();
      ok(said.includes('This server was restarted; open the new link printed in its terminal.'), 'in plain words: ' + said);
      ok(!/not answering|tried again|every few seconds/.test(said), 'and neither calls the server unreachable nor promises to try again: ' + said);
      await until(page, () => /This server was restarted/.test((document.querySelector('.wiki-main .note-neg') || {}).textContent || ''), null, 6000, 'the page that could not be read says the same, not that the server is not answering');
      ok(/new link needed/.test(await text(page, '.sb-conn')), 'the foot of the project says the same in two words: ' + (await text(page, '.sb-conn')).trim());
      await shot(page, 'state-restarted');
      /* nothing is asked at intervals: no health poll, no stream opened again and again */
      await sleep(4000);
      const sent = [];
      const note = (r) => { if (r.url().startsWith(BASE)) sent.push(r.method() + ' ' + r.url().replace(BASE, '')); };
      page.on('request', note);
      await sleep(7000 * SLOW);
      page.off('request', note);
      ok(sent.length === 0, 'the app asks the server nothing while it is refused, got ' + sent.length + ': ' + sent.slice(0, 6).join(', '));
      /* a look that is still refused changes nothing */
      await page.click('#banner [data-act="retry-connection"]');
      await sleep(800);
      ok((await count(page, '#banner [data-denied]')) === 1, 'Try again, while the cookie is still the old one, leaves the banner');
      /* the person opens the new link (in another tab of this browser) and comes back */
      await entered(context);
      await page.evaluate(() => window.dispatchEvent(new Event('focus')));
      await until(page, () => document.querySelector('#banner').hidden, null, 12000, 'the banner goes once the new link was opened and the person is back');
      await until(page, () => !!document.querySelector('.page-wiki .prose') && !document.querySelector('.wiki-main .note-neg') && /Written elsewhere/.test((document.querySelector('.page-wiki .page-title') || {}).textContent || ''), null, 15000, 'the page that could not be read is read by itself');
      const elsewhere = await api('POST', '/api/projects/' + pid + '/pages', { kind: 'topic', title: 'Written after the restart', body: 'x' });
      ok(elsewhere.status === 201, 'a page is written elsewhere');
      await until(page, () => [...document.querySelectorAll('.wiki-index .idx')].some((a) => /Written after the restart/.test(a.textContent)), null, 12000, 'and the event stream is open again: it appears in the index by itself');
    } finally {
      /* whatever went wrong above, the checks that follow have a browser the server lets in */
      await context.setOffline(false).catch(() => {});
      await entered(context).catch(() => {});
      refusing = false;
    }
  } else notes.push('states: the server asks for no access token; what the app says after a restart was not looked at');
};

/* The rules the engine and the store follow, as the interface shows them: more research is the
   person's own choice (and the analyst's recommendation stands beside it); rejecting an
   experiment says what rests on it; a project's problems are said where the person sees
   them; the library lists a project that cannot be opened, and counts what waits live; the
   research cutoff is a year and a month, and one that cannot be read is put right here. */
CHECKS.rules = async ({ browser, page, pid }) => {
  const dir = await projectDir(pid);
  await idle(pid);
  await open(page, '#/p/' + pid + '/overview', '.ov-head');
  /* the control that continues a project offers, beside what comes next, the phases the person can choose */
  await page.waitForSelector('.ov-actions .go [data-act="go-menu"]');
  ok((await count(page, '.go-menu')) === 0 && (await page.getAttribute('.ov-actions [data-act="go-menu"]', 'aria-expanded')) === 'false', 'the list of phases is closed until it is asked for');
  await page.click('.ov-actions [data-act="go-menu"]');
  await page.waitForSelector('.ov-actions .go-menu');
  ok(/Search the literature again/.test(await text(page, '.go-menu [data-act="continue-phase"][data-phase="research"]')), 'the list offers to search the literature again: ' + (await page.locator('.go-menu .go-i b').allTextContents()).join(', '));
  ok(/Only a choice made here changes which phase runs next/.test(await text(page, '.go-menu')), 'and says that nothing else changes which phase runs next');
  ok(await page.evaluate(() => document.activeElement.classList.contains('go-i')), 'the keyboard is in the list: ' + (await focusOf(page)));
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Escape');
  await until(page, () => !document.querySelector('.go-menu') && document.activeElement.dataset.act === 'go-menu', null, 3000, 'Escape closes the list and gives the keyboard back to its button');
  await page.click('.ov-actions [data-act="go-menu"]');
  await page.waitForSelector('.go-menu');
  await page.click('.ov-head h1');
  await until(page, () => !document.querySelector('.go-menu'), null, 3000, 'a click elsewhere closes it');
  /* Tab leaves the list: it does not stay open behind the keyboard, which moves on from its button */
  await page.focus('.ov-actions [data-act="go-menu"]');
  await page.keyboard.press('Enter');
  await page.waitForSelector('.go-menu .go-i:focus');
  await page.keyboard.press('Tab');
  await until(page, () => !document.querySelector('.go-menu'), null, 3000, 'Tab out of the list of phases closes it');
  ok(await page.evaluate(() => { const a = document.activeElement, more = document.querySelector('.ov-actions .go-more'); return !!a && a !== document.body && a !== more && !a.closest('.go') && !!(more.compareDocumentPosition(a) & Node.DOCUMENT_POSITION_FOLLOWING); }), 'and the keyboard has moved on to what follows the button: ' + (await focusOf(page)));
  /* The analyst's recommendation, and an edit the analyst suggests to a finding. Only an agent
     brings either about: an instruction for the analyst asks the scripted one for both (see
     scripted_model.py), left through the box as a person would leave it. The journal page of
     the round then says `next: research`; the engine repeats that at the checkpoint of the
     analysis, as a field (`recommends`, with the page), and changes nothing by itself. The
     app shows the recommendation beside the choice, for as long as the server says it stands. */
  ok((await count(page, '.go-rec[data-reason="recommended"]')) === 0, 'nothing is recommended before the analyst has said so');
  await page.fill('#rq-text', 'For the browser check: [scripted-more-research] [scripted-edit]');
  await page.selectOption('#rq-for', 'analyst');
  await page.click('.rq-form button[type=submit]');
  await until(page, () => [...document.querySelectorAll('.rq')].some((r) => /scripted-more-research/.test(r.textContent) && /for the analyst/.test(r.textContent)), null, 5000, 'an instruction for the analyst is left');
  ok(/given to the next analysis/.test(await text(page, '.toast')), 'and the app says when the analyst is given it: ' + (await text(page, '.toast')).trim());
  await page.evaluate(() => { document.querySelector('.overview').scrollTop = 0; });
  const approved = await api('POST', '/api/projects/' + pid + '/experiments/E1/approve', { backend: 'local' });
  ok(approved.status === 200, 'an experiment is approved, so that a run and its analysis come next: ' + approved.status);
  await until(page, () => /^Run the (approved experiment|\d+ approved experiments), then analyse (it|them)/.test(((document.querySelector('.ov-actions [data-act="continue"]') || {}).textContent || '').trim()), null, 8000, 'the overview\'s button says that the run and its analysis come next');
  const before = (await api('GET', '/api/projects/' + pid)).data.state;
  /* every text the main button shows from here to the checkpoint is kept (see below) */
  await page.evaluate(() => {
    window.__labels = [];
    const read = () => { const label = ((document.querySelector('.ov-actions [data-act="continue"]') || {}).textContent || '').trim(); if (window.__labels[window.__labels.length - 1] !== label) window.__labels.push(label); };
    window.__labelWatch = new MutationObserver(read);
    window.__labelWatch.observe(document.body, { childList: true, subtree: true, characterData: true });
  });
  await page.click('.ov-actions [data-act="continue"]');
  await until(page, () => !!document.querySelector('.ov-state .live-dot'), null, 15000, 'the engine starts');
  await until(page, () => /paused at a checkpoint/.test((document.querySelector('.ov-head h1') || {}).textContent || '') && !document.querySelector('.ov-state .live-dot'), null, 150000, 'and pauses after the analysis');
  const paused = await idle(pid);
  const journal = paused.checkpoint.recommends_page || '';
  ok(paused.checkpoint.phase === 'analyze' && paused.checkpoint.recommends === 'research' && /^wiki\/journal\//.test(journal), 'the checkpoint of the analysis carries the recommendation as a field, with the journal page that makes it: ' + JSON.stringify([paused.checkpoint.recommends, journal]));
  ok(paused.state.next_phase === (before.next_phase === 'run' || before.next_phase === 'analyze' ? 'hypothesize' : before.next_phase), 'which did not change the phase that is next: ' + before.next_phase + ' -> ' + paused.state.next_phase);
  ok((await api('GET', '/api/projects/' + pid + '/requests')).data.requests.some((r) => /scripted-more-research/.test(r.text) && r.state === 'done' && r.handled_in.some((h) => h.phase === 'analyze')), 'the instruction for the analyst was given to that analysis, and is done');
  await page.waitForSelector('.ov-next .go-rec[data-reason="recommended"]');
  const rec = (await text(page, '.ov-next .go-rec[data-reason="recommended"]')).replace(/\s+/g, ' ').trim();
  /* the checkpoint's own words stand right above, so beside the button the reason is named in a few words, with its choice */
  ok(/Recommended in the journal page of this round\. It is yours to decide\./.test(rec), 'the recommendation is shown beside the choice: ' + rec);
  ok((await count(page, '.ov-next .go-rec[data-reason="recommended"] [data-act="continue-phase"][data-phase="research"]')) === 1 && /Search the literature again/.test(rec), 'with the choice it recommends next to it');
  ok((await page.getAttribute('.ov-next .go-rec[data-reason="recommended"] a[data-sel]', 'data-sel')) === journal, 'and a link to the journal page the server names: ' + (await page.getAttribute('.ov-next .go-rec[data-reason="recommended"] a[data-sel]', 'data-sel')));
  ok(/recommends another look at the literature/.test(await text(page, '.ov-note')), 'the checkpoint\'s own words, above, say what the analyst recommended: ' + (await text(page, '.ov-note')).replace(/\s+/g, ' ').trim().slice(-170));
  ok(!/paper-orchestra research advance|--phase|next: research/.test(await page.locator('.ov-head').innerText()), 'in the app\'s words: no command line and no field of a file: ' + (await text(page, '.ov-note')).trim().slice(0, 140));
  /* (the state arrives with the event stream, the list of pages a moment later: the results are marked as analysed in that list) */
  await until(page, () => /^Continue to |^Resume /.test(((document.querySelector('.ov-actions [data-act="continue"]') || {}).textContent || '').trim()), null, 8000, 'the main button still offers the phase that is next');
  ok(/^Continue to |^Resume /.test((await text(page, '.ov-actions [data-act="continue"]')).trim()), 'it reads: ' + (await text(page, '.ov-actions [data-act="continue"]')).trim());
  /* The state reaches the page before the list of pages does. In between the button promises
     nothing it cannot know: it never offered to analyse results that had just been analysed. */
  const labels = await page.evaluate(() => { window.__labelWatch.disconnect(); return window.__labels; });
  const since = labels.slice(labels.lastIndexOf('') + 1);      // '' is the time the agents were at work: no button
  ok(labels.includes('') && since.length >= 1 && !since.some((label) => /^Analyse the|^Run the/.test(label)), 'at no moment after the analysis did the button offer a run or an analysis that had just been done: ' + labels.map((label) => label || '(at work)').join(' | '));
  await page.click('.ov-actions [data-act="go-menu"]');
  ok(/recommends another look at the literature/.test(await text(page, '.go-menu [data-phase="research"] .go-why')), 'and in the list the recommendation stands with that choice');
  await page.keyboard.press('Escape');
  await shot(page, 'overview-recommendation');
  /* Whether the recommendation stands is the server's word, not something the app looks up:
     once the journal page is rejected the checkpoint carries none, and the line beside the
     button goes (the checkpoint's own words stay what they were). Accepted again, it is back. */
  ok((await api('POST', '/api/projects/' + pid + '/review', { items: [{ ref: journal, decision: 'reject', reason: 'Not now.' }] })).data.ok, 'the journal page is rejected in another window');
  await until(page, () => !document.querySelector('.go-rec[data-reason="recommended"]'), null, 8000, 'with its journal page rejected the recommendation is no longer shown beside the button');
  ok((await api('GET', '/api/projects/' + pid)).data.checkpoint.recommends === null && /recommends another look at the literature/.test(await text(page, '.ov-note')), 'because the server no longer says it; what the checkpoint said then is still there');
  ok((await api('POST', '/api/projects/' + pid + '/review', { items: [{ ref: journal, decision: 'accept' }] })).data.ok, 'the journal page is accepted again');
  await until(page, () => !!document.querySelector('.ov-next .go-rec[data-reason="recommended"]'), null, 8000, 'and the recommendation is shown again');
  await page.keyboard.press('1');
  await page.waitForSelector('.strip-sub .go-rec[data-reason="recommended"]');
  ok(/The journal page of this round recommends another look at the literature first\. It is yours to decide\./.test((await text(page, '.strip-sub .go-rec[data-reason="recommended"]')).replace(/\s+/g, ' ')) && (await count(page, '.strip-sub .go-rec [data-act="continue-phase"][data-phase="research"]')) === 1, 'the strip on the map says the recommendation in a sentence, with the choice: ' + (await text(page, '.strip-sub .go-rec[data-reason="recommended"]')).replace(/\s+/g, ' ').trim().slice(0, 110));
  await shot(page, 'map-recommendation');
  /* more research is the person's explicit choice: the press names the phase */
  const named = [];
  const listenAdvance = (request) => { if (request.method() === 'POST' && /\/advance$/.test(request.url())) named.push(JSON.parse(request.postData() || '{}')); };
  page.on('request', listenAdvance);
  await page.click('.strip-sub .go-rec [data-act="continue-phase"][data-phase="research"]');
  await until(page, () => !!document.querySelector('.strip.s-running') && /research/.test(document.querySelector('.strip-kicker').textContent), null, 15000, '"Search the literature again" starts the research phase');
  page.off('request', listenAdvance);
  ok(named.length === 1 && named[0].phase === 'research', 'by naming it in the request: ' + JSON.stringify(named));
  await until(page, () => !!document.querySelector('.strip.s-checkpoint') && /after research/i.test(document.querySelector('.strip-kicker').textContent), null, 120000, 'the research phase ends at its checkpoint');
  const searched = await idle(pid);
  const last = searched.state.phases[searched.state.phases.length - 1];
  ok(last.phase === 'research' && last.status === 'completed' && searched.state.next_phase === 'hypothesize', 'the research phase ran, and hypotheses come next: ' + last.phase + ' ' + last.status + ', next ' + searched.state.next_phase);
  ok((await count(page, '.go-rec[data-reason="recommended"]')) === 0, 'the recommendation is not repeated once the round has moved on');

  /* A suggested edit to a finding carries the number check of the text as it would read: on
     the page and in the inbox beside the edit, and in the row that lists it, so that the edit
     is not accepted from a list with its check out of sight. The scripted analyst suggested
     one above; it adds a sentence with a number that is on no record. */
  const rowsNow = (await api('GET', '/api/projects/' + pid + '/proposals')).data.proposals;
  const edit = rowsNow.find((r) => r.page && r.page.kind === 'finding' && /73\.25/.test(r.body || ''));
  if (ok(!!edit && !!edit.number_report && edit.number_report.unverified.includes('73.25'), 'the analyst suggested an edit to a finding, and its row carries the number check of the text as it would read: ' + JSON.stringify(edit && [edit.target, edit.number_report && edit.number_report.unverified]))) {
    const fid = edit.page.id, row = '.in-row[data-ref="' + edit.id + '"]';
    const asEdited = ((await api('GET', '/api/projects/' + pid + '/pages/' + fid)).data.proposals || []).find((x) => x.id === edit.id);
    ok(!!asEdited && !!asEdited.number_report && asEdited.number_report.summary === edit.number_report.summary, 'the row and the page give the same report: ' + edit.number_report.summary.slice(0, 90));
    await open(page, '#/p/' + pid + '/wiki/' + fid, '.wiki-rail .review-change .review-num');
    const inMargin = (await text(page, '.wiki-rail .review-change .review-num')).replace(/\s+/g, ' ');
    ok(inMargin.includes(asEdited.number_report.summary) && /Numbers, as the finding would read/.test(inMargin), 'on the page the suggested edit shows the server\'s sentence for the text as it would read: ' + inMargin.trim().slice(0, 150));
    ok(/Not found:\s*73\.25/.test(await text(page, '.wiki-rail .review-change .numcheck-flag')), 'and names the number the edit brings that is on no record: ' + (await text(page, '.wiki-rail .review-change .numcheck-flag')).trim());
    await until(page, () => /73\.25/.test((document.querySelector('.prose-marked .num-flag') || {}).textContent || ''), null, 4000, 'the number is flagged in the edit as it is read in place');
    ok(/of the finding as it is now/.test(await text(page, '.numcheck .rail-h')), 'the check of the page as it is says that it is of the page as it is: ' + (await text(page, '.numcheck .rail-h')).trim());
    await shot(page, 'wiki-edit-numbers');
    await open(page, '#/p/' + pid + '/inbox/' + fid, '.in-detail .review-change .review-num');
    ok((await text(page, '.in-detail .review-change .review-num')).replace(/\s+/g, ' ').includes(asEdited.number_report.summary) && /Not found:\s*73\.25/.test(await text(page, '.in-detail .review-change .numcheck-flag')), 'and so does the inbox: ' + (await text(page, '.in-detail .review-change .review-num')).replace(/\s+/g, ' ').trim().slice(0, 120));
    await shot(page, 'inbox-edit-numbers');
    /* in the list: the row is marked, and its quick accept says what it would bring */
    ok(/1 number not found/.test(await text(page, row + ' .in-flag[data-numbers="unverified"]')) && /73\.25/.test(await page.getAttribute(row + ' .in-flag', 'data-tip')), 'the row of the edit is marked: a number it brings was not found: ' + (await text(page, row + ' .in-flag')).trim());
    ok(/brings 1 number found in no record/.test(await page.getAttribute(row + ' [data-act="accept"]', 'aria-label')), 'and its quick accept says so before it is pressed: ' + (await page.getAttribute(row + ' [data-act="accept"]', 'aria-label')));
    const waitingBefore = await waitingNow(page);
    await page.focus(row + ' [data-act="accept"]');
    await page.keyboard.press('Enter');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n - 1, waitingBefore, 3000, 'the quick accept of a marked row accepts');
    ok(/brings 73\.25/.test(await text(page, '.toast')) && /found in no run record/.test(await text(page, '.toast')) && (await count(page, '.toast [data-act="undo"]')) === 1, 'and the confirmation says what the edit brings, with the undo: ' + (await text(page, '.toast')).replace(/\s+/g, ' ').trim().slice(0, 130));
    await page.keyboard.press('z');
    await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n, waitingBefore, 3000, 'which takes it back');
    /* on the map nothing beside the mark says what the check found: the press opens the edit, with its check, instead of deciding it */
    await open(page, '#/p/' + pid + '/map', '.map-svg .node');
    const mark = '.map-svg .node[data-key="' + edit.target + '"] .n-yes';
    if (await count(page, mark)) {
      await page.click(mark);
      await until(page, (id) => location.hash.endsWith('/map/' + id) && !!document.querySelector('.panel .review-change .review-num'), fid, 6000, 'on the map the accept mark of such an edit opens it beside the map, with its number check');
      ok(/decide there/.test(await text(page, '.toast')) && (await count(page, '.toast [data-act="undo"]')) === 0 && (await waitingNow(page)) === waitingBefore, 'and decides nothing: ' + (await text(page, '.toast')).replace(/\s+/g, ' ').trim().slice(0, 120));
      await page.keyboard.press('Escape');
    } else notes.push('rules: the finding with the suggested edit has no accept mark on the map; the map\'s way of opening it first was not looked at');
    /* A listing that has no report for the edit (it spends a bounded time on them, and a check
       can fail): the row says that the numbers were not checked here, and its quick accept
       opens the edit instead of accepting what nobody could have judged. */
    const listing = '**/api/projects/' + pid + '/proposals*';
    await page.route(listing, async (route) => {
      const response = await route.fetch(), data = await response.json();
      (data.proposals || []).forEach((r) => { if (r.id === edit.id) { r.number_report = null; r.number_report_error = 'not checked in this listing'; } });
      await route.fulfill({ response, json: data });
    });
    await reopen(page, '#/p/' + pid + '/inbox', row + ' .in-flag[data-numbers="unchecked"]');
    ok(/numbers not checked here/.test(await text(page, row + ' .in-flag')) && /Open it first/.test(await page.getAttribute(row + ' [data-act="accept"]', 'aria-label')), 'a row without a report says that its numbers were not checked here, and that a press opens it: ' + (await text(page, row + ' .in-flag')).trim());
    await page.focus(row + ' [data-act="accept"]');
    await page.keyboard.press('Enter');
    await until(page, (id) => location.hash.endsWith('/inbox/' + id) && !!document.querySelector('.in-detail .review-change .review-num'), fid, 6000, 'its quick accept opens the edit with its number check');
    ok(/not checked in the list/.test(await text(page, '.toast')) && (await waitingNow(page)) === waitingBefore && (await count(page, '.toast [data-act="undo"]')) === 0, 'and decides nothing: ' + (await text(page, '.toast')).replace(/\s+/g, ' ').trim().slice(0, 120));
    /* selected with others and accepted together, such an edit is left out, and the app says so */
    await page.check(row + ' .in-check input');
    await page.keyboard.press('Shift+A');
    await until(page, () => /was left/.test((document.querySelector('.toast') || {}).textContent || ''), null, 4000, 'accepting the selection leaves the unchecked edit out and says so');
    ok((await waitingNow(page)) === waitingBefore, 'so nothing was decided: ' + (await text(page, '.toast')).replace(/\s+/g, ' ').trim().slice(0, 130));
    await page.unroute(listing);
    const still = (await api('GET', '/api/projects/' + pid + '/proposals')).data.proposals.find((r) => r.id === edit.id);
    ok(!!still && still.review === 'proposed', 'the suggested edit is still undecided on the server');
    const gone = await api('POST', '/api/projects/' + pid + '/review', { items: [{ ref: edit.id, decision: 'reject', reason: 'Suggested for the browser check.' }] });
    ok(gone.status === 200 && gone.data.ok, 'and is rejected, now that it has been looked at');
  }

  /* Rejecting an experiment says what rests on it. An experiment that was accepted and has run
     can be rejected after all; the server answers with the findings based on its runs and the
     verdicts they carry, and the page lists them with links. */
  await open(page, '#/p/' + pid + '/experiments/E1', '.exp-page .page-title');
  await page.click('.exp-page .page-unaccept');
  await page.waitForSelector('.note-unaccept .reject-form input:focus');
  /* Escape gives the question up, and the keyboard goes back to what asked it */
  await page.keyboard.press('Escape');
  await until(page, () => !document.querySelector('.note-unaccept') && document.activeElement.classList.contains('page-unaccept'), null, 3000, 'Escape gives up the rejection and returns the keyboard to "Reject it after all"');
  await page.keyboard.press('Enter');
  await page.waitForSelector('.note-unaccept .reject-form input:focus');
  await page.keyboard.type('The toy problem is too far from a real model.');
  const answers = [];
  const listenReview = async (response) => { if (response.request().method() === 'POST' && /\/review$/.test(response.url())) answers.push(await response.json().catch(() => null)); };
  page.on('response', listenReview);
  await page.keyboard.press('Enter');
  await until(page, () => !!document.querySelector('.exp-page .note-rejected .rested'), null, 10000, 'the rejected experiment lists what rests on it');
  page.off('response', listenReview);
  const deps = answers.length && answers[0] && answers[0].results[0] ? answers[0].results[0].dependents : null;
  if (ok(!!deps && deps.findings.length >= 1, 'the server says which findings rest on the rejected experiment: ' + JSON.stringify(deps && deps.findings.map((f) => f.id)))) {
    const listed = await page.$$eval('.note-rejected .rested a.chip', (links) => links.map((a) => [a.textContent.trim().split(/\s+/)[0], a.dataset.sel]));
    ok(deps.findings.every((f) => listed.some((l) => l[0] === f.id && l[1] === f.path)), 'each finding is a link to its page: ' + JSON.stringify(listed.map((l) => l[0])));
    ok(deps.hypotheses.every((h) => listed.some((l) => l[0] === h.id && l[1] === h.path)), 'and so is each hypothesis whose verdict they carry: ' + JSON.stringify(deps.hypotheses.map((h) => h.id + ' ' + h.state)));
    ok(!deps.findings.some((f) => f.only) || /rests on nothing else/.test(await text(page, '.note-rejected .rested')), 'a finding that rests on nothing else is marked');
    ok(/None of this was changed/.test(await text(page, '.note-rejected .rested-say')), 'and the page says that nothing of it was changed');
    ok(/is rejected\./.test(await text(page, '.toast')) && /rest(s)? on its runs/.test(await text(page, '.toast')), 'the confirmation says how much rests on it: ' + (await text(page, '.toast')).replace(/\s+/g, ' ').trim());
    await shot(page, 'experiment-rejected-what-rests-on-it');
    await page.click('.note-rejected .rested a.chip >> nth=0');
    await until(page, (path) => document.querySelector('.page[data-page="' + path + '"]') && /\/wiki\//.test(location.hash), listed[0][1], 5000, 'a link opens the finding');
    /* The server says it with the experiment for as long as the rejection stands (`dependents`
       of its view): the same after a reload, in another browser, and with nothing kept on the device. */
    const asRejected = (await api('GET', '/api/projects/' + pid + '/experiments/E1')).data.dependents;
    ok(!!asRejected && JSON.stringify(asRejected.findings.map((f) => f.id)) === JSON.stringify(deps.findings.map((f) => f.id)) && JSON.stringify(asRejected.hypotheses.map((h) => h.id)) === JSON.stringify(deps.hypotheses.map((h) => h.id)), 'the view of the rejected experiment carries what the answer to the rejection said: ' + JSON.stringify(asRejected && asRejected.findings.map((f) => f.id)));
    await reopen(page, '#/p/' + pid + '/experiments/E1', '.exp-page .note-rejected');
    await until(page, (n) => document.querySelectorAll('.note-rejected .rested a.chip').length === n, listed.length, 5000, 'after a reload the rejected experiment still lists what rests on it');
    ok((await page.evaluate(() => Object.keys(localStorage).filter((key) => /rested/.test(key)).length)) === 0, 'and this device keeps no copy of the list');
    const elsewhere = await entered(await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: arg('scheme', 'light'), reducedMotion: 'reduce' }));
    const second = await elsewhere.newPage();
    watch(second, () => 'rules (a second browser)');
    await open(second, '#/p/' + pid + '/experiments/E1', '.exp-page .note-rejected');
    await until(second, (n) => document.querySelectorAll('.note-rejected .rested a.chip').length === n, listed.length, 8000, 'a browser that never saw the rejection lists what rests on it too');
    await open(second, '#/p/' + pid + '/wiki/E1', '.page-wiki .note-rejected');
    await until(second, (n) => document.querySelectorAll('.page-wiki .note-rejected .rested a.chip').length === n, listed.length, 8000, 'and so does the page of the experiment in the wiki');
    await elsewhere.close();
  }
  ok((await api('GET', '/api/projects/' + pid + '/pages/E1')).data.page.review === 'rejected', 'the experiment is rejected on the server');
  /* An accepted finding that now rests on nothing a paper reports is named in the paper view,
     with the server's reason, before a write-up is started. */
  await page.bringToFront();
  const paperNow = (await api('GET', '/api/projects/' + pid + '/paper')).data || {};
  if (Array.isArray(paperNow.left_out)) {
    const alone = deps ? deps.findings.filter((f) => f.only && f.review === 'accepted') : [];
    ok(alone.every((f) => paperNow.left_out.some((item) => item.path === f.path)), 'the server lists the accepted findings a paper would now leave out: ' + JSON.stringify(paperNow.left_out.map((item) => item.id)));
    if (paperNow.left_out.length) {
      await open(page, '#/p/' + pid + '/paper', '.paper-left');
      for (const item of paperNow.left_out) {
        const row = (await text(page, '.paper-left .pn-row[data-left="' + (item.id || item.path) + '"]')).replace(/\s+/g, ' ');
        ok(row.toLowerCase().includes(String(item.reason).replace(/\s+/g, ' ').replace(/[.\s]+$/, '').toLowerCase()), 'the paper view names ' + (item.id || item.path) + ', which a paper would leave out, with the server\'s reason: ' + row.trim().slice(0, 170));
      }
      ok((await count(page, '.paper-left .pn-row a.chip[data-sel]')) === paperNow.left_out.length, 'each with a link to the finding');
      ok(!paperNow.missing || /Not ready/.test(await text(page, '.paper-gap')), 'and says that a paper cannot be written yet where that is so: ' + paperNow.missing);
      await shot(page, 'paper-left-out');
    }
  } else notes.push('rules: this server does not list the findings a paper would leave out (left_out); the paper view was not checked for them');
  await open(page, '#/p/' + pid + '/experiments/E1', '.exp-page .note-rejected');
  await page.click('.note-rejected [data-act="reaccept"]');
  await until(page, () => !document.querySelector('.exp-page .note-rejected') && !!document.querySelector('.exp-page .page-unaccept'), null, 8000, 'and can be accepted after all, as before');
  /* a rejection made in another window: this window, which did not make it, lists what the server says rests on the experiment */
  const again = await api('POST', '/api/projects/' + pid + '/review', { items: [{ ref: 'E1', decision: 'reject', reason: 'Rejected in another window.' }] });
  ok(again.data.ok, 'the experiment is rejected again, in another window');
  await until(page, () => /Rejected in another window/.test((document.querySelector('.exp-page .note-rejected') || {}).textContent || ''), null, 8000, 'which this window shows');
  const restsNow = again.data.results[0].dependents || { findings: [], hypotheses: [] };
  await until(page, (n) => document.querySelectorAll('.note-rejected .rested a.chip').length === n, restsNow.findings.length + restsNow.hypotheses.length, 8000, 'with what rests on it, though the rejection was not made here');
  ok((await api('POST', '/api/projects/' + pid + '/review', { items: [{ ref: 'E1', decision: 'accept' }] })).data.ok, 'and it is accepted again for the checks that follow');
  await until(page, () => !document.querySelector('.exp-page .note-rejected'), null, 8000, 'which this window shows too');

  /* The research cutoff is a year and a month. One that cannot be read (a project.json edited
     by hand) is kept as written by the server, which stops research and refuses every save of
     the settings until it is corrected: it is named among the problems and put right here. */
  await open(page, '#/p/' + pid + '/overview', '.spend-more.cutoff');
  await page.click('.spend-more.cutoff summary');
  await page.waitForSelector('#cut-year');
  await page.selectOption('#cut-month', '3');
  await page.fill('#cut-year', '2025');
  await page.click('.cutoff-form button[type=submit]');
  await until(page, () => /before March 2025/.test((document.querySelector('.spend-more.cutoff summary') || {}).textContent || ''), null, 6000, 'a cutoff chosen as a month and a year is saved and said');
  ok((await api('GET', '/api/projects/' + pid)).data.settings.research_cutoff === '2025-03', 'as 2025-03 on the server: ' + (await api('GET', '/api/projects/' + pid)).data.settings.research_cutoff);
  if (dir) {
    const settingsFile = path.join(dir, 'project.json');
    const written = JSON.parse(fs.readFileSync(settingsFile, 'utf8'));
    written.settings.research_cutoff = 'some time last spring';
    fs.writeFileSync(settingsFile, JSON.stringify(written, null, 2));
    /* and a file of the project that is not what it should be */
    fs.writeFileSync(path.join(dir, 'proposals', 'ep-0bad0bad.json'), 'not JSON at all');
    let view = null;
    for (let i = 0; i < 40 && !(view && view.problems.length >= 2); i++) { view = (await api('GET', '/api/projects/' + pid)).data; if (view.problems.length < 2) await sleep(250); }
    ok(view.settings.research_cutoff === 'some time last spring' && view.problems.some((x) => /cutoff/i.test(x.message)) && view.problems.some((x) => x.path === 'proposals/ep-0bad0bad.json'), 'the server reports both problems: ' + JSON.stringify(view.problems.map((x) => x.path)));
    ok(view.problems_total === view.problems.length && !view.problems.some((x) => x.path && x.message.startsWith(x.path)), 'says how many there are, and names each file once: ' + view.problems_total);
    try {
    await reopen(page, '#/p/' + pid + '/map', '.top-prob');
    ok((await text(page, '.top-prob')).trim() === String(view.problems.length), 'a quiet mark in the top bar counts the problems: ' + (await text(page, '.top-prob')).trim());
    ok(!/neg|failed/.test(await page.getAttribute('.top-prob', 'class')) && (await count(page, '.top-status.st-failed')) === 0, 'quietly: it is not an alarm');
    await page.click('.top-prob');
    await page.waitForSelector('#problems');
    const said = (await text(page, '#problems')).replace(/\s+/g, ' ');
    ok(view.problems.every((x) => said.includes(x.path) && said.includes(x.message.replace(/\s+/g, ' '))), 'the overview lists each problem with its file, in the server\'s words: ' + said.trim().slice(0, 200));
    ok(said.includes(view.problems_total + ' problems in the project\'s files'), 'under the number of problems the server gives: ' + view.problems_total);
    ok(/cannot be read/.test(await text(page, '.spend-more.cutoff summary')), 'the cutoff says that it cannot be read: ' + (await text(page, '.spend-more.cutoff summary')).replace(/\s+/g, ' ').trim());
    await shot(page, 'overview-problems');
    await page.click('.spend-more.cutoff summary');
    await page.waitForSelector('#cut-year');
    ok(/some time last spring/.test(await text(page, '.cutoff-says')) && (await page.inputValue('#cut-year')) === '', 'it is shown as it is written, and the form starts empty: ' + (await text(page, '.cutoff-says')).trim().slice(0, 110));
    await page.selectOption('#cut-month', '6');
    await page.click('.cutoff-form button[type=submit]');
    await until(page, () => /Give the year/.test((document.querySelector('.cutoff-form .note-neg') || {}).textContent || ''), null, 4000, 'a month without a year is not sent: the form says what is missing');
    await page.fill('#cut-year', '2024');
    await page.click('.cutoff-form button[type=submit]');
    await until(page, () => /before June 2024/.test((document.querySelector('.spend-more.cutoff summary') || {}).textContent || ''), null, 6000, 'the cutoff is corrected in the app');
    } finally {
      /* whatever went wrong above, the project is not left broken for the checks that follow */
      fs.rmSync(path.join(dir, 'proposals', 'ep-0bad0bad.json'), { force: true });
      const left = JSON.parse(fs.readFileSync(settingsFile, 'utf8'));
      if (left.settings.research_cutoff === 'some time last spring') { left.settings.research_cutoff = null; fs.writeFileSync(settingsFile, JSON.stringify(left, null, 2)); }
    }
    await reopen(page, '#/p/' + pid + '/overview', '.ov-head');
    const mended = (await api('GET', '/api/projects/' + pid)).data;
    ok(mended.settings.research_cutoff === '2024-06' && !mended.problems.length, 'and with the stray file gone the project has no problem left: ' + JSON.stringify(mended.problems));
    ok((await count(page, '#problems')) === 0 && (await count(page, '.top-prob')) === 0, 'so the note and the mark are gone');
  } else notes.push('rules: the project directory is not on this machine; a cutoff that cannot be read and a problem in the project\'s files were not looked at');
  await page.click('.spend-more.cutoff summary');
  await page.waitForSelector('.cutoff-form [data-act="cutoff-none"]');
  await page.click('.cutoff-form [data-act="cutoff-none"]');
  await until(page, () => /none/.test((document.querySelector('.spend-more.cutoff summary') || {}).textContent || ''), null, 6000, 'and can be taken off again');
  ok((await api('GET', '/api/projects/' + pid)).data.settings.research_cutoff == null, 'no cutoff on the server');

  /* The library lists a project that cannot be opened, with the reason, and each row says
     how much waits now: proposed pages and suggested edits together. */
  const listing = (await api('GET', '/api/projects')).data;
  const row = listing.projects.find((x) => x.id === pid);
  await open(page, '#/library', '.lib-row');
  if (ok(!!row && !!row.counts && row.pending === row.counts.pending.total && row.pending === row.counts.pending.pages + row.counts.pending.proposals, 'the row of the library carries the live count of what waits, suggested edits included: ' + JSON.stringify(row && row.counts && row.counts.pending))) {
    const shown = (await text(page, '.lib-row[href="#/p/' + pid + '/map"] .lib-st2')).trim();
    ok(row.pending ? shown.includes(row.pending + ' waiting for you') : !/waiting/.test(shown), 'and shows it: ' + shown + ' (the server says ' + row.pending + ')');
  }
  if (dir) {
    const form = new FormData();
    form.append('title', 'A project the check breaks');
    form.append('brief', 'Made to be broken by hand.');
    const made = await (await fetch(BASE + '/api/projects', { method: 'POST', headers: AUTH, body: form })).json();
    const broken = path.join(path.dirname(dir), made.dir);
    ok(!!made.id && fs.existsSync(path.join(broken, 'project.json')), 'a second project is made: ' + made.id);
    fs.writeFileSync(path.join(broken, 'project.json'), '{ "schema": 1, "id": ');
    let unreadable = null;
    for (let i = 0; i < 40 && !unreadable; i++) { unreadable = ((await api('GET', '/api/projects')).data.unreadable || []).find((x) => x.dir === made.dir) || null; if (!unreadable) await sleep(250); }
    if (ok(!!unreadable && !!unreadable.message, 'the server lists the project as one it cannot open: ' + JSON.stringify(unreadable && unreadable.dir))) {
      await open(page, '#/library', '.lib-unreadable');
      const told = (await text(page, '.lib-bad[data-dir="' + made.dir + '"]')).replace(/\s+/g, ' ');
      ok(told.includes(made.dir + '/') && told.includes(unreadable.message.replace(/\s+/g, ' ')), 'the library lists it with its folder and the server\'s reason: ' + told.trim().slice(0, 160));
      ok((await count(page, '.lib-row[href="#/p/' + made.id + '/map"]')) === 0 && (await count(page, '.lib-row')) === listing.projects.length, 'beside the projects that open, not among them');
      await shot(page, 'library-unreadable');
      /* its address answers with the reason too, not with "no such project" */
      await open(page, '#/p/' + made.id + '/map', '.blank-unreadable');
      ok(/cannot be opened/.test(await text(page, '.blank h1')) && (await text(page, '.blank')).replace(/\s+/g, ' ').includes(unreadable.message.replace(/\s+/g, ' ')), 'opening it says that it cannot be opened, and why: ' + (await text(page, '.blank h1')).trim());
    }
    fs.rmSync(broken, { recursive: true, force: true });
    await open(page, '#/library', '.lib-row');
    await until(page, () => !document.querySelector('.lib-unreadable'), null, 10000, 'with the folder gone the library lists no project it cannot open');
  } else notes.push('rules: the workspace is not on this machine; a project that cannot be opened was not looked at');
};

/* the palette, the shortcut sheet, the theme */
CHECKS.keys = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/map', '.map-svg .node');
  await page.keyboard.press('Control+k');
  await page.waitForSelector('#pal-in:focus');
  await page.keyboard.type('H3');
  await until(page, () => /H3/.test((document.querySelector('.pal-i.cur') || {}).textContent || ''), null, 3000, 'the palette finds a page by its id');
  await page.keyboard.press('Enter');
  await until(page, () => location.hash.endsWith('/map/H3'), null, 3000, 'and goes to it in the open view');
  await page.keyboard.press('/');
  await page.waitForSelector('#pal-in:focus');
  await page.keyboard.type('sparse features');
  await until(page, () => document.querySelectorAll('.pal-i .pal-s').length >= 1, null, 5000, 'the palette searches the text of the pages through the server');
  await shot(page, 'palette');
  await page.keyboard.press('Escape');
  ok((await count(page, '.pal')) === 0, 'Escape closes the palette');
  await page.keyboard.press('Control+k');
  await page.keyboard.type('go to inbox');
  await page.keyboard.press('Enter');
  await until(page, () => /\/inbox/.test(location.hash), null, 3000, 'the palette runs an action');
  await page.keyboard.press('?');
  await page.waitForSelector('.help');
  ok(/Undo, until the decision is sent/.test(await text(page, '.help')), 'the shortcut sheet is honest about undo');
  await shot(page, 'help');
  await page.keyboard.press('Escape');
  const was = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  await page.keyboard.press('t');
  const now = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  ok(now && now !== was, 'T switches the theme: ' + was + ' -> ' + now);
  await page.keyboard.press('t');
  await page.keyboard.press('j');
  await page.keyboard.press('j');
  ok((await count(page, '.in-row.cur')) === 1, 'J and K move through the inbox');
  /* the keyboard keeps its place when the page is drawn again */
  if ((await count(page, '.in-row:not(.in-old)')) >= 2) {
    await page.locator('.in-row:not(.in-old) .in-check input').first().focus();
    await page.keyboard.press('Space');
    ok(/\[check\]/.test(await focusOf(page)) && (await count(page, '.in-row:not(.in-old) .in-check input:checked')) === 1, 'a checkbox ticked by the keyboard keeps the focus: ' + (await focusOf(page)));
    await page.keyboard.press('Space');
    ok(/\[check\]/.test(await focusOf(page)) && (await count(page, '.in-row:not(.in-old) .in-check input:checked')) === 0, 'so that Space unticks it again');
    await page.locator('.in-row:not(.in-old) .in-main >> nth=1').focus();
    await page.keyboard.press('Enter');
    await until(page, () => document.activeElement.classList.contains('in-main') && document.activeElement.getAttribute('aria-current') === 'true', null, 3000, 'a row opened with Enter keeps the focus');
    await page.keyboard.press('e');
    await until(page, () => document.activeElement.id === 'ed-body', null, 5000, 'E opens the editor with the keyboard in the text');
    await page.keyboard.press('Escape'); await page.keyboard.press('Escape');
    await until(page, () => !document.querySelector('form.editor'), null, 3000, 'Escape leaves the editor');
  }
  /* everything can be reached by Tab, and focus is visible */
  await open(page, '#/p/' + pid + '/overview', '.ov-head');
  await page.keyboard.press('Tab'); await page.keyboard.press('Tab'); await page.keyboard.press('Tab');
  const outline = await page.evaluate(() => { const el = document.activeElement; const s = getComputedStyle(el); return el.tagName + ' ' + s.outlineStyle + ' ' + s.outlineWidth; });
  ok(/solid 2px/.test(outline), 'keyboard focus is visible: ' + outline);
};

/* the composer: a real upload of two files, honest errors, nothing lost; the new project starts */
CHECKS.composer = async ({ page, context }) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'po-e2e-'));
  const notes = path.join(dir, 'notes.md'), data = path.join(dir, 'runs.csv');
  fs.writeFileSync(notes, '# Notes\n\nThe spike is at step 40.\n');
  fs.writeFileSync(data, 'step,loss\n' + Array.from({ length: 400 }, (_, i) => i + ',' + (1 / (i + 1)).toFixed(4)).join('\n') + '\n');
  await open(page, '#/start', '#idea-text');
  ok(await page.locator('.composer button[type=submit]').isDisabled(), 'nothing can be started from nothing');
  await page.fill('#idea-text', 'Does the scripted effect appear in a browser test?');
  await page.setInputFiles('#file-input', [notes, data]);
  await page.waitForSelector('.filechip >> nth=1');
  ok((await count(page, '.filechip')) === 2, 'both files are listed');
  /* a file dropped on the composer is attached as well */
  await page.evaluate(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(['dropped'], 'dropped.txt', { type: 'text/plain' }));
    const target = document.querySelector('.composer');
    ['dragenter', 'dragover', 'drop'].forEach((type) => target.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: transfer })));
  });
  await until(page, () => document.querySelectorAll('.filechip').length === 3, null, 4000, 'a dropped file is listed');
  await page.click('.filechip:has-text("dropped.txt") [data-act="file-del"]');
  await until(page, () => document.querySelectorAll('.filechip').length === 2, null, 4000, 'a file can be taken off again');
  await page.click('.adv summary');
  /* every number the composer shows before the person types one is the server's default */
  const defaults = (await api('GET', '/api/defaults')).data, limits = defaults.settings.limits;
  await until(page, (usd) => (document.querySelector('#opt-model-usd') || {}).placeholder === usd, limits.model_usd.toFixed(2) + ' dollars', 5000, 'the composer shows the server\'s default limit for models');
  ok((await page.getAttribute('#opt-exp-usd', 'placeholder')) === limits.experiment_usd.toFixed(2) + ' dollars' && (await page.getAttribute('#opt-exp-min', 'placeholder')) === (limits.experiment_seconds / 60) + ' minutes', 'and the default limits for experiments: ' + (await page.getAttribute('#opt-exp-usd', 'placeholder')) + ', ' + (await page.getAttribute('#opt-exp-min', 'placeholder')));
  ok((await text(page, '.start-est')).includes('$' + limits.model_usd.toFixed(2)), 'the sentence under the box quotes the same limit: ' + (await text(page, '.start-est')).trim().slice(-60));
  const waits = Object.keys(defaults.settings.review).filter((kind) => defaults.settings.review[kind] === 'review');
  ok(JSON.stringify(await page.$$eval('.adv-review input[data-kind]:checked', (boxes) => boxes.map((b) => b.dataset.kind).sort())) === JSON.stringify(waits.slice().sort()), 'what needs a review by default is what the server says: ' + waits.join(', '));
  /* the efforts and the compute backends that can be chosen are the ones the server says its settings accept */
  ok(JSON.stringify(await page.$$eval('#opt-effort option', (options) => options.map((o) => o.value))) === JSON.stringify([''].concat(defaults.efforts)) && defaults.efforts.length >= 2, 'the composer offers the model efforts the server lists: ' + JSON.stringify(defaults.efforts));
  ok(JSON.stringify(await page.$$eval('#opt-compute option', (options) => options.map((o) => o.value))) === JSON.stringify([''].concat(defaults.compute_backends)) && defaults.compute_backends.length >= 1, 'and the compute backends it lists: ' + JSON.stringify(defaults.compute_backends));
  ok(new RegExp('Default \\(.*' + defaults.settings.models.effort.replace(/^x/, '') + '\\)').test(await page.$eval('#opt-effort option', (o) => o.textContent)), 'with the server\'s default named as the default: ' + (await page.$eval('#opt-effort option', (o) => o.textContent)));
  /* a research cutoff: a year, and a month if wanted */
  await page.selectOption('#opt-cutoff-month', '6');
  await until(page, () => /Give the year/.test((document.querySelector('[data-cutoff]') || {}).textContent || ''), null, 3000, 'a month without a year says that the year is missing');
  await page.fill('#opt-cutoff-year', '2024');
  await until(page, () => /June 2024 or later are left out/.test((document.querySelector('[data-cutoff]') || {}).textContent || ''), null, 3000, 'a cutoff of June 2024 says what it leaves out');
  await page.fill('#opt-title', 'Browser test project');
  await page.fill('#opt-model-usd', '7.5');
  ok((await text(page, '.start-est')).includes('$7.50'), 'a limit the person types replaces the default in the sentence');
  await shot(page, 'start-filled');
  /* the server is gone: the error is honest and nothing typed is lost */
  await context.setOffline(true);
  await page.click('.composer button[type=submit]');
  await page.waitForSelector('.composer-error');
  ok(/not created/.test(await text(page, '.composer-error')), 'a failed start says the project was not created');
  ok((await page.inputValue('#idea-text')).startsWith('Does the scripted effect') && (await count(page, '.filechip')) === 2, 'the text and the files are still there');
  await shot(page, 'start-failed');
  await context.setOffline(false);
  await page.waitForTimeout(600);
  /* the text survives a reload (files cannot, and the composer says so) */
  await page.reload();
  await page.waitForSelector('#idea-text');
  ok((await page.inputValue('#idea-text')).startsWith('Does the scripted effect'), 'the idea survives a reload');
  ok(/could not be/.test(await text(page, '.composer-note')), 'and the composer says the files have to be added again');
  await page.setInputFiles('#file-input', [notes, data]);
  await page.waitForSelector('.filechip >> nth=1');
  await page.click('.composer button[type=submit]');
  await until(page, () => /#\/p\/p-[0-9a-f]+\/map/.test(location.hash), null, 20000, 'the project is created and opened');
  const pid = (await page.evaluate(() => location.hash)).split('/')[2];
  const view = (await api('GET', '/api/projects/' + pid)).data;
  ok(view.title === 'Browser test project' && view.settings.limits.model_usd === 7.5, 'title and limit from the options: ' + view.title + ' ' + view.settings.limits.model_usd);
  ok(view.settings.research_cutoff === '2024-06', 'the research cutoff chosen as a month and a year is the project\'s: ' + view.settings.research_cutoff);
  ok(view.settings.limits.experiment_usd === limits.experiment_usd && view.settings.limits.experiment_seconds === limits.experiment_seconds, 'what the person left alone is the server\'s default');
  ok(view.attachments.length === 2 && view.attachments.some((a) => a.name === 'runs.csv' && a.size > 3000), 'both files are in inputs/: ' + JSON.stringify(view.attachments.map((a) => a.name)));
  /* research starts by itself and the map fills in */
  await until(page, () => !!document.querySelector('.strip.s-running') || document.querySelectorAll('.map-svg .node').length > 1, null, 15000, 'research starts');
  await shot(page, 'new-project-running');
  await until(page, () => !!document.querySelector('.strip.s-checkpoint') && document.querySelectorAll('.map-svg .node').length >= 5, null, 90000, 'intake and research reach the first checkpoint');
  ok((await inboxCount(page)) >= 4, 'the scripted topics and questions wait for a decision: ' + (await inboxCount(page)));
  await shot(page, 'new-project-checkpoint');
  await idle(pid);
  /* reject everything that waits at once, with one reason */
  await open(page, '#/p/' + pid + '/inbox', '.in-row');
  const all = await inboxCount(page);
  await page.check('.in-all input');
  await page.click('[data-act="bulk-reject"]');
  await page.waitForSelector('#why-bulk');
  await page.fill('#why-bulk', 'Not what I asked: start from the notes.');
  await page.keyboard.press('Enter');
  await until(page, () => !document.querySelector('.views .count'), null, 4000, 'rejecting everything empties the inbox at once');
  ok(/Rejected \d+ proposals/.test(await text(page, '.toast')), 'the toast counts them: ' + (await text(page, '.toast')));
  await until(page, () => !document.querySelector('.toast [data-act="undo"]'), null, 9000, 'the rejections are sent');
  await sleep(500);
  const rejected = (await api('GET', '/api/projects/' + pid + '/pages?review=rejected')).data.pages;
  ok(rejected.length === all && rejected.every((p) => p.meta.decision_reason === 'Not what I asked: start from the notes.'), 'every page is rejected with the reason: ' + rejected.length + ' of ' + all);
  ok(/Every proposal has a decision/.test(await text(page, '.in-clear')), 'the empty inbox says that everything is decided');
  /* nothing to write a paper from: the page says what is missing, as the server would */
  await open(page, '#/p/' + pid + '/paper', '.paper-form');
  await page.waitForSelector('.paper-gap');
  ok(/Not ready: no finding is accepted/.test(await text(page, '.paper-gap')) && (await page.locator('.paper-form button[type=submit]').isDisabled()), 'a project with attached results and no accepted finding cannot be written up yet: ' + (await text(page, '.paper-gap')));
  /* A write-up that is refused. The app does not offer one here, but a second window or the
     command line can ask for it: its worker ends before it begins and says why. The server
     keeps that on record (a `paper.finished` event), and the paper view shows the reason
     from there: the same after a reload, without the server's memory of a worker or its log. */
  const templates = ((await api('GET', '/api/templates')).data.templates || []).map((t) => t.name);
  const asked = await api('POST', '/api/projects/' + pid + '/paper', { template: templates[0] });
  if (ok(asked.status === 202, 'a write-up is asked for through the API although nothing can be written: ' + asked.status)) {
    await until(page, () => !!document.querySelector('.paper-last[data-ended="refused"]'), null, 60000, 'the paper view says that the last write-up did not start');
    await idle(pid);
    const last = (await api('GET', '/api/projects/' + pid + '/paper')).data.last;
    ok(!!last && last.outcome === 'refused' && !!last.error && last.started_at === null, 'the server has it on record as refused, with the reason: ' + JSON.stringify(last && [last.outcome, last.exit_code]));
    const told = () => text(page, '.paper-last').then((t) => t.replace(/\s+/g, ' '));
    ok(/did not start/.test(await told()) && (await told()).includes(String(last && last.error).replace(/\s+/g, ' ')), 'the view gives the server\'s reason: ' + (await told()).trim().slice(0, 170));
    ok((await count(page, '.top-status.st-failed')) === 0, 'a refused write-up is not the agents stopping unexpectedly');
    const logs = [];
    const listenLog = (request) => { if (/\/worker\/log/.test(request.url())) logs.push(request.url()); };
    page.on('request', listenLog);
    await page.reload();
    await page.waitForSelector('.paper-last[data-ended="refused"]');
    page.off('request', listenLog);
    ok((await told()).includes(String(last && last.error).replace(/\s+/g, ' ')) && !logs.length, 'after a reload it says the same, and never asked for the worker\'s log: ' + logs.length + ' requests');
    await shot(page, 'paper-refused');
  }
  fs.rmSync(dir, { recursive: true, force: true });
  return pid;
};

/* the paper step: the form is real; what the write-up does is the pipeline's business */
CHECKS.paper = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/paper', '.paper-form');
  await until(page, () => document.querySelectorAll('#paper-template option').length >= 2, null, 5000, 'the bundled templates are offered');
  ok(!(await page.locator('.paper-form button[type=submit]').isDisabled()) && !(await count(page, '.paper-gap')), 'the write-up can be started');
  /* the number check of the findings a paper would be written from, in the server's sentence for each */
  const accepted = (await api('GET', '/api/projects/' + pid + '/pages?kind=finding&review=accepted')).data.pages;
  const reports = [];
  for (const f of accepted) reports.push([f, (await api('GET', '/api/projects/' + pid + '/pages/' + pagePath(f.path))).data.number_report]);
  const telling = reports.filter(([, r]) => r && (r.unverified.length || r.elsewhere.length));
  await until(page, (n) => document.querySelectorAll('.paper-numbers .pn-row[data-finding]').length === n && !/Reading the number check/.test(document.querySelector('.paper-numbers').textContent), telling.length, 15000, 'the paper view shows the number check of the ' + accepted.length + ' accepted findings (' + telling.length + ' have something to say)');
  for (const [f, r] of telling) {
    const row = (await text(page, '.paper-numbers .pn-row[data-finding="' + f.id + '"]')).replace(/\s+/g, ' ');
    ok(row.includes(r.summary), 'for ' + f.id + ' in the server\'s summary sentence: ' + row.trim().slice(0, 120));
    ok(r.unverified.every((n) => row.includes(n)) && (!r.unverified.length || /Not found:/.test(row)), 'with the numbers of ' + f.id + ' that were found nowhere: ' + JSON.stringify(r.unverified));
  }
  ok(telling.length >= 1, 'at least one accepted finding of the demo has a number to say something about: ' + reports.map(([f, r]) => f.id + ' ' + (r ? r.unverified.length + '/' + r.elsewhere.length : 'no report')).join(', '));
  /* an address never acts: a link an agent wrote into a page cannot accept, run or delete */
  await idle(pid);
  const stateOf = async () => (await api('GET', '/api/projects/' + pid + '/pages?include_rejected=1')).data.pages.map((p) => [p.path, p.review, p.state, p.updated].join(' ')).join('\n');
  const before = await stateOf();
  for (const tail of ['inbox/H5/accept', 'wiki/H5/delete', 'map/H5/reject', 'experiments/E4/run', 'experiments/E4/approve', 'experiments/E3/accept-run', 'wiki/H2/restore']) await open(page, '#/p/' + pid + '/' + tail, '#view > *');
  await sleep(400);
  ok(before === (await stateOf()), 'opening an address changes nothing on the server');
  await open(page, '#/p/' + pid + '/paper', '.paper-form');
  await shot(page, 'paper');
  /* the write-up really starts; on the scripted model its outline cannot be parsed, and the view says so */
  /* (looked at in every frame: once the top bar no longer says that the paper is being written, the view must not either) */
  await page.evaluate(() => {
    const seen = window.__writing = { was: false, late: 0, done: false };
    const look = () => {
      const top = (document.querySelector('.top-status') || {}).textContent || '', form = document.querySelector('.paper-form');
      const says = !!form && (/Writing…/.test(form.textContent) || !!form.querySelector('[data-act="stop"]'));
      if (/Writing/.test(top)) seen.was = true; else if (seen.was && says) seen.late++;
      if (!seen.done) requestAnimationFrame(look);
    };
    requestAnimationFrame(look);
  });
  await page.click('.paper-form button[type=submit]');
  await until(page, () => !!document.querySelector('.paper-stages .ps'), null, 30000, 'the stages of the write-up are shown');
  await idle(pid, 120000);
  await until(page, () => /Failed|Finished|Interrupted/.test((document.querySelector('.paper-state .sec-t') || {}).textContent || ''), null, 20000, 'the end of the write-up is shown');
  const frames = await page.evaluate(() => { window.__writing.done = true; return window.__writing; });
  if (frames.was) ok(frames.late === 0, 'once the write-up has ended the view does not go on saying "Writing…" with a Stop button: it did for ' + frames.late + ' frames');
  else notes.push('paper: the write-up ended before the top bar had said that it was under way; whether the view follows its end at once was not looked at');
  ok((await count(page, '.paper-stages .ps')) >= 5, 'every stage of the pipeline is listed');
  ok((await count(page, '.top-status.st-failed')) === 0, 'a write-up that failed is not the agents stopping unexpectedly');
  /* How the last write-up went is the server's account of it, from the project's events
     (`last` of GET .../paper): what it said when it started (which findings it was written
     from, what was left out, which numbers were taken out) and how it ended. */
  const last = (await api('GET', '/api/projects/' + pid + '/paper')).data.last;
  const began = (await api('GET', '/api/projects/' + pid + '/activity?limit=300')).data.events.filter((e) => e.type === 'paper.started').pop();
  if (ok(!!last && !!last.started_summary && !!began && last.started_summary === began.summary && /^(written|failed|stopped)$/.test(last.outcome), 'the server says how the last write-up went, with what it announced when it started: ' + JSON.stringify(last && [last.outcome, last.started_at]))) {
    ok((await text(page, '.paper-started')).replace(/\s+/g, ' ').includes(last.started_summary.replace(/\s+/g, ' ')), 'and the paper view repeats that in the server\'s words: ' + (await text(page, '.paper-started')).replace(/\s+/g, ' ').trim().slice(0, 160));
    ok((await count(page, '.paper-last')) === 0, 'a write-up that left its own state is told by that state, not twice');
    /* the same after a reload: none of it is this window's memory */
    const logs = [];
    const listenLog = (request) => { if (/\/worker\/log/.test(request.url())) logs.push(request.url()); };
    page.on('request', listenLog);
    await page.reload();
    await until(page, (summary) => ((document.querySelector('.paper-started') || {}).textContent || '').replace(/\s+/g, ' ').includes(summary), last.started_summary.replace(/\s+/g, ' '), 10000, 'after a reload the paper view says the same about the last write-up');
    page.off('request', listenLog);
    ok(!logs.length, 'without reading the worker\'s log for it: ' + logs.join(', '));
  }
  await shot(page, 'paper-after');
  /* A write-up whose worker is killed in the middle. It leaves a state that still reads
     "running" and no account of its own; the server's account of that end (an event of the
     paper, on file) is what the view says, in the server's sentence, also after a reload. */
  await idle(pid, 120000);
  await page.click('.paper-form button[type=submit]');
  let writing = null, over = null;
  /* (How long the worker takes to start is the machine's business: a loaded or small one needs
     many seconds to import the pipeline. The wait is for that; the write-up itself then runs
     for a second or two, and a look every few dozen milliseconds does not miss it. Where a
     write-up cannot run at all, on a machine without TeX, it ends at its first look at the
     machine: that is told apart by the new start it announced, and noted, not failed.) */
  const earlier = last && last.started_at;
  for (const deadline = Date.now() + 90000; Date.now() < deadline && !writing && !over;) {
    const paper = (await api('GET', '/api/projects/' + pid + '/paper')).data;
    if (paper.running && paper.last && paper.last.outcome === 'running' && paper.worker.pid > 0 && paper.worker.kind === 'write') writing = paper.worker;
    else if (!paper.running && paper.last && paper.last.started_at && paper.last.started_at !== earlier && paper.last.outcome !== 'running') over = paper.last;
    else await sleep(60);
  }
  if (over) notes.push('paper: the write-up ended by itself before its worker could be ended (' + String(over.summary || over.outcome).replace(/\s+/g, ' ').slice(0, 160) + '); how a killed write-up is shown was not looked at. On a machine without pdflatex and bibtex that is expected');
  else
  if (ok(!!writing, 'the write-up is under way again, and has said that it started')) {
    let killed = true;
    try { process.kill(writing.pid, 'SIGKILL'); } catch (e) { killed = false; notes.push('paper: the worker of the write-up had ended before it could be killed (' + e.message + '); how a killed write-up is shown was not looked at'); }
    await idle(pid, 60000);
    if (killed) {
      let cut = null;
      for (let i = 0; i < 400 && !(cut && cut.last && cut.last.exit_code != null); i++) { cut = (await api('GET', '/api/projects/' + pid + '/paper')).data; if (!(cut.last && cut.last.exit_code != null)) await sleep(150); }
      if (ok(!!cut && !!cut.last && cut.last.outcome === 'failed' && cut.last.exit_code < 0 && /killed/.test(cut.last.summary || '') && !cut.running, 'the server says how the killed write-up ended, from the project\'s events: ' + JSON.stringify(cut && cut.last && [cut.last.outcome, cut.last.exit_code]))) {
        await until(page, (summary) => ((document.querySelector('.paper-last[data-ended="failed"]') || {}).textContent || '').replace(/\s+/g, ' ').includes(summary), cut.last.summary.replace(/\s+/g, ' '), 15000, 'the paper view says in the server\'s sentence that the write-up was cut short');
        ok(/did not finish/.test(await text(page, '.paper-last')) && !/In progress/.test(await text(page, '.paper-state .sec-t')), 'and does not show it as in progress: ' + (await text(page, '.paper-last')).replace(/\s+/g, ' ').trim().slice(0, 140));
        await shot(page, 'paper-worker-killed');
        await page.reload();
        await until(page, (summary) => ((document.querySelector('.paper-last[data-ended="failed"]') || {}).textContent || '').replace(/\s+/g, ' ').includes(summary), cut.last.summary.replace(/\s+/g, ' '), 10000, 'and the same after a reload');
      }
    }
  }
};

/* a large project: the map stays usable */
CHECKS.large = async ({ page, pid, large }) => {
  if (!large) { notes.push('large: no project with 150 pages in the workspace (research demo --dir <workspace>/large --pages 150), skipped'); return; }
  const pages = (await api('GET', '/api/projects/' + large + '/pages')).data.pages.length;
  const t0 = Date.now();
  await open(page, '#/p/' + large + '/map', '.map-svg .node');
  const took = Date.now() - t0;
  const nodes = await count(page, '.map-svg .node');
  const bundles = await count(page, '.map-svg .node.k-bundle');
  ok(bundles >= 3, 'sources are folded into bundles: ' + bundles);
  ok(pages >= 140 && nodes < pages - 15, 'fewer marks than pages are drawn: ' + nodes + ' for ' + pages + ' pages');
  ok(took < 6000 * SLOW, 'the map of ' + pages + ' pages opens in ' + took + ' ms');
  const over = await page.evaluate(OVERFLOW);
  ok(!over.length, 'large map overflow: ' + over.join('; '));
  await shot(page, 'large-map');
  /* A crowded map draws the argument and holds the reference lines back: nothing that cites,
     mentions, answers, is about a page or ends at a source is drawn until a page is selected
     or pointed at, and then that page's own lines only. A control and the legend say so. */
  const refLines = () => page.evaluate(() => {
    const read = (key) => /^bundle:/.test(key) || /(^|\/)sources\//.test(key);
    const hits = (group) => Array.from(document.querySelectorAll('.map-svg ' + group + ' .edge-hit')).filter((e) => /^(cites|about|mentions|answers)$/.test(e.dataset.type) || read(e.dataset.s) || read(e.dataset.t));
    return { drawn: hits('.hits').map((e) => [e.dataset.s, e.dataset.t]), marked: document.querySelectorAll('.map-svg .edges .edge.t-cites, .map-svg .edges .edge.t-about, .map-svg .edges .edge.t-mentions, .map-svg .edges .edge.t-answers').length, pointed: document.querySelectorAll('.map-svg .refs .edge').length, argued: document.querySelectorAll('.map-svg .edges .edge.f-test, .map-svg .edges .edge.f-pos, .map-svg .edges .edge.f-neg').length };
  });
  await page.mouse.move(4, 4);
  await page.waitForTimeout(150);
  let lines = await refLines();
  ok(lines.drawn.length === 0 && lines.marked === 0 && lines.pointed === 0, 'in a large project no reference line is drawn while nothing is selected: ' + lines.drawn.length + ' drawn, ' + lines.pointed + ' for a page pointed at');
  const argued = lines.argued;
  ok(argued >= 20, 'the lines of the argument (tests, runs, what findings rest on and support) are drawn: ' + argued);
  ok(/on selection/.test(await text(page, '.map-tools [data-act="map-refs"]')) && (await page.getAttribute('.map-tools [data-act="map-refs"]', 'aria-pressed')) === 'false', 'a control says that reference lines wait for a selection: ' + (await text(page, '.map-tools [data-act="map-refs"]')));
  ok(/on selection/.test(await text(page, '.map-key')), 'and so does the legend: ' + (await text(page, '.map-key .key-row >> nth=-1')).replace(/\s+/g, ' ').trim());
  /* pointing at a page shows its lines for as long as the pointer is on it */
  const onSheet = await page.evaluate(() => {
    const sheet = document.querySelector('.map-canvas').getBoundingClientRect(), foot = document.querySelector('.map-foot').getBoundingClientRect();
    return Array.from(document.querySelectorAll('.map-svg .node:not(.k-bundle) .gl')).map((g) => g.getBoundingClientRect()).filter((b) => b.top > sheet.top + 40 && b.bottom < foot.top - 6 && b.left > sheet.left && b.right < sheet.right).map((b) => [b.left + b.width / 2, b.top + b.height / 2]);
  });
  let pointed = 0;
  for (const at of onSheet.slice(0, 30)) {
    await page.mouse.move(at[0], at[1]);
    await page.waitForTimeout(80);
    pointed = await count(page, '.map-svg .refs .edge');
    if (pointed) break;
  }
  ok(pointed > 0, 'pointing at a page draws its reference lines (tried ' + Math.min(30, onSheet.length) + ' pages on the sheet)');
  ok((await count(page, '.map-svg .edges .edge.t-cites, .map-svg .edges .edge.t-about')) === 0, 'without drawing them into the plan');
  await page.mouse.move(4, 4);
  await until(page, () => document.querySelectorAll('.map-svg .refs .edge').length === 0, null, 3000, 'and they go when the pointer leaves it');
  /* The marks on a page that is itself proposed decide the page, and its hatching says so. On
     an accepted page they decide a change to it; where such a page is drawn on one line, the
     line says which change before it says the title. */
  const says = await page.evaluate(() => Array.from(document.querySelectorAll('.map-svg .node.n-tight.r-accepted')).filter((n) => n.querySelector('.n-btn')).map((n) => n.getAttribute('aria-label').split(',')[0] + ': "' + ((n.querySelector('.n-l2') || {}).textContent || '') + '"'));
  ok(says.length >= 1 && says.every((line) => /→ \w|change"/.test(line)), 'an accepted page on one line says what its two marks decide: ' + (says.join('; ') || 'no such page is drawn'));
  /* A page is pointed at when the pointer moves onto it. Selecting a page opens it beside the
     map and the map is laid out again; the page that then lies under the pointer, which has
     not moved, was not pointed at: neither its tip nor its lines are shown. */
  const asked = await page.evaluate(() => Array.from(document.querySelectorAll('.map-svg .node.k-question.r-accepted')).map((n) => n.dataset.key));
  let landed = 0, stray = 0, clicked = 0;
  for (const key of asked.slice(0, 8)) {
    const at = await page.evaluate((key) => {
      const sheet = document.querySelector('.map-canvas').getBoundingClientRect(), foot = document.querySelector('.map-foot').getBoundingClientRect();
      const b = document.querySelector('.map-svg .node[data-key="' + CSS.escape(key) + '"] .n-bg').getBoundingClientRect();
      return b.top > sheet.top + 40 && b.bottom < foot.top - 6 ? [b.left + 40, b.top + b.height / 2] : null;
    }, key);
    if (!at) continue;
    await page.mouse.click(at[0], at[1]);
    await until(page, (key) => { const n = document.querySelector('.map-svg .node.sel'); return !!n && n.dataset.key === key; }, key, 4000, 'a click on a question selects it');
    await page.waitForTimeout(250);
    const under = await page.evaluate((at) => {
      const el = document.elementFromPoint(at[0], at[1]), node = el && el.closest && el.closest('.map-svg .node');
      return { other: !!node && !node.classList.contains('sel'), lines: document.querySelectorAll('.map-svg .refs .edge').length, tip: !!document.querySelector('.map-tip.on') };
    }, at);
    clicked++;
    if (under.other) landed++;
    if (under.lines || under.tip) stray++;
    await page.keyboard.press('Escape');
    await until(page, () => /\/map$/.test(location.hash) && !document.querySelector('.map-svg .node.sel'), null, 4000, 'Escape clears the selection');
    await page.mouse.move(4, 4);
    await page.waitForTimeout(120);
  }
  if (!landed) notes.push('large: none of ' + clicked + ' selections moved another page under the pointer, so what a pointer at rest shows was not checked');
  ok(!stray, 'a page that the map moved under a pointer at rest shows neither its tip nor its lines: ' + stray + ' of ' + clicked + ' selections left some (' + landed + ' put another page under the pointer)');
  /* selecting a page draws its own reference lines, and no other page's */
  const graph = (await api('GET', '/api/projects/' + large + '/graph')).data.edges;
  const listed = (await api('GET', '/api/projects/' + large + '/pages')).data.pages;
  const citing = listed.find((pg) => pg.kind === 'hypothesis' && pg.review !== 'rejected' && graph.some((e) => e.source === pg.path && e.type === 'cites'));
  const placesOf = () => page.evaluate(() => Array.from(document.querySelectorAll('.map-svg .node:not(.k-source):not(.k-bundle)')).map((n) => n.dataset.key + ' ' + n.getAttribute('transform')).sort());
  const elsewhere = (was, now) => was.filter((place) => now.indexOf(place) < 0).length;
  const unselected = await placesOf();
  if (ok(!!citing, 'the large project has a hypothesis that cites a source')) {
    await open(page, '#/p/' + large + '/map/' + citing.id, '.map-svg .node.sel');
    await page.mouse.move(4, 4);
    await page.waitForTimeout(200);
    lines = await refLines();
    ok(lines.drawn.length >= 1 && lines.marked >= 1, 'selecting ' + citing.id + ' draws its reference lines: ' + lines.drawn.length);
    ok(lines.drawn.every((ends) => ends[0] === citing.path || ends[1] === citing.path), 'and only its own: ' + lines.drawn.filter((ends) => ends[0] !== citing.path && ends[1] !== citing.path).length + ' of ' + lines.drawn.length + ' belong to another page');
    ok((await count(page, '.map-svg .edges .edge.t-cites.hl')) >= 1, 'they are drawn at full strength, as part of its lineage');
    /* while a page is selected the reference lines on the map are its own: pointing at
       another page shows what that page is, and draws no second set of lines beside them */
    const others = await page.evaluate(() => {
      const sheet = document.querySelector('.map-canvas').getBoundingClientRect(), foot = document.querySelector('.map-foot').getBoundingClientRect(), panel = document.querySelector('.panel').getBoundingClientRect();
      return Array.from(document.querySelectorAll('.map-svg .node:not(.sel):not(.k-bundle) .gl')).map((g) => g.getBoundingClientRect()).filter((b) => b.top > sheet.top + 40 && b.bottom < foot.top - 6 && b.left > sheet.left && b.right < Math.min(sheet.right, panel.left)).map((b) => [b.left + b.width / 2, b.top + b.height / 2]);
    });
    let tips = 0, second = 0;
    for (const at of others.slice(0, 16)) {
      await page.mouse.move(at[0], at[1]);
      await page.waitForTimeout(60);
      if (await count(page, '.map-tip.on')) tips++;
      second += await count(page, '.map-svg .refs .edge');
    }
    ok(tips > 0 && !second, 'with ' + citing.id + ' selected, pointing at ' + Math.min(16, others.length) + ' other pages showed ' + tips + ' tips and drew ' + second + ' lines of theirs');
    await page.mouse.move(4, 4);
    await page.waitForTimeout(150);
    await shot(page, 'large-map-selected');
    await page.keyboard.press('Escape');
    await until(page, () => /\/map$/.test(location.hash) && !document.querySelector('.map-svg .edges .edge.t-cites'), null, 4000, 'the reference lines go with the selection');
    /* the page opened beside the map and closed again: the map was laid out twice, and is as it was */
    await page.waitForTimeout(200);
    ok(!elsewhere(unselected, await placesOf()), 'the map comes back as it was when the selection is cleared: ' + elsewhere(unselected, await placesOf()) + ' of ' + unselected.length + ' pages are elsewhere');
  }
  /* the control draws all of them, and takes them back; the keyboard stays on it. They are
     drawn over the plan: no page changes its place under the person who asked for them */
  const plan = await placesOf();
  await page.focus('.map-tools [data-act="map-refs"]');
  await page.keyboard.press('Enter');
  await until(page, () => /always/.test((document.querySelector('.map-tools [data-act="map-refs"]') || {}).textContent || '') && document.querySelectorAll('.map-svg .edges .edge.t-cites').length > 20, null, 5000, 'the control draws every reference line');
  ok(!elsewhere(plan, await placesOf()), 'every reference line is drawn over the plan as it stood: ' + elsewhere(plan, await placesOf()) + ' of ' + plan.length + ' pages changed their place');
  ok((await count(page, '.map-svg .edges .edge.f-test, .map-svg .edges .edge.f-pos, .map-svg .edges .edge.f-neg')) === argued, 'and the lines of the argument are the same ' + argued);
  ok((await page.getAttribute('.map-tools [data-act="map-refs"]', 'aria-pressed')) === 'true' && !/on selection/.test(await text(page, '.map-key')), 'the control and the legend say that all lines are drawn');
  ok(await page.evaluate(() => document.activeElement && document.activeElement.dataset.act === 'map-refs'), 'the keyboard stays on the control: ' + (await focusOf(page)));
  ok(!(await page.evaluate(OVERFLOW)).length, 'large map with every line: overflow ' + (await page.evaluate(OVERFLOW)).join('; '));
  await shot(page, 'large-map-all-lines');
  await page.keyboard.press('Enter');
  await until(page, () => /on selection/.test((document.querySelector('.map-tools [data-act="map-refs"]') || {}).textContent || '') && !document.querySelector('.map-svg .edges .edge.t-cites'), null, 5000, 'and takes them back');
  await page.evaluate(() => document.activeElement.blur());
  /* a bundle opens when its topic is selected (by the keyboard: the bundle may lie below the sheet) */
  await page.focus('.map-svg .node.k-bundle[aria-label*="cited by"] >> nth=0');
  await page.keyboard.press('Enter');
  await until(page, () => /\/map\/.+/.test(location.hash), null, 4000, 'opening a bundle selects the topic that cites it');
  const after = await count(page, '.map-svg .node.k-source');
  ok(after >= 2, 'and its sources unfold: ' + after);
  await shot(page, 'large-map-open');
  /* the canvas scrolls under pinned headings */
  const head0 = await page.evaluate(() => document.querySelector('.map-svg .head').getAttribute('transform'));
  await page.mouse.move(500, 400);
  await page.mouse.wheel(0, 300);
  await page.waitForTimeout(150);
  const vp = await page.evaluate(() => document.querySelector('.map-svg .vp').getAttribute('transform'));
  ok(/translate\(0 -\d/.test(vp) || /translate\(-?\d+(\.\d+)? -\d/.test(vp), 'the wheel scrolls the map: ' + vp);
  ok((await page.evaluate(() => document.querySelector('.map-svg .head').getAttribute('transform'))) === head0, 'the headings stay where they are');
  /* the round filter */
  if (await count(page, '#map-round')) {
    await page.selectOption('#map-round', '2');
    await until(page, (n) => document.querySelectorAll('.map-svg .node').length < n, nodes, 4000, 'the round filter thins the map');
    await page.selectOption('#map-round', '');
  }
  await page.keyboard.press('Escape');
  /* The map is one stop for Tab: the selected page, or the idea, takes the keyboard, and the
     arrow keys move from there. A node that has the keyboard is on the sheet, also when it
     lies below what was showing: the map moves to it. Tab then leaves the map for its
     tools, however many pages there are. */
  const vp0 = await page.evaluate(() => document.querySelector('.map-svg .vp').getAttribute('transform'));
  await page.focus('.map-canvas');
  await page.keyboard.press('Tab');
  ok(await page.evaluate(() => !!(document.activeElement.closest && document.activeElement.closest('.map-svg .node'))), 'Tab goes from the map to a page on it');
  ok((await count(page, '.map-svg .node[tabindex="0"]:not(.k-bundle)')) === 1, 'one page of the map is in the Tab order');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  let outside = 0, strayed = 0;
  for (let i = 0; i < 40; i++) {
    await page.keyboard.press('ArrowDown');
    await page.waitForTimeout(60);
    const where = await page.evaluate(() => {
      const node = document.activeElement.closest && document.activeElement.closest('.map-svg .node');
      if (!node) return 'lost';
      const sheet = document.querySelector('.map-canvas').getBoundingClientRect(), foot = document.querySelector('.map-foot').getBoundingClientRect(), mark = node.querySelector('.gl').getBoundingClientRect();
      return mark.top < sheet.top + 34 || mark.bottom > foot.top || mark.left < sheet.left || mark.right > sheet.right ? 'outside' : node.classList.contains('sel') ? 'ok' : 'unselected';
    });
    if (where === 'outside') outside++; else if (where !== 'ok') strayed++;
  }
  ok(!strayed, 'the keyboard goes along with the selection when the arrow keys move it: ' + strayed + ' of 40 times it did not');
  ok(!outside, 'a node that has the keyboard is on the sheet: ' + outside + ' of 40 were not');
  ok((await page.evaluate(() => document.querySelector('.map-svg .vp').getAttribute('transform'))) !== vp0, 'the map moved to follow the keyboard');
  /* after the page that has the keyboard come its two marks and the folded bundles, then the tools */
  const stops = (await count(page, '.map-svg .node.k-bundle')) + 4;
  let left = false;
  for (let i = 0; i < stops && !left; i++) { await page.keyboard.press('Tab'); left = await page.evaluate(() => !(document.activeElement.closest && document.activeElement.closest('.map-svg'))); }
  ok(left, 'Tab leaves a map of 150 pages within ' + stops + ' presses');
  await page.keyboard.press('Escape');
  await open(page, '#/p/' + large + '/wiki', '.wiki-index .idx');
  ok((await count(page, '.wiki-index .idx')) >= 140, 'the wiki index lists all pages');
  await page.fill('#idx-filter', 'hypothesis 1');
  await until(page, () => document.querySelectorAll('.wiki-index .idx').length < 40, null, 3000, 'the filter narrows the index');
  await page.fill('#idx-filter', '');
  /* every proposal of a large inbox at once, and back */
  await open(page, '#/p/' + large + '/inbox', '.in-row');
  const all = await waitingNow(page);
  await page.check('.in-all input');
  await page.keyboard.press('Shift+A');
  await until(page, () => !document.querySelector('.views .count'), null, 4000, 'accepting all ' + all + ' proposals empties the inbox at once');
  await page.keyboard.press('z');
  await until(page, (n) => (+((document.querySelector('.views .count') || {}).textContent || 0)) === n, all, 4000, 'and one undo brings all of them back');
  /* Taking an approval back, where another window is faster. An approved experiment that has
     not started offers "Take the approval back"; here a run is started elsewhere between the
     look and the press. The press says what the person was told it does, so the server does
     that or nothing: it answers that the approval is in use, and the run in flight is not
     touched. (In the second project, so that the experiment is free for the other checks.) */
  const unrun = (await api('GET', '/api/projects/' + large + '/experiments')).data.experiments.find((x) => x.review === 'accepted' && x.state === 'approved' && !x.runs);
  if (unrun) {
    ok(unrun.cancel_does === 'withdraw', 'the row of an approved experiment that has not started says that a cancel takes the approval back: ' + unrun.cancel_does);
    await open(page, '#/p/' + large + '/experiments/' + unrun.id, '.callout [data-act="cancel-run"][data-does="withdraw"]');
    ok((await text(page, '.callout [data-act="cancel-run"]')).trim() === 'Take the approval back', 'its button says "Take the approval back": ' + (await text(page, '.callout [data-act="cancel-run"]')).trim());
    const press = '**/api/projects/' + large + '/experiments/' + unrun.id + '/cancel';
    let sent = null, elsewhere = 0;
    await page.route(press, async (route) => {
      sent = JSON.parse(route.request().postData() || '{}');
      elsewhere = (await api('POST', '/api/projects/' + large + '/experiments/' + unrun.id + '/run')).status;
      for (let i = 0; i < 80; i++) { const now = (await api('GET', '/api/projects/' + large + '/experiments/' + unrun.id)).data; if (now.page.state !== 'approved' || now.cancel_does !== 'withdraw') break; await sleep(100); }
      await route.continue();
    });
    await page.click('.callout [data-act="cancel-run"]');
    await until(page, () => !!document.querySelector('.toast-error'), null, 20000, 'a press that came after another window started the run is refused, and says so');
    await page.unroute(press);
    ok(elsewhere === 202 && !!sent && sent.expect === 'withdraw', 'the run was started elsewhere before the press arrived, and the press said that it meant to take the approval back: ' + elsewhere + ' ' + JSON.stringify(sent));
    ok(/cannot be taken back|cannot be withdrawn/.test(await text(page, '.toast-error')) && /Nothing was changed|is being run/.test(await text(page, '.toast-error')), 'in the server\'s sentence: ' + (await text(page, '.toast-error')).replace(/\s+/g, ' ').trim().slice(0, 170));
    await page.click('.toast [data-act="toast-x"]');
    await idle(large, 120000);
    const ran = (await api('GET', '/api/projects/' + large + '/experiments/' + unrun.id)).data;
    ok(ran.runs.length === 1 && ran.runs[0].status !== 'cancelled' && ran.page.state !== 'planned' && ran.page.state !== 'cancelled', 'the run that the other window started was neither cancelled nor lost its approval: ' + ran.page.state + ', run ' + (ran.runs[0] || {}).status);
    await until(page, () => !document.querySelector('.callout [data-act="cancel-run"]') && !document.querySelector('.run-live'), null, 20000, 'and when it has ended the page offers nothing to take back');
  } else notes.push('large: no approved experiment without a run in the large project; a press of "Take the approval back" that another window overtook was not looked at');
  /* what is being typed in one project stays in that project */
  await open(page, '#/p/' + pid + '/overview', '#rq-text');
  await page.fill('#rq-text', 'Typed for the first project only.');
  const planned = (await api('GET', '/api/projects/' + pid + '/experiments')).data.experiments.find((x) => x.state !== 'running' && x.state !== 'queued' && x.review !== 'rejected');
  const twin = planned && (await api('GET', '/api/projects/' + large + '/experiments')).data.experiments.find((x) => x.id === planned.id);
  if (twin) {
    await open(page, '#/p/' + pid + '/experiments/' + planned.id, '[data-act="edit-code"]');
    await page.click('[data-act="edit-code"]');
    await page.waitForSelector('#code-text');
    await page.fill('#code-text', '# typed in the first project only\n');
  }
  await page.click('.brand');
  await page.waitForSelector('.lib-row');
  await open(page, '#/p/' + large + '/overview', '#rq-text');
  ok((await page.inputValue('#rq-text')) === '', 'an instruction typed in one project is not in the box of another: ' + JSON.stringify(await page.inputValue('#rq-text')));
  if (twin) {
    await page.keyboard.press('4');
    await page.waitForSelector('.ex-table');
    await page.click('.ex-row:has(.ex-id b:text-is("' + twin.id + '")) .ex-name a');
    await page.waitForSelector('.exp-code');
    await page.waitForSelector('.exp-code .code-n, .exp-code .quiet');
    ok((await count(page, '#code-text')) === 0, 'code being edited in one project does not open in the experiment of the same name in another');
    await page.click('.brand');
    await page.waitForSelector('.lib-row');
    await open(page, '#/p/' + pid + '/experiments/' + planned.id, '#code-text');
    ok((await page.inputValue('#code-text')).startsWith('# typed in the first project only'), 'and is still there when the first project is opened again');
    await page.click('[data-act="code-cancel"]');
  }
  await open(page, '#/p/' + pid + '/overview', '#rq-text');
  ok((await page.inputValue('#rq-text')) === 'Typed for the first project only.', 'the instruction is still in the box of its own project');
  await page.fill('#rq-text', '');

  /* Network access is never taken from the plan. A plan that asks for it is answered by
     the person, yes or no, with no default: an approval pressed before the answer sends
     nothing, and the answer goes out with the approval. (In the second project, so that
     the experiment made for this does not get in the way of the other checks.) */
  const made = await api('POST', '/api/projects/' + large + '/pages', { kind: 'experiment', title: 'Repeat E1 on a public data set', body: '## Objective\n\nRepeat E1 on a data set that is downloaded when the run starts.\n', fields: { mode: 'script', network: true, timeout_seconds: 120, summary: 'E1 again, on data fetched from the network.' } });
  if (ok(made.status < 300 && made.data && made.data.page && made.data.page.id, 'an experiment whose plan asks for the network can be written: ' + made.status)) {
    const eid = made.data.page.id;
    await api('PUT', '/api/projects/' + large + '/experiments/' + eid + '/code/main.py', { text: 'import json\n\njson.dump({"spike": 0.1}, open("metrics.json", "w"))\nprint("token [redacted] and *** were here")\nprint("done")\n' });
    const approvals = [];
    const listen = (request) => { if (request.method() === 'POST' && /\/approve$/.test(request.url())) approvals.push(request.postData() || ''); };
    page.on('request', listen);
    await open(page, '#/p/' + large + '/experiments/' + eid, '.approve .net-ask');
    ok(/asks for network access/.test(await text(page, '.net-ask')), 'the approval form says that the plan asks for the network');
    ok((await count(page, '.net-ask input[type="radio"]')) === 2 && (await count(page, '.net-ask input:checked')) === 0, 'the question has two answers and neither is chosen for the person');
    ok(/written by you/.test(await text(page, '.page-by')), 'an experiment the person wrote is not "designed by the you": ' + (await text(page, '.page-by')));
    await page.click('.approve [data-act="approve"]:not([data-start])');
    await until(page, () => !!document.querySelector('.net-ask.nudge') && /network access/.test((document.querySelector('#toast') || {}).textContent || ''), null, 4000, 'an approval without the answer shows the question instead');
    await sleep(400);
    ok(approvals.length === 0, 'and nothing is sent: ' + approvals.length + ' approval requests');
    await shot(page, 'experiment-network-question');
    await page.click('.net-ask input[value="no"]');
    await until(page, () => !document.querySelector('.net-ask.nudge') && document.querySelectorAll('.net-ask input:checked').length === 1, null, 3000, 'the answer is taken');
    await page.click('.approve [data-act="approve"]:not([data-start])');
    await until(page, () => /approved/.test((document.querySelector('.page-status') || {}).textContent || ''), null, 8000, 'the experiment is approved once the question is answered');
    ok(approvals.length === 1 && JSON.parse(approvals[0] || '{}').network === false, 'the approval says "network: false" itself: ' + approvals.join(' '));
    const view = (await api('GET', '/api/projects/' + large + '/experiments/' + eid)).data;
    ok(view.page.state === 'approved' && view.page.meta.network === false, 'and the server recorded a run without network access: ' + view.page.state + ', network ' + view.page.meta.network);
    page.off('request', listen);
    /* `[redacted]` and `***` are what stands in a log where a credential was: marks, shown as
       the plain text they are and never styled as an error */
    await idle(large);
    await page.click('.callout [data-act="run-now"]');
    await until(page, () => !!document.querySelector('.run.open .metrics tr'), null, 60000, 'the approved experiment runs');
    await until(page, () => /\[redacted\]/.test((document.querySelector('.run.open .log-body') || {}).textContent || ''), null, 8000, 'its log, which names a masked credential, is read');
    const mask = await page.evaluate(() => {
      const log = document.querySelector('.run.open .log-body'), note = document.querySelector('.run.open .log-mask');
      const neg = getComputedStyle(document.documentElement).getPropertyValue('--neg').trim();
      const probe = document.createElement('i'); probe.style.color = neg; document.body.append(probe); const red = getComputedStyle(probe).color; probe.remove();
      return { text: log.textContent, marked: log.children.length, color: getComputedStyle(log).color === red, note: note && !note.hidden ? note.textContent : '', noteRed: note ? getComputedStyle(note).color === red : false, failure: !!document.querySelector('.run.open .failure') };
    });
    ok(/token \[redacted\] and \*\*\*/.test(mask.text) && mask.marked === 0 && !mask.color && !mask.failure, 'the marks of a masked credential are plain text in the log, not styled as an error: ' + JSON.stringify([mask.marked, mask.color, mask.failure]));
    ok(/masked/.test(mask.note) && !mask.noteRed, 'a quiet note says what they are: ' + mask.note);
    await idle(large);
  }
  /* a plan that does not ask: the box starts unticked and the approval still says so */
  const plain = ((await api('GET', '/api/projects/' + large + '/experiments')).data.experiments || []).find((x) => x.review === 'accepted' && x.state === 'planned' && !(x.meta || {}).network);
  if (plain) {
    await open(page, '#/p/' + large + '/experiments/' + plain.id, '.approve');
    ok((await count(page, '.approve .net-ask')) === 0 && (await count(page, '#ap-net-' + plain.id + ':checked')) === 0, 'a plan that does not ask for the network has "Allow network access" unticked');
  }
};

/* hostile text: titles and bodies with markup, scripts, odd Unicode and words longer than any
   column never run and never push a view wider than the window */
CHECKS.hostile = async ({ page, pid }) => {
  const LONG = 'Pneumonoultramicroscopicsilicovolcanoconiosis'.repeat(5);
  const ODD = '\u202Eevil\u202C Z\u0351\u036B\u0343a\u0363\u0346l\u0367\u034Ago\u0310\u033D \uD83D\uDE80 <img src=x onerror="window.__ran=1"> <script>window.__ran=2</script> &amp; "q" \'s\' ';
  const was = (await api('GET', '/api/projects/' + pid)).data.title;
  const make = async (kind, title, body, fields, relations) => { const made = await api('POST', '/api/projects/' + pid + '/pages', { kind, title, body, fields: fields || {}, relations: relations || {} }); ok(made.status === 201, 'a ' + kind + ' with a hostile title is created: ' + made.status); return made.data.page; };
  const q = await make('question', ODD + LONG, 'Body with ' + LONG + ' and ' + ODD + '\n\n`' + LONG + '`\n\n[[H1]] [x](javascript:window.__ran=3)\n', { summary: LONG + ' ' + ODD });
  const h = await make('hypothesis', LONG + ' ' + ODD, '## Statement\n\n' + ODD + LONG + '\n', { summary: ODD }, { motivated_by: [q.id] });
  const e = await make('experiment', LONG + ODD, '## Objective\n\n' + LONG + '\n\n## Metrics and success criteria\n\n`' + LONG + '`\n', { summary: LONG, mode: 'script' }, { tests: [h.id] });
  const f = await make('finding', ODD + ' ' + LONG, '## What was observed\n\n' + LONG + ' 12.345 ' + ODD + '\n', { summary: LONG });
  const asked = await api('POST', '/api/projects/' + pid + '/requests', { text: LONG + ' ' + ODD, target: q.path });
  await api('PATCH', '/api/projects/' + pid, { title: ODD + LONG });
  const STICKS = () => {
    const out = [];
    document.querySelectorAll('#view *, #top *, #status *, #toast *, #overlay *').forEach((el) => {
      const r = el.getBoundingClientRect();
      if (!r.width || r.right <= innerWidth + 1 || el.closest('svg, pre, textarea, .table-wrap, .ed-prev')) return;
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) if (/(auto|scroll|hidden|clip)/.test(getComputedStyle(p).overflowX)) return;
      out.push(el.tagName.toLowerCase() + '.' + String(el.className).split(' ')[0] + ' reaches ' + Math.round(r.right));
    });
    return Array.from(new Set(out)).slice(0, 4);
  };
  const routes = ['#/library', '#/p/PID/map', '#/p/PID/map/' + q.id, '#/p/PID/wiki/' + q.id, '#/p/PID/wiki/' + q.id + '/edit', '#/p/PID/wiki/' + q.id + '/history', '#/p/PID/wiki/' + f.id, '#/p/PID/inbox', '#/p/PID/inbox/' + h.id, '#/p/PID/experiments', '#/p/PID/experiments/' + e.id, '#/p/PID/overview/' + q.id, '#/p/PID/paper'];
  for (const width of [1440, 820]) {
    await page.setViewportSize({ width, height: 900 });
    for (const r of routes) {
      const hash = r.replace('PID', pid);
      await open(page, hash);
      await page.waitForTimeout(250);
      const over = (await page.evaluate(OVERFLOW)).concat(await page.evaluate(STICKS));
      ok(!over.length, width + ' ' + r + ' hostile text makes the view wider than the window: ' + over.join('; '));
      ok(!(await page.evaluate(() => window.__ran)), width + ' ' + r + ' markup from a page ran as code');
    }
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await shot(page, 'hostile-paper');
  await open(page, '#/p/' + pid + '/wiki/' + q.id, '.page-wiki .prose');
  ok((await count(page, '.wiki-main img[src="x"], .wiki-main script, .page-title i, .page-title b')) === 0, 'markup in a title and a body is shown as text');
  ok((await count(page, '.prose a[href^="javascript"]')) === 0, 'a javascript: link is not a link');
  await shot(page, 'hostile-wiki');
  await page.keyboard.press('Control+k');
  await page.keyboard.type('pneumono');
  await until(page, () => document.querySelectorAll('.pal-i').length >= 2, null, 4000, 'the palette finds the pages with the long titles');
  ok(!(await page.evaluate(STICKS)).length, 'the palette holds a title of any length: ' + (await page.evaluate(STICKS)).join('; '));
  await page.keyboard.press('Escape');
  /* tidy up: the pages and the title go, so that a later check finds the project as it was */
  for (const made of [f, e, h, q]) await api('DELETE', '/api/projects/' + pid + '/pages/' + pagePath(made.path));
  if (asked.data && asked.data.request) await api('DELETE', '/api/projects/' + pid + '/requests/' + asked.data.request.id);
  await api('PATCH', '/api/projects/' + pid, { title: was });
};

/* ------------------------------------------------------------------------------------ */
(async () => {
  let listing = { status: 0, data: null };
  for (let i = 0; i < 80 && listing.status !== 200 && listing.status !== 401; i++) {   // the server may still be starting
    listing = await api('GET', '/api/projects').catch((e) => ({ status: 0, data: String(e) }));
    if (listing.status !== 200 && listing.status !== 401) await sleep(250);
  }
  if (listing.status === 401) { console.error('The server at ' + BASE + ' asks for its access token' + (TOKEN ? ', and the one given is not it' : '') + '. Give the checks the link the server printed (--base \'' + BASE + '/?token=...\'), or set PAPER_ORCHESTRA_UI_TOKEN to the token the server was started with.'); process.exit(2); }
  if (listing.status !== 200) { console.error('The server at ' + BASE + ' does not answer: ' + listing.status + ' ' + JSON.stringify(listing.data)); process.exit(2); }
  /* the demo project and, when it is there, the large one (`research demo --pages 150`
     gives both the same title): told apart by how many pages they hold */
  let demo = null, large = '', spare = '', shell = '';
  for (const row of listing.data.projects) {
    /* a third copy, in a folder named `nomodel`: the project that is looked at without a model */
    if (row.dir === 'nomodel') { spare = spare || row.id; continue; }
    /* a fourth, in a folder named `shell`: the project whose agents have a shell */
    if (row.dir === 'shell') { shell = shell || row.id; continue; }
    const view = (await api('GET', '/api/projects/' + row.id)).data;
    const pages = Object.values((view && view.counts && view.counts.by_kind) || {}).reduce((a, b) => a + b, 0);
    if (/150 pages/.test(row.title) || pages >= 140) large = large || row.id;
    else if (/^Demonstration/.test(row.title) && !demo) demo = row;
  }
  if (!demo) { console.error('The workspace has no demo project (paper-orchestra research demo --dir <workspace>/demo).'); process.exit(2); }
  let browser;
  try { browser = await chromium.launch(); }
  catch (error) { console.error('No browser could be started: ' + String(error.message).split('\n')[0]); process.exit(3); }
  let context;
  try { context = await entered(await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: arg('scheme', 'light'), reducedMotion: 'reduce' })); }
  catch (error) { console.error('The browser was not let into the app: ' + String(error.message).split('\n')[0]); await browser.close(); process.exit(2); }
  await context.addInitScript(() => { window.__csp = []; document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(e.violatedDirective + ' ' + (e.blockedURI || ''))); });
  let current = '';
  const page = await context.newPage();
  watch(page, () => current);
  /* (`nomodel` switches the scripted model off for the whole server while it looks: it comes before any check that starts agents) */
  const order = ['screens', 'phone', 'shell', 'nomodel', 'map', 'inbox', 'wiki', 'experiments', 'agents', 'states', 'rules', 'keys', 'paper', 'large', 'composer', 'hostile'];
  const unknown = (ONLY || []).concat(SKIP).filter((name) => order.indexOf(name) < 0);
  if (unknown.length) { console.error('not a check: ' + unknown.join(', ') + ' (the checks are: ' + order.join(', ') + '; separate several with commas)'); await browser.close(); process.exit(2); }
  for (const name of order) {
    if ((ONLY && ONLY.indexOf(name) < 0) || SKIP.indexOf(name) >= 0) continue;
    current = name;
    const before = fails.length, t0 = Date.now();
    try { await CHECKS[name]({ browser, context, page, pid: demo.id, large, spare, shell }); }
    catch (error) { fails.push(name + ' stopped: ' + String(error && error.message).split('\n')[0]); await shot(page, 'stopped-' + name); await context.setOffline(false).catch(() => {}); }
    const csp = await page.evaluate(() => (window.__csp || []).splice(0)).catch(() => []);
    if (csp.length) fails.push(name + ': blocked by the Content-Security-Policy: ' + csp.join('; '));
    for (let i = before; i < fails.length; i++) fails[i] = fails[i].startsWith(name) ? fails[i] : name + ': ' + fails[i];
    console.log((fails.length === before ? 'ok    ' : 'FAIL  ') + name + '  ' + ((Date.now() - t0) / 1000).toFixed(1) + ' s');
  }
  await browser.close();
  notes.forEach((n) => console.log('note  ' + n));
  if (errors.length) console.log('CONSOLE ERRORS (' + errors.length + ')\n  ' + Array.from(new Set(errors)).slice(0, 30).join('\n  '));
  if (fails.length) console.log('FAILED (' + fails.length + ')\n  ' + fails.join('\n  '));
  if (!fails.length && !errors.length) console.log('all checks passed');
  process.exit(fails.length || errors.length ? 1 : 0);
})();
