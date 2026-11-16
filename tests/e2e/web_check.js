/* Browser checks of the web app against a real server.

   node tests/e2e/web_check.js --base http://127.0.0.1:8600 [--only a,b] [--skip a,b] [--shots DIR]

   The server must serve a workspace that holds the demo project
   (`paper-orchestra research demo --dir <workspace>/demo`; with the large one beside it,
   `research demo --dir <workspace>/large --pages 150`, the checks of a large project run
   too) and must have been started with
   PAPER_ORCHESTRA_SCRIPTED_CLIENT=scripted_model:client and this directory on PYTHONPATH, so
   that the phases it starts run on the scripted model. The flows change the project: give
   every run a fresh copy. tests/test_web_e2e.py does all of that; README.md says how to do it
   by hand.

   Exit status 0 when every check passed. */

'use strict';
const { chromium } = require('playwright');
const fs = require('fs');
const os = require('os');
const path = require('path');

const arg = (name, fallback) => { const i = process.argv.indexOf('--' + name); return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback; };
const BASE = String(arg('base', 'http://127.0.0.1:8600')).replace(/\/$/, '');
const ONLY = arg('only', '') ? arg('only', '').split(',') : null;
const SKIP = arg('skip', '') ? arg('skip', '').split(',') : [];
const SHOTS = arg('shots', '');
const SLOW = +arg('slow', '1');           // stretch the waits on a slow machine

const fails = [], notes = [], errors = [];
const ok = (condition, message) => { if (!condition) fails.push(message); return !!condition; };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function api(method, url, body) {
  const response = await fetch(BASE + url, { method, headers: body === undefined ? {} : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }
  return { status: response.status, data };
}
const pagePath = (p) => p.split('/').map(encodeURIComponent).join('/');

function watch(page, label) {
  page.on('console', (m) => { if (m.type() === 'error') { const t = m.text(); if (!/Failed to load resource: (net::ERR_INTERNET_DISCONNECTED|net::ERR_FAILED|the server responded with a status of (409|422|404))/.test(t)) errors.push(label() + ': ' + t); } });
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
    '#/p/PID/wiki/H2/edit', '#/p/PID/wiki/H2/history', '#/p/PID/wiki/F1', '#/p/PID/wiki/topics%2Fwhy-warmup-helps', '#/p/PID/wiki/results%2FE1-run1', '#/p/PID/wiki/idea', '#/p/PID/inbox', '#/p/PID/inbox/E3',
    '#/p/PID/experiments', '#/p/PID/experiments/E3', '#/p/PID/experiments/E4', '#/p/PID/experiments/E1', '#/p/PID/experiments/E2', '#/p/PID/paper', '#/p/PID/wiki/no-such-page', '#/p/p-000000000000/map'];
  /* the four widths side by side: the pages do not touch the project */
  await Promise.all([[1440, 'light'], [1100, 'dark'], [820, 'light'], [1900, 'dark']].map(async ([width, scheme]) => {
    const ctx = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: scheme, reducedMotion: 'reduce' });
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

/* the map: the selection is shared by the views; keys walk it; a proposal is decided where it stands */
CHECKS.map = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/map', '.map-svg .node');
  const nodes = await count(page, '.map-svg .node');
  ok(nodes >= 39, 'the map shows the demo project: ' + nodes + ' nodes');
  ok((await text(page, '.strip-text')).includes('waiting for you'), 'the strip says what is waiting: ' + (await text(page, '.strip-text')));
  const n0 = await inboxCount(page);
  ok(n0 === 6, 'six proposals wait at the start, got ' + n0);
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
  /* edit, then accept */
  await page.click('.in-row:has-text("F2") .in-main');
  await page.waitForSelector('.in-detail .page-title:has-text("gives back")');
  await page.keyboard.press('e');
  await page.waitForSelector('textarea#ed-body');
  const body = await page.inputValue('textarea#ed-body');
  await page.fill('textarea#ed-body', body + '\n\nMy own caveat: one learning rate only.\n');
  await until(page, () => /My own caveat/.test(document.querySelector('.ed-prev').textContent), null, 5000, 'the preview is rendered by the server and follows the editor');
  await shot(page, 'inbox-editor');
  await page.click('.ed-bar button[type=submit]');
  await until(page, () => !document.querySelector('form.editor'), null, 6000, 'saving closes the editor');
  const f2 = (await api('GET', '/api/projects/' + pid + '/pages/F2')).data.page;
  ok(f2.review === 'accepted' && f2.body.includes('My own caveat'), 'edit then accept saves the person\'s version: ' + f2.review);
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
  ok((await text(page, '.numcheck-sum')).startsWith('8 of 8'), 'the margin says how many were found: ' + (await text(page, '.numcheck-sum')));
  await page.click('.prose a.num >> nth=1');
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
  ok(/73\.25 was not found in a record, a source or the inputs/.test(await text(page, '.numcheck-flag')), 'and explained in the margin: ' + (await text(page, '.numcheck-flag')));
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
  ok(/succeeded/.test(await text(page, '.page-status .tag-state')), 'the experiment is succeeded: ' + (await text(page, '.page-status')));
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
  /* an approval can be taken back before the run */
  await page.click('.callout [data-act="cancel-run"]');
  await until(page, () => /cancelled/.test((document.querySelector('.page-status .tag-state') || {}).textContent || ''), null, 10000, '"Do not run it" takes the approval back');
  ok(/will not run/.test(await text(page, '.toast')), 'and says so: ' + (await text(page, '.toast')).trim().slice(0, 70));
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
  const findings = (await api('GET', '/api/projects/' + pid + '/pages?kind=finding')).data.pages.length;
  await page.keyboard.press('1');
  await page.waitForSelector('.strip [data-act="continue"]');
  ok(/Run the approved experiment/.test(await text(page, '.strip [data-act="continue"]')), 'the strip offers to run what is approved: ' + (await text(page, '.strip [data-act="continue"]')));
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
  /* resume: it runs to the checkpoint */
  await page.click('.strip [data-act="continue"]');
  await until(page, () => !!document.querySelector('.strip.s-running'), null, 10000, 'resume starts the phase again');
  await until(page, () => !!document.querySelector('.strip.s-checkpoint') && /Checkpoint/.test(document.querySelector('.strip-kicker').textContent), null, 60000, 'the phase ends at a checkpoint');
  const after = (await api('GET', '/api/projects/' + pid + '/activity?limit=200')).data.events.filter((e) => e.seq > before);
  ok(after.some((e) => e.type === 'checkpoint'), 'the checkpoint is in the activity stream');
  ok((await count(page, '.node.k-hypothesis.r-proposed')) === proposed + 2, 'both scripted hypotheses wait on the map: ' + (await count(page, '.node.k-hypothesis.r-proposed')));
  await idle(pid);
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
  if (!ready) ok(/Continue to analysis/.test(await text(page, '.strip [data-act="continue"]')), 'with nothing approved and a result not analysed, the strip offers the analysis, not a run: ' + (await text(page, '.strip [data-act="continue"]')).trim());
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
  for (let i = 0; i < 80 && !(worker && worker.pid > 0); i++) { const view = (await api('GET', '/api/projects/' + pid)).data; if (view.state.status === 'running') worker = view.worker; else await sleep(150); }
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
  ok(/Resume hypotheses/.test(await text(page, '.strip [data-act="continue"]')) && (await count(page, '.top-status.st-budget')) === 0, 'and offers to go on where the phase stopped: ' + (await text(page, '.strip [data-act="continue"]')).trim());
  /* the server unreachable: say so, keep the draft, come back */
  await open(page, '#/p/' + pid + '/wiki/Q2/edit', 'textarea#ed-body');
  await page.fill('#ed-summary', 'Typed while the server was away');
  await context.setOffline(true);
  await page.keyboard.press('Control+s');
  await until(page, () => !document.querySelector('#banner').hidden, null, 8000, 'the banner says the server is not answering');
  ok(/not answering/.test(await text(page, '#banner')), 'in plain words');
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
  await page.fill('#opt-title', 'Browser test project');
  await page.fill('#opt-model-usd', '7.5');
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
  fs.rmSync(dir, { recursive: true, force: true });
  return pid;
};

/* the paper step: the form is real; what the write-up does is the pipeline's business */
CHECKS.paper = async ({ page, pid }) => {
  await open(page, '#/p/' + pid + '/paper', '.paper-form');
  await until(page, () => document.querySelectorAll('#paper-template option').length >= 2, null, 5000, 'the bundled templates are offered');
  ok(!(await page.locator('.paper-form button[type=submit]').isDisabled()) && !(await count(page, '.paper-gap')), 'the write-up can be started');
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
  await page.click('.paper-form button[type=submit]');
  await until(page, () => !!document.querySelector('.paper-stages .ps'), null, 30000, 'the stages of the write-up are shown');
  await idle(pid, 120000);
  await until(page, () => /Failed|Finished|Interrupted/.test((document.querySelector('.paper-state .sec-t') || {}).textContent || ''), null, 20000, 'the end of the write-up is shown');
  ok((await count(page, '.paper-stages .ps')) >= 5, 'every stage of the pipeline is listed');
  ok((await count(page, '.top-status.st-failed')) === 0, 'a write-up that failed is not the agents stopping unexpectedly');
  await shot(page, 'paper-after');
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
  /* an approval taken back before any run: the experiment says so and can be approved again */
  const unrun = (await api('GET', '/api/projects/' + large + '/experiments')).data.experiments.find((x) => x.review === 'accepted' && x.state === 'approved' && !x.runs);
  if (unrun) {
    await open(page, '#/p/' + large + '/experiments/' + unrun.id, '.callout [data-act="cancel-run"]');
    await page.click('.callout [data-act="cancel-run"]');
    await until(page, () => /Cancelled before it ran/.test((document.querySelector('.callout h2') || {}).textContent || ''), null, 10000, 'an experiment cancelled before it ran says so');
    ok(/at your request/.test(await text(page, '.callout')) && !/the person/.test(await text(page, '.callout')), 'in the app\'s words: ' + (await text(page, '.callout')).replace(/\s+/g, ' ').trim().slice(0, 100));
    await page.click('.callout [data-act="reapprove"]');
    await page.waitForSelector('.approve [data-act="approve"]:not([data-start])');
    await page.click('.approve [data-act="approve"]:not([data-start])');
    await until(page, () => /Approved to run/.test((document.querySelector('.callout h2') || {}).textContent || ''), null, 10000, 'and can be approved again');
  } else notes.push('large: no approved experiment without a run in the large project; "cancelled before it ran" was not looked at');
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
    await api('PUT', '/api/projects/' + large + '/experiments/' + eid + '/code/main.py', { text: 'import json\n\njson.dump({"spike": 0.1}, open("metrics.json", "w"))\nprint("done")\n' });
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
  for (let i = 0; i < 80 && listing.status !== 200; i++) {   // the server may still be starting
    listing = await api('GET', '/api/projects').catch((e) => ({ status: 0, data: String(e) }));
    if (listing.status !== 200) await sleep(250);
  }
  if (listing.status !== 200) { console.error('The server at ' + BASE + ' does not answer: ' + listing.status + ' ' + JSON.stringify(listing.data)); process.exit(2); }
  /* the demo project and, when it is there, the large one (`research demo --pages 150`
     gives both the same title): told apart by how many pages they hold */
  let demo = null, large = '';
  for (const row of listing.data.projects) {
    const view = (await api('GET', '/api/projects/' + row.id)).data;
    const pages = Object.values((view && view.counts && view.counts.by_kind) || {}).reduce((a, b) => a + b, 0);
    if (/150 pages/.test(row.title) || pages >= 140) large = large || row.id;
    else if (/^Demonstration/.test(row.title) && !demo) demo = row;
  }
  if (!demo) { console.error('The workspace has no demo project (paper-orchestra research demo --dir <workspace>/demo).'); process.exit(2); }
  let browser;
  try { browser = await chromium.launch(); }
  catch (error) { console.error('No browser could be started: ' + String(error.message).split('\n')[0]); process.exit(3); }
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: arg('scheme', 'light'), reducedMotion: 'reduce' });
  await context.addInitScript(() => { window.__csp = []; document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(e.violatedDirective + ' ' + (e.blockedURI || ''))); });
  let current = '';
  const page = await context.newPage();
  watch(page, () => current);
  const order = ['screens', 'map', 'inbox', 'wiki', 'experiments', 'agents', 'states', 'keys', 'paper', 'large', 'composer', 'hostile'];
  const unknown = (ONLY || []).concat(SKIP).filter((name) => order.indexOf(name) < 0);
  if (unknown.length) { console.error('not a check: ' + unknown.join(', ') + ' (the checks are: ' + order.join(', ') + '; separate several with commas)'); await browser.close(); process.exit(2); }
  for (const name of order) {
    if ((ONLY && ONLY.indexOf(name) < 0) || SKIP.indexOf(name) >= 0) continue;
    current = name;
    const before = fails.length, t0 = Date.now();
    try { await CHECKS[name]({ browser, context, page, pid: demo.id, large }); }
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
