/* Screenshots of every screen, for looking at the interface.

   node tests/e2e/web_shots.js --base http://127.0.0.1:8600 --out DIR [--only REGEX] [--themes light,dark] [--dsf 2]

   The server asks for its access token: it is taken from PAPER_ORCHESTRA_UI_TOKEN, or from
   --base when that is the link the server printed (http://127.0.0.1:8600/?token=...).

   Needs the demo project in the server's workspace (see README.md); with the project of 150
   pages beside it, its map is photographed too: with nothing selected, with a hypothesis, an
   experiment and a finding selected, and with every reference line drawn. Nothing is changed in
   the project: the shots that need a typed word or an open form type it and leave. The
   moving states (a running phase, a live log, a failure) are photographed by
   web_check.js --shots while it drives them. */

'use strict';
const { chromium } = require('playwright');
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
const OUT = path.resolve(arg('out', 'shots'));
const DSF = +arg('dsf', '2');
const ONLY = arg('only', '') ? new RegExp(arg('only', '')) : null;
const THEMES = arg('themes', 'light,dark').split(',');

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'po-shots-'));
const notes = path.join(tmp, 'notes.md'), runs = path.join(tmp, 'prior_run.csv');
fs.writeFileSync(notes, '# Notes\n\nThe spike is near step 40 at batch size 16.\n');
fs.writeFileSync(runs, 'step,loss\n' + Array.from({ length: 2000 }, (_, i) => i + ',' + (1 / (i + 1)).toFixed(5)).join('\n'));

const type = (selector, text) => async (page) => { await page.fill(selector, text); await page.waitForTimeout(400); };
/* a crowded map draws its reference lines for the selected page only; this asks for all of them */
const allLines = async (page) => { await page.click('.map-tools [data-act="map-refs"]'); await page.waitForSelector('.map-svg .edges .edge.t-cites'); await page.mouse.move(4, 4); await page.waitForTimeout(400); };
const SHOTS = [
  ['01-start-empty', '#/start'],
  ['01b-start-options', '#/start', async (page) => { await page.click('.adv summary'); await page.selectOption('#opt-cutoff-month', '6'); await page.fill('#opt-cutoff-year', '2024'); await page.waitForTimeout(300); }, async (page) => { await page.evaluate(() => { localStorage.removeItem('po-composer'); }); }],
  ['02-start-filled', '#/start', async (page) => {
    await page.fill('#idea-text', 'Learning-rate warmup is assumed to matter most for large batches. I suspect that with adaptive optimisers it is small batches that need it, because early second-moment estimates are noisy.');
    await page.setInputFiles('#file-input', [notes, runs]);
    await page.click('.adv summary');
    await page.waitForTimeout(200);
  }, async (page) => { await page.evaluate(() => { localStorage.removeItem('po-composer'); }); }],
  ['03-library', '#/library'],
  ['04-setup-check', '#/setup'],
  ['05-map-home', '#/p/PID/map'],
  ['06-map-lineage-h2', '#/p/PID/map/H2'],
  ['07-map-proposed-h5', '#/p/PID/map/H5'],
  ['08-map-reject-on-node', '#/p/PID/map', async (page) => { await page.click('.node.k-question.r-proposed .n-no'); await page.waitForTimeout(400); }],
  ['09-overview', '#/p/PID/overview'],
  ['09b-overview-lower', '#/p/PID/overview', async (page) => { await page.evaluate(() => { document.querySelector('.overview').scrollTop = 760; }); await page.waitForTimeout(200); }],
  ['09c-overview-other-phases', '#/p/PID/overview', async (page) => { await page.click('.ov-actions [data-act="go-menu"]'); await page.waitForSelector('.go-menu'); }],
  ['09d-overview-instruction-for', '#/p/PID/overview', async (page) => { await page.evaluate(() => { document.querySelector('.rq-form').scrollIntoView({ block: 'center' }); }); await page.selectOption('#rq-for', 'scout'); await page.waitForTimeout(200); }],
  ['05b-map-other-phases', '#/p/PID/map', async (page) => { await page.click('.strip [data-act="go-menu"]'); await page.waitForSelector('.go-menu'); }],
  ['10-wiki-page-h2', '#/p/PID/wiki/H2'],
  ['11-wiki-proposed-h5', '#/p/PID/wiki/H5'],
  ['12-wiki-editor', '#/p/PID/wiki/H2/edit', async (page) => { await page.waitForSelector('.ed-prev h2'); }],
  ['12b-wiki-editor-proposal', '#/p/PID/wiki/H5/edit', async (page) => { await page.waitForSelector('.ed-bar [data-act="save-accept"]'); await page.waitForSelector('.ed-prev h2'); }],
  ['13-wiki-history', '#/p/PID/wiki/topics%2Fwhy-warmup-helps/history', async (page) => { await page.click('.hist-i [data-act="version"] >> nth=-1'); await page.waitForSelector('.hist-body pre.diff'); }],
  ['14-wiki-finding-f1', '#/p/PID/wiki/F1'],
  ['14f-wiki-finding-f2-flagged', '#/p/PID/wiki/F2', async (page) => { await page.waitForSelector('.numcheck'); await page.waitForTimeout(300); }],
  ['14b-wiki-suggested-edit', '#/p/PID/wiki/topics%2Fwhy-warmup-helps', async (page) => { await page.waitForSelector('.prose-marked ins'); await page.evaluate(() => { const el = document.querySelector('.prose-marked .diff-changed'); if (el) el.scrollIntoView({ block: 'center' }); }); await page.waitForTimeout(200); }],
  ['14c-wiki-result-record', '#/p/PID/wiki/results%2FE1-run1'],
  ['14d-wiki-idea', '#/p/PID/wiki/idea'],
  ['14e-wiki-rejected', '#/p/PID/wiki/H4'],
  ['15-inbox', '#/p/PID/inbox'],
  ['15b-inbox-experiment', '#/p/PID/inbox/E3', async (page) => { await page.waitForSelector('.runon, .review .note'); }],
  ['15c-inbox-reject-reason', '#/p/PID/inbox/H5', async (page) => { await page.keyboard.press('r'); await page.keyboard.type('Covered by F1 already; narrow it first.'); await page.waitForTimeout(200); }],
  ['15d-inbox-bulk', '#/p/PID/inbox', async (page) => { await page.keyboard.press('x'); await page.keyboard.press('j'); await page.keyboard.press('x'); await page.waitForTimeout(200); }],
  ['17-experiments-table', '#/p/PID/experiments'],
  ['18-experiment-proposed-e3', '#/p/PID/experiments/E3', async (page) => { await page.waitForSelector('.approve .backend'); }],
  ['18b-experiment-approved-e4', '#/p/PID/experiments/E4', async (page) => { await page.waitForSelector('.exp-code .code-n'); }],
  ['18c-experiment-code-edit', '#/p/PID/experiments/E4', async (page) => { await page.waitForSelector('[data-act="edit-code"]'); await page.click('[data-act="edit-code"]'); await page.waitForTimeout(300); }],
  ['20-experiment-succeeded-e1', '#/p/PID/experiments/E1', async (page) => { await page.waitForSelector('.run.open .fig img'); await page.waitForTimeout(400); }],
  ['21-experiment-failed-e2', '#/p/PID/experiments/E2', async (page) => { await page.waitForSelector('.failure'); await page.waitForTimeout(500); }],
  ['24-paper', '#/p/PID/paper', async (page) => { await page.waitForSelector('#paper-template option', { state: 'attached' }); }],
  ['25-palette', '#/p/PID/map/H2', async (page) => { await page.keyboard.press('Control+k'); await page.keyboard.type('warm'); await page.waitForTimeout(700); }],
  ['26-keyboard-sheet', '#/p/PID/wiki/H2', async (page) => { await page.keyboard.press('?'); await page.waitForTimeout(200); }],
  ['27-new-page-dialog', '#/p/PID/wiki/H2', async (page) => { await page.click('[data-act="new-page"]'); await page.fill('#np-title', 'Why I doubt the variance account'); await page.waitForTimeout(200); }],
  ['28-large-map', '#/p/LARGE/map'],
  ['29-large-map-selected', '#/p/LARGE/map/H1'],
  ['29b-large-map-finding', '#/p/LARGE/map/F1'],
  ['29c-large-map-all-lines', '#/p/LARGE/map', allLines],
  ['29d-large-map-experiment', '#/p/LARGE/map/E1'],
  ['30-missing-project', '#/p/p-000000000000/map'],
];
const SIZES = [
  ['05-map-home', '#/p/PID/map', 1900, 1000], ['06-map-lineage-h2', '#/p/PID/map/H2', 1900, 1000], ['09-overview', '#/p/PID/overview', 1900, 1000], ['10-wiki-page-h2', '#/p/PID/wiki/H2', 1900, 1000],
  ['05-map-home', '#/p/PID/map', 820, 1000], ['06-map-lineage-h2', '#/p/PID/map/H2', 820, 1000], ['09-overview', '#/p/PID/overview', 820, 1000], ['10-wiki-page-h2', '#/p/PID/wiki/H2', 820, 1000],
  ['15-inbox', '#/p/PID/inbox', 1900, 1000],
  ['15-inbox', '#/p/PID/inbox', 820, 1000], ['17-experiments-table', '#/p/PID/experiments', 820, 1000], ['21-experiment-failed-e2', '#/p/PID/experiments/E2', 820, 1000],
  ['05-map-home', '#/p/PID/map', 1100, 800], ['10-wiki-page-h2', '#/p/PID/wiki/H2', 1100, 800],
  ['28-large-map', '#/p/LARGE/map', 1900, 1000], ['29-large-map-selected', '#/p/LARGE/map/H1', 1900, 1000], ['29b-large-map-finding', '#/p/LARGE/map/F1', 1900, 1000], ['29c-large-map-all-lines', '#/p/LARGE/map', 1900, 1000, allLines],
  ['28-large-map', '#/p/LARGE/map', 820, 1000], ['29-large-map-selected', '#/p/LARGE/map/H1', 820, 1000],
];

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const answer = await fetch(BASE + '/api/projects', { headers: AUTH });
  if (answer.status === 401) { console.error('The server at ' + BASE + ' asks for its access token. Give the link it printed (--base \'' + BASE + '/?token=...\'), or set PAPER_ORCHESTRA_UI_TOKEN to the token the server was started with.'); fs.rmSync(tmp, { recursive: true, force: true }); process.exit(2); }
  const listing = await answer.json();
  /* the demo project and the large one (`research demo --pages 150` gives both the same
     title) are told apart by how many pages they hold */
  let demo = null, large = null;
  for (const row of listing.projects) {
    /* (a third copy in a folder named `nomodel` belongs to web_check.js, which changes it) */
    if (row.dir === 'nomodel') continue;
    const view = await (await fetch(BASE + '/api/projects/' + row.id, { headers: AUTH })).json();
    if (/150 pages/.test(row.title) || ((view.counts || {}).pages || 0) >= 140) large = large || row;
    else if (/^Demonstration/.test(row.title) && !demo) demo = row;
  }
  if (!demo) { console.error('The workspace has no demo project.'); process.exit(2); }
  const browser = await chromium.launch();
  const errors = [];
  const shoot = async (name, hash, w, h, theme, pre, post) => {
    if (hash.includes('LARGE') && !large) return;
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: DSF, colorScheme: theme, reducedMotion: 'reduce' });
    /* the link with the token is opened once in every browser context, before anything else */
    if (TOKEN) { const door = await ctx.newPage(); await door.goto(BASE + '/?token=' + encodeURIComponent(TOKEN)); await door.close(); }
    const page = await ctx.newPage();
    page.on('console', (m) => { if (m.type() === 'error' && !/status of 404/.test(m.text())) errors.push(name + ': ' + m.text()); });
    page.on('pageerror', (e) => errors.push(name + ': ' + e.message));
    await page.goto(BASE + '/' + hash.replace('PID', demo.id).replace('LARGE', large ? large.id : ''));
    await page.waitForSelector('#view > *:not(.boot):not(.loading)', { timeout: 15000 });
    await page.waitForTimeout(700);
    try { if (pre) await pre(page); } catch (e) { errors.push(name + ' (preparing): ' + String(e.message).split('\n')[0]); }
    await page.screenshot({ path: path.join(OUT, name + (w === 1440 ? '' : '-' + w) + '-' + theme + '.png') });
    try { if (post) await post(page); } catch (e) { /* tidying up only */ }
    await ctx.close();
  };
  for (const theme of THEMES) for (const s of SHOTS) { if (ONLY && !ONLY.test(s[0])) continue; await shoot(s[0], s[1], 1440, 900, theme, s[2], s[3]); }
  for (const s of SIZES) { if (ONLY && !ONLY.test(s[0] + '-' + s[2])) continue; for (const theme of THEMES) await shoot(s[0], s[1], s[2], s[3], theme, s[4]); }
  await browser.close();
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(errors.length ? 'ERRORS\n' + errors.join('\n') : 'no console errors');
  console.log(fs.readdirSync(OUT).filter((f) => f.endsWith('.png')).length + ' files in ' + OUT);
})();
void type;
