/* The entry point: the render loop, the store of the open project, the clock.

   Views are functions of (store, route) that return markup; `render` puts the markup of
   the open route into the page, keeping scroll positions and the caret. A render caused by
   the event stream waits while the person is in the middle of typing. */

import { app, openProjectUi, parseHash } from './core.js';
import { api, connection, Unreachable } from './api.js';
import { $, $$, clock, html, parseStamp, plain, setHTML } from './util.js';
import { Store } from './store.js';
import { banner, statusbar, topbar } from './shell.js';
import { label, noteTrail } from './page.js';
import { mapView } from './map.js';
import { wikiView } from './wiki.js';
import { inboxView } from './inbox.js';
import { experimentsView } from './experiments.js';
import { overviewView } from './overview.js';
import { paperView } from './paper.js';
import { libraryView, loadLibrary, setupView, startView } from './home.js';
import { flushAll, installActions } from './actions.js';
import { renderToast, sayError } from './overlays.js';

app.views = { map: mapView, wiki: wikiView, inbox: inboxView, experiments: experimentsView, overview: overviewView, paper: paperView, start: startView, library: libraryView, setup: setupView };

const KEEP = ['.idx-scroll', '.in-scroll', '.exp-rail', '.wiki-rail'];
const KEEP_SAME = ['.wiki-main', '.in-detail', '.exp-main', '.overview', '.library', '.setup', '.start', '.paper', '.ed-prev'];
const TYPING_MS = 1200;
const CHOOSING_MS = 12000;
let lastKey = '', pending = 0, waitingForPause = 0;

/* ---------- the keyboard keeps its place across a render ---------- */
const data = (el, names) => names.filter((k) => el.dataset[k] != null).map((k) => '[data-' + k + '="' + CSS.escape(el.dataset[k]) + '"]').join('');
/** What names the focused element once the markup around it has been written again: its id,
    or what it does and to what. `n` tells apart several that answer to the same name. */
function focusKey(el) {
  if (!el || !el.closest || !el.closest('#top, #view, #status, #toast')) return null;
  const tag = el.tagName.toLowerCase();
  let sel = '';
  if (el.id) sel = '#' + CSS.escape(el.id);
  else if (el.dataset.check) sel = 'input[data-check="' + CSS.escape(el.dataset.check) + '"]';
  else if (el.dataset.act) sel = tag + '[data-act="' + CSS.escape(el.dataset.act) + '"]' + data(el, ['ref', 'eid', 'run', 'name', 'stamp', 'key', 'i', 'path', 'start', 'backend', 'minutes']);
  else if (el.dataset.input) sel = tag + '[data-input="' + CSS.escape(el.dataset.input) + '"]' + data(el, ['eid', 'name', 'kind', 'ref', 'key']) + (el.type === 'radio' ? '[value="' + CSS.escape(el.value) + '"]' : '');
  else if (tag === 'a' && el.getAttribute('href')) sel = 'a[href="' + CSS.escape(el.getAttribute('href')) + '"]';
  else if (el.classList.contains('node') && el.dataset.key) sel = '.node[data-key="' + CSS.escape(el.dataset.key) + '"]';
  else if (el.dataset.log) sel = '[data-log="' + CSS.escape(el.dataset.log) + '"]';
  if (!sel) return null;
  let n = 0;
  try { n = Math.max(0, $$(sel).indexOf(el)); } catch (e) { return null; }
  const text = /^(INPUT|TEXTAREA)$/.test(el.tagName) && el.type !== 'checkbox' && el.type !== 'radio';
  let caret = null;
  try { if (text && el.selectionStart != null) caret = [el.selectionStart, el.selectionEnd, el.scrollTop]; } catch (e) { /* a field without a caret */ }
  return { sel, n, caret };
}
function refocus(key) {
  if (!key) return;
  const list = $$(key.sel);
  const el = list[Math.min(key.n, list.length - 1)];
  if (!el || el === document.activeElement || el.disabled || !el.focus) return;
  el.focus({ preventScroll: true });
  if (key.caret && el.setSelectionRange) { try { el.setSelectionRange(key.caret[0], key.caret[1]); } catch (e) { /* not a text field */ } el.scrollTop = key.caret[2]; }
}

function onStoreChange(reason, background, event) {
  if (reason === 'run.log') { if (app.hooks.runLog) app.hooks.runLog(event); return; }
  /* An experiment the person approved was put back to planned instead of being run (its
     code or plan is no longer what was approved): said aloud, wherever the person is looking. */
  if (reason === 'event' && event && event.type === 'experiment.state') {
    const d = event.data || {};
    if (d.by === 'system' && d.state === 'planned' && d.previous && d.previous !== 'planned' && d.reason) sayError((d.id || 'The experiment') + ' did not run and is planned again: ' + plain(d.reason).replace(/^Not run:\s*/i, ''));
  }
  renderSoon(reason, background !== false);
}
/** Several changes in one frame make one render. */
function renderSoon(reason, background) {
  if (pending) return;
  pending = requestAnimationFrame(() => { pending = 0; render(reason, background); });
}

function openProject(pid) {
  if (app.store && app.store.pid === pid) return app.store;
  if (app.store) { flushAll(); app.store.close(); }
  openProjectUi(pid);
  mapView.reset();
  app.store = new Store(pid, onStoreChange);
  app.store.load();
  return app.store;
}

function projectShell(route, store) {
  const error = store.loadError;
  if (!error) return html`<div class="blank loading" role="status"><div class="skeleton" aria-hidden="true"><i></i><i></i><i class="short"></i></div><p>Opening the project…</p></div>`;
  if (error.status === 404) return html`<div class="blank"><h1>There is no project “${route.pid}” here</h1><p>It may have been moved or deleted, or this address is from another workspace.</p><p><a class="btn btn-primary" href="#/library">All projects</a></p></div>`;
  return html`<div class="blank" role="alert"><h1>${error instanceof Unreachable ? 'The server is not answering' : 'The project could not be opened'}</h1><p>${error.message}</p><p><button class="btn btn-primary" data-act="reload-project">Try again</button> <a class="btn" href="#/library">All projects</a></p></div>`;
}

function render(reason, background) {
  const typed = Date.now() - (app.lastInputAt || 0);
  const active = document.activeElement;
  if (background && active && /^(INPUT|TEXTAREA)$/.test(active.tagName) && typed < TYPING_MS) {
    /* in the middle of a word: the page changes when the typing pauses */
    clearTimeout(waitingForPause);
    waitingForPause = setTimeout(() => render(reason, true), TYPING_MS - typed + 30);
    return;
  }
  if (background && active && active.tagName === 'SELECT' && Date.now() - (app.choosingAt || 0) < CHOOSING_MS) {
    /* a list of choices is open: writing its markup again would close it under the pointer.
       The page changes when the choice is made or given up (or after a while, at the latest) */
    clearTimeout(waitingForPause);
    waitingForPause = setTimeout(() => render(reason, true), 400);
    return;
  }
  const route = parseHash(location.hash);
  let store = null;
  if (route.name === 'project') {
    store = openProject(route.pid);
    if (store.loaded) route.sel = store.resolve(route.ref);
  } else if (app.store) { flushAll(); app.store.close(); app.store = null; }
  app.route = route;
  const main = $('#view');
  const key = [route.name, route.pid || '', route.view || '', route.sel || route.ref || '', route.sub || ''].join('/');
  app.renderKey = key;
  const same = key === lastKey;
  const saved = {};
  KEEP.concat(same ? KEEP_SAME : []).forEach((sel) => { const el = $(sel, main); if (el) saved[sel] = el.scrollTop; });
  const logs = same ? $$('.log-body', main).map((el) => ({ key: el.dataset.log, top: el.scrollTop, stuck: el.scrollTop + el.clientHeight >= el.scrollHeight - 30 })) : [];
  const opened = same ? $$('details[data-keep][open]', main).map((el) => el.dataset.keep) : [];
  const focused = focusKey(active);
  if (!same) app.ui.focusEditor = route.name === 'project' && route.sub === 'edit' ? key : '';

  setHTML($('#top'), topbar(route));
  const viewKey = route.name === 'project' ? route.pid + '/' + route.view + (store.loaded ? '' : '/loading') : route.name;
  if (main.dataset.view !== viewKey) { setHTML(main, html``); main.dataset.view = viewKey; main.className = 'view view-' + (route.view || route.name); }
  const bar = $('#status');
  if (route.name === 'project') {
    if (!store.loaded) { setHTML(main, projectShell(route, store)); bar.hidden = true; }
    else {
      if (route.sel && (route.view === 'wiki' || route.view === 'map' || route.view === 'inbox')) noteTrail(store, route.sel);
      (app.views[route.view] || app.views.map).render(main, route);
      bar.hidden = false;
      setHTML(bar, statusbar(store));
    }
  } else {
    app.views[route.name].render(main, route);
    bar.hidden = true;
  }
  document.body.classList.toggle('in-project', route.name === 'project' && !!store && store.loaded);

  Object.keys(saved).forEach((sel) => { const el = $(sel, main); if (el) el.scrollTop = saved[sel]; });
  logs.forEach((log) => { const el = $('.log-body[data-log="' + CSS.escape(log.key || '') + '"]', main); if (el && !log.stuck) el.scrollTop = log.top; });
  opened.forEach((keep) => { const el = $('details[data-keep="' + CSS.escape(keep) + '"]', main); if (el) el.open = true; });
  renderToast();
  const ban = $('#banner');
  ban.hidden = connection.ok;
  setHTML(ban, banner());
  /* the keyboard is where it was: on the same control, though its markup is new */
  refocus(focused);
  /* an editor that has just opened takes the keyboard: E, then type */
  if (app.ui.focusEditor === key) {
    const body = $('#ed-body', main);
    if (body) { app.ui.focusEditor = ''; if (!main.contains(document.activeElement) || !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) body.focus({ preventScroll: true }); }
  }
  if (app.ui.rejecting) { const input = $('.reject-form input', main); if (input && document.activeElement !== input) input.focus(); }
  const page = store && store.loaded && route.sel ? store.byPath.get(route.sel) : null;
  document.title = (store && store.loaded ? (page ? label(page) + ' · ' : '') + store.project.title : { start: 'New project', setup: 'Setup', library: 'Projects', project: 'Project' }[route.name]) + ' · PaperOrchestra';
  app.lastRenderKey = key;
  lastKey = key;
}
app.render = render;

/* ---------- the clock: stopwatches and meters move without a render ---------- */
function tick() {
  const now = Date.now();
  if (app.route.name === 'library' && !document.hidden) loadLibrary();   // a project that is running, or was just named, shows without a reload
  $$('[data-since]').forEach((el) => {
    const started = parseStamp(el.dataset.since);
    if (started == null) return;
    const elapsed = Math.max(0, (now - started) / 1000);
    if (el.tagName === 'PROGRESS') el.value = Math.min(1, elapsed / (+el.dataset.total || 1));
    else el.textContent = clock(elapsed);
  });
}

/* ---------- boot ---------- */
installActions();
document.addEventListener('click', (ev) => { if (ev.target.closest && ev.target.closest('[data-act="reload-project"]')) { if (app.store) { app.store.close(); app.store = null; } render('reload'); } });
let handledHash = location.hash;
window.addEventListener('hashchange', () => {
  if (location.hash === handledHash) return;      // the browser's own event for a change go() has already handled
  handledHash = location.hash;
  if (app.ui.keepReject) app.ui.keepReject = false; else app.ui.rejecting = null;
  app.ui.confirmDelete = null;
  render('route');
});
window.addEventListener('resize', () => renderSoon('resize', true));
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render('theme'));
connection.on((ok) => {
  /* the server is back: what could not be read while it was gone is asked for again */
  if (ok) { if (app.store && app.store.loaded) { app.store.retryFailed(); app.store.refreshNow(); } else if (app.store) app.store.load(); if (app.route.name === 'library') loadLibrary(true); }
});
/* while the server is gone, ask again every few seconds; fetches flip the state back */
setInterval(() => { if (!connection.ok) api.health().catch(() => {}); }, 3000);
setInterval(tick, 1000);
/* "38 min ago" should not stay true for ever */
setInterval(() => { if (document.visibilityState === 'visible') renderSoon('minute', true); }, 60000);
api.health().then((health) => { app.ui.health = health; if (app.route.name !== 'project') renderSoon('health', true); }, () => {});
if (!location.hash || location.hash === '#' || location.hash === '#/') {
  /* straight to the composer when there is no project yet, to the library otherwise */
  history.replaceState(null, '', '#/library');
  api.projects().then((data) => { app.ui.library = { projects: data.projects || [], error: null, loading: false, at: Date.now() }; if (!(data.projects || []).length && location.hash === '#/library') { history.replaceState(null, '', '#/start'); } render('boot'); }, () => render('boot'));
}
render('boot');
