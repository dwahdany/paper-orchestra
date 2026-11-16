/* The entry point: the render loop, the store of the open project, the clock.

   Views are functions of (store, route) that return markup; `render` puts the markup of
   the open route into the page, keeping scroll positions and the caret. A render caused by
   the event stream waits while the person is in the middle of typing. */

import { app, parseHash } from './core.js';
import { api, connection, Unreachable } from './api.js';
import { $, $$, clock, html, parseStamp, setHTML } from './util.js';
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
import { renderToast } from './overlays.js';

app.views = { map: mapView, wiki: wikiView, inbox: inboxView, experiments: experimentsView, overview: overviewView, paper: paperView, start: startView, library: libraryView, setup: setupView };

const KEEP = ['.idx-scroll', '.in-scroll', '.exp-rail', '.wiki-rail'];
const KEEP_SAME = ['.wiki-main', '.in-detail', '.exp-main', '.overview', '.library', '.setup', '.start', '.paper', '.ed-prev'];
const TYPING_MS = 1200;
let lastKey = '', pending = 0, waitingForPause = 0;

function onStoreChange(reason, background, event) {
  if (reason === 'run.log') { if (app.hooks.runLog) app.hooks.runLog(event); return; }
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
  app.ui.checked = {}; app.ui.rejecting = null; app.ui.confirmDelete = null; app.ui.activityOpen = false; app.ui.feedLimit = 40;
  app.ui.map = { fold: null, round: null, lineageOnly: false, deep: false, open: {} };
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
  const focusId = active && active.id && main.contains(active) ? active.id : '';
  const caret = focusId && active.selectionStart != null ? [active.selectionStart, active.selectionEnd, active.scrollTop] : null;

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
  if (focusId) {
    const el = document.getElementById(focusId);
    if (el && el !== document.activeElement) {
      el.focus({ preventScroll: true });
      if (caret && el.setSelectionRange) { try { el.setSelectionRange(caret[0], caret[1]); } catch (e) { /* not a text field */ } el.scrollTop = caret[2]; }
    }
  }
  opened.forEach((keep) => { const el = $('details[data-keep="' + CSS.escape(keep) + '"]', main); if (el) el.open = true; });
  if (app.ui.rejecting) { const input = $('.reject-form input', main); if (input && document.activeElement !== input) input.focus(); }
  renderToast();
  const ban = $('#banner');
  ban.hidden = connection.ok;
  setHTML(ban, banner());
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
  if (ok) { if (app.store && app.store.loaded) app.store.refreshNow(); else if (app.store) app.store.load(); if (app.route.name === 'library') loadLibrary(true); }
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
