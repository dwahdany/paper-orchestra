/* What lies over the page: the toast (with the short undo), the command palette, the
   keyboard sheet and the small dialog that adds a page. */

import { VIEWS, app, defaults, go } from './core.js';
import { api } from './api.js';
import { $, $$, debounce, html, raw, setHTML } from './util.js';
import { KIND, STAGES, glyph, icon, kindName, pageGlyph } from './vocab.js';
import { shortTitle } from './page.js';
import { engineView, nextStep, phaseChoices, theme } from './shell.js';

/* ---------- toast ---------- */
let toastTimer = 0;
/** Where the toast stands: at the foot of the window, in the middle, unless that is where
    the view keeps something to read or to press. On the map it stands above the legend and
    the tools; in the inbox over the proposal, not over the list and its last column. */
function placeToast(el) {
  const box = el.firstElementChild;
  el.style.bottom = ''; el.style.left = '';
  /* (on a phone the list of the inbox ends above its toast: the stylesheet reads both of these) */
  const onList = !!box && !!$('.inbox:not(.in-open)');
  document.body.classList.toggle('toast-on-list', onList);
  if (onList) document.body.style.setProperty('--toast-h', box.offsetHeight + 'px');
  if (!box) return;
  const foot = $('.map-view .map-foot'), pane = $('.inbox .in-detail');
  if (foot) { const top = foot.getBoundingClientRect().top; if (top > 0) el.style.bottom = Math.round(window.innerHeight - top + 10) + 'px'; }
  const at = pane ? pane.getBoundingClientRect() : null;
  /* (beside the list only where the proposal has a column of its own, wide enough for the toast) */
  if (at && at.left > 200 && at.width >= box.offsetWidth + 24) el.style.left = Math.round(at.left + at.width / 2) + 'px';
}
export function renderToast() {
  const el = $('#toast'), t = app.ui.toast;
  if (!el) return;
  setHTML(el, t ? html`<div class="toast${t.kind === 'error' ? ' toast-error' : ''}" role="${t.kind === 'error' ? 'alert' : 'status'}">${t.kind === 'error' ? icon('warn') : ''}<span class="toast-t">${t.text}${t.undo ? html` <span class="toast-q">Not sent yet.${t.note ? ' ' + t.note : ''}</span>` : ''}</span>
    ${t.undo ? html`<button class="btn btn-s" data-act="undo">${icon('undo')}Undo <kbd>Z</kbd></button><i class="toast-timer" aria-hidden="true"></i>` : ''}
    ${t.action ? html`<a class="btn btn-s" href="${t.action.href}" data-act="toast-x-follow">${t.action.label}</a>` : ''}
    <button class="ibtn" data-act="toast-x" aria-label="Dismiss">${icon('x')}</button></div>` : html``);
  placeToast(el);
}
/** Say something briefly. `options.undo` is the batch of decisions that can still be taken back. */
export function say(text, options) {
  options = options || {};
  app.ui.toast = { text, kind: options.kind || 'info', undo: options.undo || null, note: options.note || '', action: options.action || null };
  renderToast();
  clearTimeout(toastTimer);
  if (options.kind !== 'error' || options.ms) toastTimer = setTimeout(() => { if (app.ui.toast && app.ui.toast.text === text) { app.ui.toast = null; renderToast(); } }, options.ms || (options.undo ? 5200 : 6000));
}
/** A refusal or a failure: stays until dismissed. */
export const sayError = (text) => say(text, { kind: 'error' });
export function clearToast() { clearTimeout(toastTimer); app.ui.toast = null; renderToast(); }

/* ---------- overlay plumbing ---------- */
let returnFocus = null;
function openOverlay(kind, markup) {
  const ov = $('#overlay');
  if (ov.hidden) returnFocus = document.activeElement;
  ov.hidden = false;
  ov.dataset.kind = kind;
  setHTML(ov, markup);
  return ov;
}
export function closeOverlay(kind) {
  const ov = $('#overlay');
  if (ov.hidden || (kind && ov.dataset.kind !== kind)) return false;
  ov.hidden = true;
  ov.dataset.kind = '';
  setHTML(ov, html``);
  app.ui.newPage = null;
  if (returnFocus && document.contains(returnFocus) && returnFocus.focus) returnFocus.focus();
  returnFocus = null;
  return true;
}
export const overlayOpen = () => !$('#overlay').hidden;
/** Tab stays inside the dialog that is open. */
export function trapTab(ev) {
  const ov = $('#overlay');
  if (ov.hidden || ev.key !== 'Tab') return;
  const items = $$('input, select, textarea, button, a[href], [tabindex="0"]', ov).filter((el) => !el.disabled && el.offsetParent !== null);
  if (!items.length) return;
  const first = items[0], last = items[items.length - 1];
  if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last.focus(); }
  else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus(); }
}

/* ---------- command palette ---------- */
function commands() {
  const r = app.route, store = app.store, inProject = r.name === 'project' && store && store.loaded;
  const list = [];
  const add = (name, hint, run, keys) => list.push({ type: 'action', name, hint, run, keys });
  if (inProject) {
    const e = engineView(store);
    VIEWS.forEach((v, i) => add('Go to ' + v[1], String(i + 1), () => go({ view: v[0], sub: '', sel: v[0] === 'paper' ? '' : r.sel })));
    if (e.working) add('Stop the agents', 'ends at the next step', () => app.act('stop'));
    else if (e.unverified) add('Forget the worker this server cannot verify', 'nothing is stopped', () => app.act('forget-worker'));
    else {
      /* what a press of the main button does, and the phases that can be chosen in its place */
      const step = nextStep(store, e);
      /* (where a phase has to be named before anything goes on, only the phases are offered) */
      if (!step.pick && !step.blocked) add(step.label, step.href === '#/setup' ? 'opens Setup' : step.href ? 'opens Experiments' : 'runs up to the next checkpoint', () => (step.href ? go(step.href) : app.act('continue')));
      phaseChoices(store, e, step).forEach((c) => add(c.title, 'a phase of your choice', () => app.act('continue-phase', { dataset: { phase: c.phase } })));
    }
    add('New page', 'a page of your own', () => app.act('new-page'));
    add('Leave an instruction for the agents', 'on the overview', () => { go({ view: 'overview', sel: r.sel, sub: '' }); setTimeout(() => { const el = $('#rq-text'); if (el) el.focus(); }, 60); });
    add(app.ui.showRejected ? 'Hide rejected pages' : 'Show rejected pages', '', () => app.act('toggle-rejected'));
  }
  add('New project', 'start from an idea', () => go('#/start'));
  add('All projects', 'the library', () => go('#/library'));
  add('Setup check', 'credentials and backends', () => go('#/setup'));
  add(theme.effective() === 'dark' ? 'Switch to light' : 'Switch to dark', 'T', () => theme.toggle());
  add('Keyboard shortcuts', '?', () => openHelp());
  return list;
}

export function openPalette() {
  const r = app.route, store = app.store, inProject = r.name === 'project' && store && store.loaded;
  const ov = openOverlay('palette', html`<div class="pal" role="dialog" aria-modal="true" aria-label="Find a page or run an action">
    <div class="pal-in">${icon('search')}<input id="pal-in" aria-label="Find a page or run an action" placeholder="${inProject ? 'Go to a page by id or title, or run an action…' : 'Open a project, or run an action…'}" autocomplete="off" spellcheck="false" role="combobox" aria-expanded="true" aria-controls="pal-list" aria-autocomplete="list"><kbd>Esc</kbd></div>
    <div id="pal-list" class="pal-list" role="listbox" aria-label="Results"></div>
    <div class="pal-foot"><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>↵</kbd> open</span><span>pages by id, title or text · actions by name</span></div></div>`);
  const input = $('#pal-in', ov), list = $('#pal-list', ov);
  let cur = 0, hits = [], remote = [], remoteFor = '', control = null;
  const all = commands();
  const localPages = (q) => {
    if (!inProject) return [];
    const scored = [];
    store.pages.forEach((pg) => {
      const id = (pg.id || '').toLowerCase(), title = (pg.title || '').toLowerCase();
      let score = -1;
      if (!q) score = 0;
      else if (id === q) score = 100;
      else if (id.startsWith(q)) score = 80;
      else if (title.startsWith(q)) score = 60;
      else if (title.indexOf(q) >= 0 || id.indexOf(q) >= 0) score = 40;
      else if ((pg.kind + ' ' + (pg.summary || '')).toLowerCase().indexOf(q) >= 0) score = 15;
      if (score >= 0) scored.push({ type: 'page', page: pg, score: score - (pg.review === 'rejected' ? 5 : 0) });
    });
    if (!q) {
      const trail = ((app.ui.trail[store.pid] || {}).list || []).slice().reverse();
      return trail.map((p) => store.byPath.get(p)).filter(Boolean).slice(0, 5).map((page) => ({ type: 'page', page, recent: true }));
    }
    return scored.sort((a, b) => b.score - a.score || STAGES.indexOf(a.page.kind) - STAGES.indexOf(b.page.kind)).slice(0, 8);
  };
  const draw = () => {
    const q = input.value.trim().toLowerCase();
    const pages = localPages(q);
    const have = new Set(pages.map((h) => h.page.path));
    const more = q && remoteFor === q && inProject ? remote.filter((h) => !have.has(h.path) && store.byPath.has(h.path)).slice(0, 6).map((h) => ({ type: 'page', page: store.byPath.get(h.path), snippet: h.snippet })) : [];
    const projects = !inProject && app.ui.library && app.ui.library.projects ? app.ui.library.projects.filter((p) => !q || (p.title || '').toLowerCase().indexOf(q) >= 0).slice(0, 8).map((p) => ({ type: 'project', project: p })) : [];
    const acts = all.filter((a) => !q || a.name.toLowerCase().indexOf(q) >= 0).slice(0, q ? 6 : 12);
    hits = pages.concat(more, projects, acts);
    cur = Math.min(cur, Math.max(0, hits.length - 1));
    const row = (h, i) => {
      const on = i === cur;
      if (h.type === 'page') {
        const pg = h.page;
        return html`<div class="pal-i${on ? ' cur' : ''}" role="option" id="pal-o-${i}" aria-selected="${on ? 'true' : 'false'}" data-i="${i}">${pageGlyph(pg, 16)}<span class="pal-n">${pg.id && pg.kind !== 'source' ? html`<b>${pg.id}</b> ` : ''}${pg.kind === 'source' ? shortTitle(pg) + ' · ' + pg.title : pg.kind === 'result' ? shortTitle(pg) : pg.title}${h.snippet ? html`<span class="pal-s">${h.snippet}</span>` : ''}</span><span class="pal-k">${h.recent ? 'recent · ' : ''}${kindName(pg.kind).toLowerCase()}</span></div>`;
      }
      if (h.type === 'project') return html`<div class="pal-i${on ? ' cur' : ''}" role="option" id="pal-o-${i}" aria-selected="${on ? 'true' : 'false'}" data-i="${i}">${glyph('idea', 'accepted', '', 16)}<span class="pal-n">${h.project.title}</span><span class="pal-k">project</span></div>`;
      return html`<div class="pal-i pal-a${on ? ' cur' : ''}" role="option" id="pal-o-${i}" aria-selected="${on ? 'true' : 'false'}" data-i="${i}">${icon('right')}<span class="pal-n">${h.name}</span><span class="pal-k">${h.hint || 'action'}</span></div>`;
    };
    setHTML(list, hits.length ? html`${hits.map(row)}` : html`<p class="quiet pal-none">Nothing matches “${input.value}”.</p>`);
    input.setAttribute('aria-activedescendant', hits.length ? 'pal-o-' + cur : '');
    const on = $('.pal-i.cur', list);
    if (on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest' });
  };
  const search = debounce(() => {
    const q = input.value.trim().toLowerCase();
    if (!inProject || q.length < 2) return;
    if (control) control.abort();
    control = new AbortController();
    api.search(store.pid, q, control.signal).then((data) => { if (overlayOpen() && input.value.trim().toLowerCase() === q) { remote = data.hits || []; remoteFor = q; draw(); } }, () => {});
  }, 160);
  const pick = (i) => {
    const h = hits[i];
    closeOverlay('palette');
    if (!h) return;
    if (h.type === 'page') go({ name: 'project', pid: store.pid, view: r.view === 'paper' || r.view === 'overview' || (r.view === 'experiments' && h.page.kind !== 'experiment') ? 'wiki' : r.view, sel: h.page.path, sub: '' });
    else if (h.type === 'project') go('#/p/' + encodeURIComponent(h.project.id) + '/map');
    else h.run();
  };
  input.addEventListener('input', () => { cur = 0; draw(); search(); });
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowDown') { cur = Math.min(hits.length - 1, cur + 1); draw(); ev.preventDefault(); }
    else if (ev.key === 'ArrowUp') { cur = Math.max(0, cur - 1); draw(); ev.preventDefault(); }
    else if (ev.key === 'Enter') { ev.preventDefault(); pick(cur); }
  });
  list.addEventListener('click', (ev) => { const it = ev.target.closest('.pal-i'); if (it) pick(+it.dataset.i); });
  list.addEventListener('mousemove', (ev) => { const it = ev.target.closest('.pal-i'); if (it && +it.dataset.i !== cur) { cur = +it.dataset.i; $$('.pal-i', list).forEach((el, i) => el.classList.toggle('cur', i === cur)); } });
  draw();
  input.focus();
}

/* ---------- keyboard sheet ---------- */
export function openHelp() {
  const rows = [['Views', [['1 … 6', 'Map, Wiki, Inbox, Experiments, Overview, Paper: the selection stays'], ['Ctrl K', 'Find a page, run an action (also /)'], ['T', 'Light or dark'], ['Esc', 'Close, or clear the selection']]],
    ['Deciding', [['A', 'Accept the proposal in front of you'], ['E', 'Edit. Saving does not decide; “Save and accept” does'], ['R', 'Reject, with a reason the agents read'], ['Z', 'Undo, until the decision is sent (five seconds)'], ['X', 'Select in the inbox'], ['⇧A', 'Accept everything selected']]],
    ['Moving', [['← →', 'On the map: along the lines'], ['↑ ↓', 'On the map: within a stage'], ['Tab', 'Into the map at the selected page, then on to its tools'], ['J K', 'Next and previous in a list'], ['H', 'History of the page'], ['L', 'Expand or collapse a log'], ['+ − 0', 'Zoom the map, fit it'], ['Ctrl S', 'Save what you are editing']]]];
  const ov = openOverlay('help', html`<div class="help" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts"><h2>Keyboard</h2><div class="help-cols">${rows.map((g) => html`<div><h3>${g[0]}</h3><dl>${g[1].map((row) => html`<div><dt>${row[0].split(' ').map((k) => (k === '…' ? '… ' : html`<kbd>${k}</kbd> `))}</dt><dd>${row[1]}</dd></div>`)}</dl></div>`)}</div>
    <p class="quiet help-note">There is no undo of a decision once it is sent. After that, edit the page or decide again: every version is kept in its history.</p>
    <button class="btn" data-act="overlay-x">Close <kbd>Esc</kbd></button></div>`);
  const b = $('button', ov);
  if (b) b.focus();
}

/* ---------- a page of the person's own ---------- */
/** The kinds this dialog offers until the server has said which a person can write. */
const NEW_KINDS = ['topic', 'question', 'hypothesis', 'experiment', 'finding', 'journal'];
/** The kinds of page a person can write, as the server lists them (`creatable_kinds` of
    /api/defaults): in the order this dialog had, with what the server adds after it (a
    lesson, which is the person's alone). A kind no agent writes says so (`kind_writers`). */
function newKinds() {
  const d = defaults();
  const listed = d && Array.isArray(d.creatable_kinds) ? d.creatable_kinds.filter((k) => typeof k === 'string' && k) : null;
  const kinds = listed && listed.length ? NEW_KINDS.filter((k) => listed.indexOf(k) >= 0).concat(listed.filter((k) => NEW_KINDS.indexOf(k) < 0)) : NEW_KINDS;
  const writers = (d && d.kind_writers && typeof d.kind_writers === 'object' && d.kind_writers) || {};
  return kinds.map((k) => ({ kind: k, name: kindName(k), hint: (KIND[k] || {}).hint || '', yours: Array.isArray(writers[k]) && !writers[k].length }));
}
export function openNewPage() {
  const ui = app.ui.newPage || (app.ui.newPage = { kind: 'topic', title: '', error: null, saving: false });
  const kinds = newKinds();
  if (!kinds.some((k) => k.kind === ui.kind)) ui.kind = kinds[0].kind;
  const ov = openOverlay('new-page', html`<form class="dialog" role="dialog" aria-modal="true" aria-label="New page" data-form="new-page">
    <h2>A page of your own</h2><p class="quiet">It is accepted as you write it: your pages are never changed by an agent, which proposes an edit instead.</p>
    <label><span>Kind</span><select id="np-kind" name="kind" data-input="new-page-kind">${kinds.map((k) => html`<option value="${k.kind}"${ui.kind === k.kind ? raw(' selected') : ''}>${k.name}${k.hint ? ' · ' + k.hint : ''}${k.yours ? ' · only you write these' : ''}</option>`)}</select></label>
    <label><span>Title</span><input id="np-title" name="title" data-input="new-page-title" value="${ui.title}" autocomplete="off" placeholder="${ui.kind === 'question' ? 'What is still open?' : ui.kind === 'hypothesis' ? 'A claim that an experiment could refute' : ui.kind === 'lesson' ? 'What the agents should do differently' : 'What the page is about'}"></label>
    ${ui.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span>${ui.error}</span></p>` : ''}
    <div class="dialog-bar"><button class="btn btn-quiet" type="button" data-act="overlay-x">Cancel</button><button class="btn btn-primary" type="submit"${ui.saving ? raw(' disabled') : ''}>Create and write it</button></div></form>`);
  const title = $('#np-title', ov);
  if (title) { title.focus(); title.setSelectionRange(title.value.length, title.value.length); }
  /* opened before the server had said which kinds there are: drawn again, once, when it has */
  if (!defaults() && !ui.waiting) {
    ui.waiting = setInterval(() => {
      const open = app.ui.newPage === ui && $('#overlay').dataset.kind === 'new-page';
      if (open && !defaults()) return;
      clearInterval(ui.waiting);
      if (open) openNewPage();
    }, 150);
  }
}
