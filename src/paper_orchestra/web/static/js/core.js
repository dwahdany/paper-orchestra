/* What every module shares: the open route, the interface state that is not in the address,
   the open project's store, and the way to ask for a render. `main.js` fills in `render`. */

import { storage } from './util.js';
import { api } from './api.js';

export const VIEWS = [['map', 'Map'], ['wiki', 'Wiki'], ['inbox', 'Inbox'], ['experiments', 'Experiments'], ['overview', 'Overview'], ['paper', 'Paper']];
const VIEW_NAMES = VIEWS.map((v) => v[0]);

export const app = {
  /** `{name: 'library' | 'start' | 'setup'}` or `{name: 'project', pid, view, ref, sel, sub, q}`;
      `ref` is what the address says, `sel` the path of the page it names (once loaded). */
  route: { name: 'library', q: {} },
  /** The store of the open project, or null. */
  store: null,
  views: {},
  /** Interface state kept across renders (never the project's data). */
  ui: {
    start: null,            // the composer: text, files, options, upload state
    wikiFilter: '',
    checked: {},            // inbox: refs selected for a bulk decision
    rejecting: null,        // the ref whose reject form is open
    reasons: {},            // ref -> the reason being typed
    activityOpen: false,
    logOpen: false,
    showRejected: storage.get('po-show-rejected', true) !== false,
    map: { fold: null, round: null, lineageOnly: false, deep: false, open: {} },
    approve: {},            // experiment id -> the options chosen in the approval form
    codeTab: {},            // experiment id -> the file shown
    codeEdit: {},           // experiment id + file -> the text being edited
    runOpen: {},            // experiment id + run -> whether its details are unfolded
    asDiff: {},             // proposal id -> show the unified diff instead of the marked text
    histVersion: {},        // page path -> the stamp of the version looked at
    confirmDelete: null,    // the path of the page about to be deleted
    trail: {},              // project id -> the pages opened on the way
    request: { text: '', about: true, for: 'all' },
    goMenu: null,           // which Continue control has its list of phases open: 'strip', 'overview' or null
    cutoffEdit: null,       // the research cutoff being changed on the overview
    starting: {},           // experiment id -> a run is being started (the server may settle an unfinished one first)
    cancelling: {},         // experiment id -> its approval is being taken back, or its run cancelled
    defaults: null,         // GET /api/defaults, as last fetched
    feedLimit: 40,
    newPage: null,          // the "new page" dialog
    limitEdit: null,        // raising a limit from the strip
    paper: {},
    drafts: {},             // editor drafts in memory (they are also kept in localStorage)
    editProposal: {},       // page path -> the edit proposal being edited before it is accepted
    reapprove: null,        // the experiment whose approval form is open again
    netNudge: null,         // the experiment whose question about network access was left unanswered
    showLessons: false,
    toast: null,
    flash: null,
    health: null,           // GET /api/health, as last answered
    library: null,          // the list of projects, as last fetched
    doctor: null,
  },
  render() {},
  /** Views that draw outside the main render (the live log) register here. */
  hooks: {},
};

const freshMap = () => ({ fold: null, round: null, lineageOnly: false, deep: false, open: {} });

/** What a new project starts with, the phases in order and whom a request can be for, as
    the server says them (GET /api/defaults). `null` until they are known: whoever shows a
    number then shows none, never one of its own. Asked for once, and again after a failure. */
export function defaults() {
  const d = app.ui.defaults || (app.ui.defaults = { data: null, loading: false, error: null, at: 0 });
  if (!d.data && !d.loading && Date.now() - d.at > 4000) {
    d.loading = true;
    api.defaults().then((data) => { d.data = data && typeof data === 'object' ? data : null; d.error = null; }, (error) => { d.error = error; })
      .finally(() => { d.loading = false; d.at = Date.now(); app.render('defaults', true); });
  }
  return d.data;
}
/** Ask the server whether it is there, and keep what it says about itself (its workspace,
    whether it stops its workers when it ends): the banner for a server that is gone says the
    last thing the server said, since it can no longer be asked. */
export function askHealth() {
  return api.health().then((health) => { if (health && typeof health === 'object') app.ui.health = health; return health; });
}
/** Where what the person is typing is kept on this device until it is sent or saved. */
export const requestKey = (pid) => 'po-request:' + pid;
export const requestForKey = (pid) => 'po-request-for:' + pid;
export const codeKey = (pid, key) => 'po-code:' + pid + ':' + key;
export const reasonsKey = (pid) => 'po-reasons:' + pid;

/** The interface state that belongs to one project. Another project starts without it: the
    unsaved code of E1 here must never turn up in the E1 of another project, nor an
    instruction typed for this one be sent to that one. What was being typed for the project
    that opens (an instruction, an edit of a code file) is read back from this device. */
export function openProjectUi(pid) {
  const ui = app.ui;
  const reasons = storage.get(reasonsKey(pid), {});
  ui.checked = {}; ui.rejecting = null; ui.confirmDelete = null;
  ui.reasons = reasons && typeof reasons === 'object' && !Array.isArray(reasons) ? reasons : {};
  ui.activityOpen = false; ui.logOpen = false; ui.feedLimit = 40; ui.map = freshMap();
  ui.approve = {}; ui.codeTab = {}; ui.codeEdit = {}; ui.runOpen = {}; ui.asDiff = {}; ui.histVersion = {};
  ui.newPage = null; ui.limitEdit = null; ui.limits = null; ui.limitsOpen = false;
  ui.editProposal = {}; ui.reapprove = null; ui.netNudge = null; ui.showLessons = false; ui.focusEditor = '';
  ui.goMenu = null; ui.cutoffEdit = null; ui.starting = {}; ui.cancelling = {}; ui.decidedAll = false;
  /* (an earlier version kept on this device what rests on a rejected experiment; the server says it now) */
  storage.remove('po-rested:' + pid);
  ui.paper = { templates: ui.paper.templates, template: ui.paper.template };
  const typed = String(storage.get(requestKey(pid), '') || '');
  /* whom the instruction being typed is for is kept with it (and only with it) */
  ui.request = { text: typed, about: true, for: typed.trim() ? String(storage.get(requestForKey(pid), 'all') || 'all') : 'all' };
  const prefix = codeKey(pid, '');
  storage.keys(prefix).forEach((key) => {
    const text = storage.get(key);
    if (typeof text !== 'string') return;
    const name = key.slice(prefix.length), cut = name.indexOf('/');
    ui.codeEdit[name] = { text, error: null, saving: false, restored: true };
    if (cut > 0) ui.codeTab[name.slice(0, cut)] = name.slice(cut + 1);   // the file with the unsaved edit is the one shown
  });
}

const decode = (text) => { try { return decodeURIComponent(text); } catch (e) { return text; } };

/** `#/p/<pid>/<view>/<selection>/<sub>`, `#/library`, `#/start`, `#/setup`. */
export function parseHash(hash) {
  const [path, qs] = String(hash || '').replace(/^#\/?/, '').split('?');
  const seg = path.split('/').filter(Boolean).map(decode);
  const q = {};
  (qs || '').split('&').filter(Boolean).forEach((kv) => { const i = kv.indexOf('='); if (i < 0) q[decode(kv)] = '1'; else q[decode(kv.slice(0, i))] = decode(kv.slice(i + 1)); });
  if (seg[0] === 'p' && seg[1]) return { name: 'project', pid: seg[1], view: VIEW_NAMES.includes(seg[2]) ? seg[2] : 'map', ref: seg[3] || '', sel: '', sub: seg[4] || '', q };
  if (seg[0] === 'start' || seg[0] === 'setup' || seg[0] === 'library') return { name: seg[0], q };
  return { name: 'library', q };
}

/** The shortest name of a page that an address can carry: its id, or its place in the wiki. */
export function shortRef(path) {
  if (!path) return '';
  const page = app.store && app.store.byPath.get(path);
  if (page && page.id) return page.id;
  return String(path).replace(/^wiki\//, '').replace(/\.md$/, '');
}

export function hrefOf(route) {
  if (route.name !== 'project') return '#/' + route.name;
  const ref = route.sel ? shortRef(route.sel) : route.ref || '';
  return '#/p/' + encodeURIComponent(route.pid) + '/' + route.view + (ref ? '/' + encodeURIComponent(ref) + (route.sub ? '/' + encodeURIComponent(route.sub) : '') : '');
}
/** The address of another view or selection of the open project. */
export const link = (patch) => hrefOf(Object.assign({}, app.route, { ref: '' }, patch));

export function go(patch, replace) {
  const next = typeof patch === 'string' ? patch : link(patch);
  if (location.hash === next) { app.render('route'); return; }
  if (replace) history.replaceState(null, '', next); else location.hash = next;
  window.dispatchEvent(new HashChangeEvent('hashchange'));
}
export const select = (path, sub) => go({ sel: path || '', sub: sub || '' });
