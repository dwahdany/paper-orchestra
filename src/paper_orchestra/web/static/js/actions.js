/* What the person can do: one table of actions (the `data-act` attributes of the markup),
   one handler each for clicks, submits, typing, changes, drops and keys. */

import { VIEWS, app, go, select } from './core.js';
import { api, connection, Unreachable } from './api.js';
import { $, ago, bytes, cap, plain, plural } from './util.js';
import { STAGES, EXTRA_KINDS, agentName } from './vocab.js';
import { dropDraft, isDirty, label, schedulePreview, touchDraft } from './page.js';
import { mapView } from './map.js';
import { currentItem } from './inbox.js';
import { EXAMPLE, clearComposer, composer, composerReady, composerSettings, keepComposer, loadDoctor, loadLibrary, totalSize } from './home.js';
import { clearToast, closeOverlay, openHelp, openNewPage, openPalette, overlayOpen, renderToast, say, sayError, trapTab } from './overlays.js';
import { theme } from './shell.js';

const GRACE_MS = 5000;
let batchCounter = 0;
const store = () => app.store;
const message = (error) => (error instanceof Unreachable ? 'The server is not answering.' : (error && error.message) || 'Something went wrong.');

/* ---------- decisions: optimistic, sent after a short grace period ---------- */
function entryFor(ref, decision, options) {
  const item = store().pending().find((it) => it.ref === ref);
  if (!item) return null;
  return {
    ref, decision, reason: options.reason || null, edits: options.edits || null,
    approve_run: options.approve_run == null ? null : options.approve_run,
    base_updated: item.type === 'page' ? options.base_updated || item.page.updated : null,
    target: item.page.path, item,
    label: item.type === 'edit' ? 'the change to ' + label(item.page) : label(item.page),
    batch: options.batch || ++batchCounter, at: Date.now(), sent: false,
  };
}
/** Where the inbox goes once the item in front is decided: the next one, or nothing. */
function nextInInbox(item) {
  const r = app.route;
  if (r.view !== 'inbox') return undefined;
  const items = store().pending(), cur = currentItem(store(), r);
  if (!cur || cur.ref !== item.ref) return undefined;
  const i = items.findIndex((it) => it.ref === item.ref);
  const next = items[i + 1] || items[i - 1];
  return next ? next.page.path : '';
}
function describe(entries, decision) {
  const verb = decision === 'accept' ? 'Accepted' : 'Rejected';
  return entries.length === 1 ? verb + ' ' + entries[0].label + '.' : verb + ' ' + plural(entries.length, 'proposal') + '.';
}
function failureText(failure) {
  const { entry, result } = failure;
  const why = plain((result.error && result.error.message) || '');
  if (result.approved === false) return why;
  if (result.status === 409 && result.page) {
    const by = (result.page.meta || {}).updated_by || '';
    const who = by === 'human' ? 'you changed it ' : by && by !== 'system' ? 'the ' + agentName(by) + ' changed it ' : 'it was changed ';
    return cap(entry.label) + ' changed after you read it (' + who + ago(result.page.updated) + '), so your decision was not applied. It is shown as it is now.';
  }
  if (result.status === 409) return cap(entry.label) + ' was not decided: ' + why + ' It is shown as it is now.';
  if (result.status === 404) return cap(entry.label) + ' no longer exists; nothing was decided.';
  return cap(entry.label) + ' was not ' + (entry.decision === 'accept' ? 'accepted' : 'rejected') + ': ' + why;
}
async function flush(entries) {
  const s = store();
  entries = entries.filter((entry) => !entry.sent && !entry.cancelled);
  if (!s || !entries.length) return { failed: [], results: [] };
  if (app.ui.toast && app.ui.toast.undo && app.ui.toast.undo.some((e) => entries.indexOf(e) >= 0)) { app.ui.toast = Object.assign({}, app.ui.toast, { undo: null }); renderToast(); }
  try {
    const out = await s.send(entries);
    if (out.failed.length) sayError(out.failed.map(failureText).join(' '));
    out.results.forEach((result, i) => { if (result.ok && result.worker_error) sayError(cap(entries[i].label) + ' is accepted and approved, but the run could not start now: ' + plain(result.worker_error.message) + ' It runs with the next run phase.'); });
    out.failed.forEach((failure) => { const page = failure.entry.item && failure.entry.item.page; if (page && page.kind === 'experiment' && page.id) s.invalidateExperiment(page.id); });   // its code may be what changed: read it again
    app.render('decided');
    return out;
  } catch (error) {
    sayError(message(error) + ' ' + (entries.length === 1 ? cap(entries[0].label) + ' was not decided and is still proposed.' : 'The ' + entries.length + ' decisions were not recorded.'));
    app.render('decision-failed');
    return { failed: entries.map((entry) => ({ entry, result: { status: 0 } })), results: [] };
  }
}
/** Send everything that still waits out its grace period (before the agents are started, a
    run begins, the project is left). */
export async function flushAll() {
  const s = store();
  if (!s) return;
  const waiting = s.unsent();
  if (waiting.length) await flush(waiting);
}
function later(entries) {
  setTimeout(() => flush(entries), GRACE_MS);
}
function decide(ref, decision, options) {
  options = options || {};
  const s = store();
  const entry = entryFor(ref, decision, options);
  if (!entry) return null;
  const next = nextInInbox(entry.item);
  s.decide(entry);
  app.ui.flash = { key: entry.target, at: Date.now(), decision };
  app.ui.rejecting = null;
  delete app.ui.reasons[ref];
  delete app.ui.checked[ref];
  if (options.immediate) { app.render('decide'); return flush([entry]); }
  later([entry]);
  say(describe([entry], decision), { undo: [entry] });
  if (next !== undefined && next !== app.route.sel) go({ sel: next, sub: '' }, true); else app.render('decide');
  return entry;
}
function bulk(decision, reason) {
  const s = store(), refs = Object.keys(app.ui.checked || {});
  const batch = ++batchCounter;
  const entries = refs.map((ref) => entryFor(ref, decision, { batch, reason })).filter(Boolean);
  if (!entries.length) return;
  entries.forEach((entry) => s.decide(entry));
  app.ui.checked = {};
  app.ui.rejecting = null;
  delete app.ui.reasons.bulk;
  later(entries);
  say(describe(entries, decision), { undo: entries });
  if (app.route.view === 'inbox') go({ sel: '', sub: '' }, true); else app.render('bulk');
}
function undo() {
  const t = app.ui.toast;
  if (!t || !t.undo || !store()) return false;
  const n = store().retract(t.undo);
  clearToast();
  if (n) { app.ui.flash = null; say(n === 1 ? 'Taken back. Nothing was sent.' : 'Taken back: ' + n + ' decisions. Nothing was sent.'); app.render('undo'); }
  return !!n;
}

/* ---------- editing a page ---------- */
const draftKey = (path) => 'po-draft:' + store().pid + ':' + path;
async function saveEdit(path) {
  const s = store(), page = s.byPath.get(path), draft = app.ui.drafts[draftKey(path)];
  if (!page || !draft || draft.saving) return;
  if (!draft.title.trim()) { draft.error = 'A page needs a title.'; app.render('edit'); return; }
  draft.saving = true; draft.error = null;
  app.render('saving');
  const fields = { summary: draft.summary };
  if (draft.state && draft.state !== (draft.base.state || '')) fields.state = draft.state;
  try {
    if (draft.proposal || page.review === 'proposed') {
      const ref = draft.proposal || path;
      const entry = { ref, decision: 'accept', edits: { body: draft.body, fields: Object.assign({ title: draft.title }, fields) }, base_updated: draft.proposal ? null : draft.base_updated, target: path, label: label(page), sent: false };
      const out = await s.send([entry]);
      if (out.failed.length) {
        const result = out.failed[0].result;
        if (result.status === 409) { const view = await api.page(s.pid, path); draft.conflict = view.page; }
        else draft.error = (result.error && result.error.message) || 'The server refused it.';
        return;
      }
      dropDraft(s, path);
      go({ sub: '' }, true);
      say(draft.proposal ? 'Saved your version of the change to ' + label(page) + '.' : 'Saved your version and accepted ' + label(page) + '.');
    } else {
      await api.putPage(s.pid, path, { title: draft.title, body: draft.body, fields, base_updated: draft.base_updated });
      dropDraft(s, path);
      s.invalidate(path);
      await s.refreshNow();
      go({ sub: '' }, true);
      say('Saved. The previous version is in the history.');
    }
  } catch (error) {
    if (error.status === 409 && error.body && error.body.page) draft.conflict = error.body.page;
    else draft.error = message(error);
  } finally {
    draft.saving = false;
    app.render('saved');
  }
}
async function restoreVersion(path, stamp) {
  const s = store(), page = s.byPath.get(path);
  try {
    const version = await api.version(s.pid, path, stamp);
    const body = { body: version.body, base_updated: page.updated };
    if (version.meta && version.meta.title) body.title = version.meta.title;
    await api.putPage(s.pid, path, body);
    s.invalidate(path);
    delete app.ui.histVersion[path];
    await s.refreshNow();
    say('Restored as a new version. The text it replaced is in the history.');
    go({ sub: '' }, true);
  } catch (error) { sayError('Not restored: ' + message(error)); }
}

/* ---------- the agents ---------- */
async function advance() {
  const s = store();
  await flushAll();
  try {
    const answer = await api.advance(s.pid, {});
    if (answer.worker) s.project.worker = answer.worker;
    if (app.route.view !== 'map' && app.route.view !== 'overview') go({ view: 'map', sel: '', sub: '' });
    s.refresh();
    app.render('advance');
  } catch (error) {
    sayError(error.status === 409 ? error.message : 'The agents were not started: ' + message(error));
    s.refresh();
  }
}
async function stopAgents() {
  const s = store();
  try {
    const answer = await api.stop(s.pid);
    if (answer.worker) s.project.worker = answer.worker;
    say(answer.worker && answer.worker.state === 'idle' ? 'Nothing was running.' : 'Stopping. What was written so far is kept; the phase can be resumed.');
    s.refresh();
    app.render('stop');
  } catch (error) { sayError('Not stopped: ' + message(error)); }
}

/* ---------- experiments ---------- */
function approvalBody(eid, start) {
  const s = store(), row = s.expById.get(eid), entry = s.exps.get(eid), chosen = app.ui.approve[eid] || {};
  const body = { start: !!start };
  if (chosen.backend) body.backend = chosen.backend;
  if (chosen.gpu != null) body.gpu = String(chosen.gpu).trim() || null;
  if (chosen.minutes != null && chosen.minutes !== '' && !Number.isNaN(+chosen.minutes)) body.timeout_seconds = Math.round(+chosen.minutes * 60);
  if (chosen.network != null) body.network = !!chosen.network;
  if (row) body.base_updated = row.updated;
  const digest = entry && entry.data && entry.data.code && entry.data.code.digest;
  if (digest) body.code_digest = digest;
  return body;
}
async function approve(eid, start, overrides) {
  const s = store();
  await flushAll();
  try {
    const body = Object.assign(approvalBody(eid, start), overrides || {});
    const answer = await api.approve(s.pid, eid, body);
    app.ui.reapprove = null;
    delete app.ui.runOpen[eid];
    if (answer.worker) s.project.worker = answer.worker;
    s.invalidateExperiment(eid);
    await s.refreshNow();
    if (answer.worker_error) sayError(eid + ' is approved, but the run could not start now: ' + answer.worker_error.message + ' It runs with the next run phase.');
    else say(start ? eid + ' is approved; its run is starting.' : eid + ' is approved. It runs with the next run phase, or now.');
  } catch (error) {
    sayError(eid + ' was not approved: ' + message(error));
    s.invalidateExperiment(eid);
    s.refresh();
  }
  app.render('approve');
}
function acceptAndRun(path) {
  const s = store(), page = s.byPath.get(path);
  if (!page) return;
  const body = approvalBody(page.id, true);
  const run = { start: true };
  ['backend', 'gpu', 'timeout_seconds', 'network', 'code_digest'].forEach((k) => { if (body[k] != null) run[k] = body[k]; });
  const done = decide(path, 'accept', { approve_run: run, immediate: true });
  if (done && done.then) done.then((out) => { if (out && !out.failed.length) { s.invalidateExperiment(page.id); say(label(page) + ' is accepted; its run is starting.'); if (app.route.view !== 'experiments') go({ view: 'experiments', sel: path, sub: '' }); } });
}

/* ---------- the composer ---------- */
function addFiles(list) {
  const s = composer();
  list.forEach((f) => { if (!s.files.some((x) => x.name === f.name && x.size === f.size)) s.files.push(f); });
  s.lostFiles = 0; s.error = null;
  keepComposer();
  app.render('files');
}
/** The files of a drop; a dropped folder is walked, its files keep their relative path. */
function filesOfDrop(transfer) {
  const entries = Array.from(transfer.items || []).map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null));
  if (!entries.some((e) => e && e.isDirectory)) return Promise.resolve(Array.from(transfer.files || []).map((file) => ({ file, name: file.name, size: file.size })));
  const out = [];
  const walk = (entry, prefix) => new Promise((resolve) => {
    if (!entry || out.length >= 2000 || entry.name.startsWith('.')) return resolve();
    if (entry.isFile) return entry.file((file) => { out.push({ file, name: prefix + file.name, size: file.size }); resolve(); }, () => resolve());
    const reader = entry.createReader();
    const read = () => reader.readEntries((batch) => { if (!batch.length) return resolve(); Promise.all(batch.map((child) => walk(child, prefix + entry.name + '/'))).then(read); }, () => resolve());
    read();
  });
  return Promise.all(entries.map((e) => walk(e, ''))).then(() => out);
}
function startProject() {
  const s = composer();
  if (!composerReady(s)) return;
  const settings = composerSettings(s);
  const bad = Object.entries(settings.limits || {}).find(([, v]) => Number.isNaN(v) || v < 0);
  if (bad) { s.error = 'A limit must be a number (' + bad[0].replace(/_/g, ' ') + ').'; app.render('start'); return; }
  const form = new FormData();
  if (s.title.trim()) form.append('title', s.title.trim());
  form.append('brief', s.text);
  if (Object.keys(settings).length) form.append('settings', JSON.stringify(settings));
  form.append('start', 'true');
  s.files.forEach((f) => form.append('files', f.file, f.name));
  s.uploading = true; s.progress = 0; s.error = null;
  app.render('upload');
  const request = api.createProject(form, (fraction) => {
    s.progress = fraction;
    const bar = $('#upload-bar'), text = $('#upload-say');
    if (bar) bar.value = fraction;
    if (text) text.textContent = fraction >= 1 ? 'Sent. The project is being created…' : 'Sending ' + (s.files.length ? plural(s.files.length, 'file') + ', ' : '') + bytes(Math.round(totalSize(s) * fraction)) + ' of ' + bytes(totalSize(s)) + '…';
  });
  s.request = request;
  request.then((project) => {
    clearComposer();
    app.ui.library = null;
    go('#/p/' + encodeURIComponent(project.id) + '/map');
    const skipped = (project.skipped || []).length;
    if (project.worker_error) sayError('The project is created, but the agents could not be started: ' + project.worker_error.message + ' Start them from here once the cause is put right.');
    else say('The project is created' + (project.worker ? ' and research has started.' : '.') + (skipped ? ' ' + plural(skipped, 'file') + ' could not be attached.' : ''));
  }, (error) => {
    s.uploading = false; s.request = null;
    s.error = error && error.name === 'AbortError' ? null : message(error);
    app.render('upload-failed');
  });
}

/* ---------- which proposal a key acts on ---------- */
function reviewTarget() {
  const r = app.route, s = store();
  if (!r || r.name !== 'project' || !s || !s.loaded) return null;
  if (r.view === 'inbox') { const it = currentItem(s, r); return it ? { ref: it.ref, page: it.page } : null; }
  const page = s.byPath.get(r.sel);
  if (!page) return null;
  if (page.review === 'proposed') return { ref: page.path, page };
  const proposal = s.pendingFor(page.path)[0];
  return proposal ? { ref: proposal.id, page } : null;
}
const editable = (page) => page && page.author !== 'system';
function selectInContext(path) {
  const r = app.route, page = store() && store().byPath.get(path);
  if (!page) return;
  const stay = r.view === 'map' || r.view === 'wiki' || r.view === 'inbox' || (r.view === 'experiments' && page.kind === 'experiment');
  go({ view: stay ? r.view : 'wiki', sel: path, sub: '' });
}

/* ---------- the table of actions ---------- */
const ACT = {
  accept: (el) => decide(el.dataset.ref, 'accept'),
  'accept-run': (el) => acceptAndRun(el.dataset.ref),
  reaccept: async (el) => {
    const s = store(), page = s.byPath.get(el.dataset.ref);
    try {
      const answer = await api.review(s.pid, [{ ref: el.dataset.ref, decision: 'accept' }]);
      const result = (answer.results || [])[0] || {};
      if (!result.ok) sayError((result.error && result.error.message) || 'It could not be accepted.'); else say('Accepted ' + (page ? label(page) : 'the page') + ' after all.');
      s.invalidate(el.dataset.ref);
      await s.refreshNow();
    } catch (error) { sayError('Not accepted: ' + message(error)); }
  },
  'reject-ask': (el) => {
    app.ui.rejecting = el.dataset.ref;
    const key = el.dataset.key, r = app.route;
    if (el.dataset.view && r.view !== el.dataset.view) { app.ui.keepReject = true; go({ view: el.dataset.view, sel: key, sub: '' }); }
    else if (key && r.sel !== key) { app.ui.keepReject = true; select(key); }
    else app.render('reject');
  },
  'reject-cancel': () => { app.ui.rejecting = null; app.render('reject'); },
  edit: (el) => {
    const r = app.route, path = el.dataset.ref || r.sel;
    if (!store().byPath.has(path)) return;
    const before = app.ui.drafts[draftKey(path)];
    if (before && before.proposal) dropDraft(store(), path);
    delete app.ui.editProposal[path];
    if (r.view === 'experiments' || r.view === 'overview') go({ view: 'wiki', sel: path, sub: 'edit' });
    else go({ sel: path, sub: r.sub === 'edit' && r.sel === path ? '' : 'edit' });
  },
  'edit-proposal': (el) => { const path = el.dataset.path; dropDraft(store(), path); app.ui.editProposal[path] = el.dataset.ref; go({ view: app.route.view === 'map' ? 'wiki' : app.route.view, sel: path, sub: 'edit' }); },
  'edit-cancel': () => go({ sub: '' }),
  'draft-discard': (el) => { dropDraft(store(), el.dataset.ref); if (app.route.sub === 'edit') go({ sub: '' }); else app.render('draft'); say('The draft is discarded. The page is as it was.'); },
  'conflict-mine': () => { const d = app.ui.drafts[draftKey(app.route.sel)]; if (!d || !d.conflict) return; d.base_updated = d.conflict.updated; d.conflict = null; saveEdit(app.route.sel); },
  'conflict-rebase': () => { const d = app.ui.drafts[draftKey(app.route.sel)]; if (!d || !d.conflict) return; const c = d.conflict; d.base = { title: c.title, summary: c.summary || '', state: c.state || '', body: c.body || '' }; d.base_updated = c.updated; d.conflict = null; touchDraft(store(), app.route.sel, 'body', d.body); say('You are now editing on top of the newer version. Saving replaces it with your text.'); app.render('conflict'); },
  'conflict-theirs': () => { dropDraft(store(), app.route.sel); store().invalidate(app.route.sel); go({ sub: '' }); },
  history: (el) => { const r = app.route, path = el.dataset.ref || r.sel; go({ sel: path, sub: r.sub === 'history' && r.sel === path ? '' : 'history' }); },
  version: (el) => { const path = el.dataset.ref; app.ui.histVersion[path] = app.ui.histVersion[path] === el.dataset.stamp ? '' : el.dataset.stamp; app.render('version'); },
  restore: (el) => restoreVersion(el.dataset.ref, el.dataset.stamp),
  'delete-ask': (el) => { app.ui.confirmDelete = el.dataset.ref; app.render('delete'); },
  'delete-cancel': () => { app.ui.confirmDelete = null; app.render('delete'); },
  delete: async (el) => {
    const s = store(), path = el.dataset.ref, page = s.byPath.get(path);
    try {
      await api.deletePage(s.pid, path);
      app.ui.confirmDelete = null;
      dropDraft(s, path);
      await s.refreshNow();
      go({ sel: '', sub: '' });
      say('Deleted ' + (page ? label(page) : 'the page') + '. Its last version is kept under history/.');
    } catch (error) { app.ui.confirmDelete = null; sayError('Not deleted: ' + message(error)); app.render('delete'); }
  },
  'reload-page': (el) => { store().invalidate(el.dataset.ref); app.render('reload'); },
  'reload-experiment': (el) => { store().invalidateExperiment(el.dataset.eid); app.render('reload'); },
  'toggle-diff': (el) => { app.ui.asDiff[el.dataset.ref] = !app.ui.asDiff[el.dataset.ref]; app.render('diff'); },
  'jump-change': () => { const el = $('.prose-marked .diff-changed, .prose-marked ins, .prose-marked del'); if (el) { el.scrollIntoView({ block: 'center', behavior: 'smooth' }); el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 1600); } },
  'close-panel': () => select(''),
  continue: () => advance(),
  stop: () => stopAgents(),
  'limit-edit': () => { const limits = (store().project.settings || {}).limits || {}; app.ui.limitEdit = { value: String(Math.ceil((limits.model_usd || 25) * 1.5)) }; app.render('limit'); setTimeout(() => { const el = $('#limit-usd'); if (el) el.select(); }, 0); },
  'limit-cancel': () => { app.ui.limitEdit = null; app.render('limit'); },
  'limits-open': () => { app.ui.limitsOpen = !app.ui.limitsOpen; if (!app.ui.limitsOpen) app.ui.limits = null; },
  'toggle-activity': () => { app.ui.activityOpen = !app.ui.activityOpen; app.render('ui'); },
  'toggle-log': () => { app.ui.logOpen = !app.ui.logOpen; app.render('ui'); },
  'toggle-rejected': () => { app.ui.showRejected = !app.ui.showRejected; try { localStorage.setItem('po-show-rejected', JSON.stringify(app.ui.showRejected)); } catch (e) { /* for this visit only */ } app.render('ui'); },
  'toggle-fold': () => { const m = app.ui.map; m.fold = !mapView.folded(); m.open = {}; mapView.reset(); app.render('ui'); },
  'map-reset': () => { const m = app.ui.map; m.round = null; m.lineageOnly = false; mapView.reset(); app.render('ui'); },
  'zoom-in': () => mapView.zoom(1), 'zoom-out': () => mapView.zoom(-1), 'zoom-fit': () => mapView.zoom(0),
  theme: () => theme.toggle(),
  palette: () => openPalette(), help: () => openHelp(), 'overlay-x': () => closeOverlay(),
  undo: () => undo(), 'toast-x': () => clearToast(), 'toast-x-follow': () => clearToast(),
  'retry-connection': () => { api.health().then(() => { if (store()) store().refreshNow(); app.render('online'); }, () => {}); },
  'start-example': () => { const s = composer(); s.text = EXAMPLE; keepComposer(); app.render('ui'); const t = $('#idea-text'); if (t) t.focus(); },
  'file-add': () => { const input = $('#file-input'); if (input) input.click(); },
  'file-del': (el) => { const s = composer(); s.files.splice(+el.dataset.i, 1); keepComposer(); app.render('ui'); },
  'lost-ok': () => { composer().lostFiles = 0; keepComposer(); app.render('ui'); },
  'upload-cancel': () => { const s = composer(); if (s.request && s.request.abort) s.request.abort(); },
  adv: () => { const s = composer(); s.adv = !s.adv; keepComposer(); },
  'reload-library': () => { loadLibrary(true); app.render('ui'); },
  recheck: () => { loadDoctor(true); app.render('ui'); },
  copy: (el) => { const text = el.dataset.text || ''; if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(() => say('Copied.'), () => say('Select the text to copy it.')); else say('Select the text to copy it.'); },
  'rq-del': async (el) => { const s = store(); try { await api.deleteRequest(s.pid, el.dataset.ref); await s.refreshNow(); say('The instruction is withdrawn.'); } catch (error) { sayError('Not withdrawn: ' + message(error)); } },
  'feed-more': async () => { const s = store(); app.ui.feedLimit += 60; if (s.activity.length < app.ui.feedLimit && s.moreActivity) { try { await s.loadEarlierActivity(); } catch (error) { sayError(message(error)); } } app.render('ui'); },
  'new-page': () => openNewPage(),
  'show-lessons': () => { app.ui.showLessons = true; app.render('ui'); },
  'bulk-accept': () => bulk('accept'),
  'bulk-reject': () => { app.ui.rejecting = app.ui.rejecting === 'bulk' ? null : 'bulk'; app.render('ui'); },
  approve: (el) => approve(el.dataset.eid, !!el.dataset.start),
  reapprove: (el) => { app.ui.reapprove = el.dataset.eid || null; app.render('ui'); if (app.ui.reapprove) setTimeout(() => { const f = $('#approve'); if (f) f.scrollIntoView({ block: 'start', behavior: 'smooth' }); }, 0); },
  rerun: (el) => { const eid = el.dataset.eid; const over = {}; if (el.dataset.backend) over.backend = el.dataset.backend; if (el.dataset.minutes) over.timeout_seconds = Math.round(+el.dataset.minutes * 60); const row = store().expById.get(eid); app.ui.approve[eid] = {}; approve(eid, true, Object.assign({ base_updated: row ? row.updated : undefined }, over)); },
  'run-now': async (el) => {
    const s = store(), eid = el.dataset.eid;
    await flushAll();
    try { const answer = await api.run(s.pid, eid); if (answer.worker) s.project.worker = answer.worker; delete app.ui.runOpen[eid]; s.invalidateExperiment(eid); await s.refreshNow(); say('The run of ' + eid + ' is starting.'); }
    catch (error) { sayError(eid + ' was not started: ' + message(error)); }
    app.render('run');
  },
  'cancel-run': async (el) => {
    const s = store(), eid = el.dataset.eid;
    try { await api.cancel(s.pid, eid); say('Cancelling the run of ' + eid + '. It is recorded as cancelled.'); s.refresh(); }
    catch (error) { sayError('Not cancelled: ' + message(error)); }
  },
  'show-run': (el) => { const eid = el.dataset.eid, n = +el.dataset.run; const s = store(), entry = s.exps.get(eid); const runs = ((entry && entry.data && entry.data.runs) || []).slice().sort((a, b) => b.number - a.number); const now = app.ui.runOpen[eid] === undefined ? (runs[0] && runs[0].status !== 'running' ? runs[0].number : null) : app.ui.runOpen[eid]; app.ui.runOpen[eid] = now === n ? null : n; app.render('ui'); },
  'code-tab': (el) => { app.ui.codeTab[el.dataset.eid] = el.dataset.name; app.render('ui'); },
  'edit-code': (el) => {
    const eid = el.dataset.eid, s = store(), entry = s.exps.get(eid), files = (entry && entry.data && entry.data.code.files) || [];
    const plan = (entry && entry.data && entry.data.plan) || {};
    const name = files.some((f) => f.name === app.ui.codeTab[eid]) ? app.ui.codeTab[eid] : (files.find((f) => f.name === (plan.entrypoint || 'main.py')) || files[0] || {}).name;
    const file = files.find((f) => f.name === name);
    if (!file) return;
    app.ui.codeTab[eid] = name;
    app.ui.codeEdit[eid + '/' + name] = { text: file.text, error: null, saving: false };
    app.render('ui');
    setTimeout(() => { const t = $('#code-text'); if (t) { t.focus(); t.scrollIntoView({ block: 'nearest' }); } }, 0);
  },
  'code-cancel': (el) => { delete app.ui.codeEdit[el.dataset.key]; app.render('ui'); },
};
/** Run an action by name (the palette does). */
app.act = (name, el) => { if (ACT[name]) return ACT[name](el || { dataset: {} }); return undefined; };

async function saveCode(form) {
  const s = store(), eid = form.dataset.eid, name = form.dataset.name, key = eid + '/' + name, edit = app.ui.codeEdit[key];
  if (!edit || edit.saving) return;
  edit.saving = true; edit.error = null;
  app.render('saving');
  try {
    const answer = await api.putCode(s.pid, eid, name, edit.text);
    delete app.ui.codeEdit[key];
    s.invalidateExperiment(eid);
    await s.refreshNow();
    const problems = answer.problems || [];
    if (problems.length) sayError('Saved ' + name + '. The check found: ' + problems.join(' ')); else say('Saved ' + name + '. It parses; the earlier version is kept.');
  } catch (error) { edit.error = message(error); }
  edit.saving = false;
  app.render('saved');
}

export function installActions() {
  document.addEventListener('click', (ev) => {
    const actEl = ev.target.closest('[data-act]');
    if (actEl && ACT[actEl.dataset.act]) {
      if (actEl.disabled || actEl.getAttribute('aria-disabled') === 'true') return;
      const follow = actEl.tagName === 'A' && actEl.dataset.act.endsWith('-follow');
      if (actEl.tagName !== 'SUMMARY' && !follow) ev.preventDefault();
      ACT[actEl.dataset.act](actEl, ev);
      return;
    }
    if (ev.defaultPrevented || ev.button || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
    const sel = ev.target.closest('[data-sel]');
    if (sel && app.route.name === 'project' && store() && store().loaded) { ev.preventDefault(); selectInContext(sel.dataset.sel); return; }
    const row = ev.target.closest('[data-go]');
    if (row && !ev.target.closest('a,button,input,label')) { go(row.dataset.go); return; }
    if (ev.target.id === 'overlay') closeOverlay();
  });

  document.addEventListener('submit', (ev) => {
    const form = ev.target.closest('form[data-form]');
    if (!form) return;
    ev.preventDefault();
    const kind = form.dataset.form, s = store();
    if (kind === 'reject') decide(form.dataset.ref, 'reject', { reason: (form.elements.reason.value || '').trim() });
    else if (kind === 'bulk-reject') bulk('reject', (form.elements.reason.value || '').trim());
    else if (kind === 'edit') saveEdit(form.dataset.ref);
    else if (kind === 'code') saveCode(form);
    else if (kind === 'start') startProject();
    else if (kind === 'request') {
      const rq = app.ui.request, text = rq.text.trim();
      if (!text || rq.sending) return;
      const target = rq.about && app.route.sel && s.byPath.has(app.route.sel) ? app.route.sel : null;
      rq.sending = true; rq.error = null;
      app.render('request');
      api.addRequest(s.pid, text, target).then(async () => { rq.text = ''; await s.refreshNow(); say('Added. It goes to the next phase that runs.'); }, (error) => { rq.error = message(error); })
        .finally(() => { rq.sending = false; app.render('request'); });
    } else if (kind === 'limit') {
      const usd = Number(String(form.elements.usd.value).replace(/[$,\s]/g, ''));
      if (!(usd > 0)) { sayError('The limit must be a number of dollars.'); return; }
      api.patchProject(s.pid, { settings: { limits: { model_usd: usd } } }).then(async () => { app.ui.limitEdit = null; await s.refreshNow(); say('The limit for models is now $' + usd.toFixed(2) + '.'); app.render('limit'); }, (error) => sayError('The limit was not changed: ' + message(error)));
    } else if (kind === 'limits') {
      const ui = app.ui.limits;
      if (!ui || ui.saving) return;
      const number = (text) => Number(String(text).replace(/[$,\s]/g, ''));
      const limits = { model_usd: number(ui.model_usd), experiment_usd: number(ui.experiment_usd), experiment_seconds: Math.round(number(ui.minutes) * 60) };
      if (Object.values(limits).some((v) => Number.isNaN(v) || v < 0)) { ui.error = 'Each limit is a number: dollars, dollars, minutes.'; app.render('limits'); return; }
      ui.saving = true; ui.error = null;
      api.patchProject(s.pid, { settings: { limits } }).then(async () => { app.ui.limits = null; app.ui.limitsOpen = false; await s.refreshNow(); say('The limits are changed.'); }, (error) => { ui.error = message(error); })
        .finally(() => { if (app.ui.limits) app.ui.limits.saving = false; app.render('limits'); });
    } else if (kind === 'new-page') {
      const ui = app.ui.newPage;
      if (!ui || ui.saving) return;
      if (!ui.title.trim()) { ui.error = 'Give the page a title.'; openNewPage(); return; }
      ui.saving = true; ui.error = null;
      api.createPage(s.pid, { kind: ui.kind, title: ui.title.trim(), body: '' }).then(async (view) => {
        await s.refreshNow();
        closeOverlay('new-page');
        go({ view: 'wiki', sel: view.page.path, sub: 'edit' });
      }, (error) => { ui.saving = false; ui.error = message(error); openNewPage(); });
    } else if (kind === 'paper') {
      const ui = app.ui.paper;
      ui.error = null;
      api.startPaper(s.pid, { template: ui.template, model: (ui.model || '').trim() || null }).then((answer) => { if (answer.worker) s.project.worker = answer.worker; s.loadPaper(); s.refresh(); say('The write-up has started.'); app.render('paper'); }, (error) => { ui.error = message(error); app.render('paper'); });
    }
  });

  document.addEventListener('input', (ev) => {
    const t = ev.target, what = t.dataset ? t.dataset.input : '';
    app.lastInputAt = Date.now();
    const form = t.closest ? t.closest('form[data-form=edit]') : null;
    if (form && t.name) {
      const draft = touchDraft(store(), form.dataset.ref, t.name, t.value);
      if (!draft) return;
      draft.restored = false;
      if (t.name === 'body') schedulePreview(store());
      const note = $('[data-draft]', form);
      if (note) note.textContent = isDirty(draft) ? 'Draft kept on this device' : 'No changes yet';
      return;
    }
    if (what === 'wiki-filter') { app.ui.wikiFilter = t.value; app.render('filter'); }
    else if (what === 'start-text') {
      const s = composer(), was = composerReady(s);
      s.text = t.value; keepComposer();
      if (was !== composerReady(s) || s.error) { s.error = null; app.render('start'); }
    } else if (what === 'opt') { const s = composer(); s[t.dataset.name] = t.value; keepComposer(); if (t.dataset.name === 'model_usd') app.render('start'); }
    else if (what === 'reason') app.ui.reasons[t.dataset.ref] = t.value;
    else if (what === 'request') app.ui.request.text = t.value;
    else if (what === 'code') { const edit = app.ui.codeEdit[t.dataset.key]; if (edit) edit.text = t.value; }
    /* drawn again while typing, never on blur: a render between the press and the release of a
       button would swallow the click that follows the typing */
    else if (what === 'ap-minutes') { (app.ui.approve[t.dataset.eid] = app.ui.approve[t.dataset.eid] || {}).minutes = t.value; app.render('backend'); }
    else if (what === 'ap-gpu') { (app.ui.approve[t.dataset.eid] = app.ui.approve[t.dataset.eid] || {}).gpu = t.value; app.render('backend'); }
    else if (what === 'paper-model') app.ui.paper.model = t.value;
    else if (what === 'limits' && app.ui.limits) app.ui.limits[t.dataset.name] = t.value;
    else if (what === 'limit' && app.ui.limitEdit) app.ui.limitEdit.value = t.value;
    else if (what === 'new-page-title' && app.ui.newPage) app.ui.newPage.title = t.value;
  });

  document.addEventListener('change', (ev) => {
    const t = ev.target, what = t.dataset ? t.dataset.input : '';
    if (t.id === 'file-input') { addFiles(Array.from(t.files || []).map((file) => ({ file, name: file.webkitRelativePath || file.name, size: file.size }))); t.value = ''; return; }
    if (t.dataset && t.dataset.check) {
      const c = app.ui.checked;
      if (t.dataset.check === 'all') { if (t.checked) store().pending().forEach((it) => (c[it.ref] = true)); else app.ui.checked = {}; }
      else if (t.checked) c[t.dataset.check] = true; else delete c[t.dataset.check];
      app.render('check');
    } else if (what === 'runon' || what === 'backend') { (app.ui.approve[t.dataset.eid] = app.ui.approve[t.dataset.eid] || {}).backend = t.value; app.render('backend'); }
    else if (what === 'ap-network') { (app.ui.approve[t.dataset.eid] = app.ui.approve[t.dataset.eid] || {}).network = t.checked; app.render('backend'); }
    else if (what === 'opt') { const s = composer(); s[t.dataset.name] = t.value; keepComposer(); if (t.tagName === 'SELECT') app.render('start'); }
    else if (what === 'opt-review') { const s = composer(); s.auto[t.dataset.kind] = !t.checked; keepComposer(); }
    else if (what === 'request-about') { app.ui.request.about = t.checked; }
    else if (what === 'map-round') { app.ui.map.round = t.value === '' ? null : +t.value; mapView.reset(); app.render('map'); }
    else if (what === 'map-lineage') { const m = app.ui.map; m.deep = t.value === 'all'; m.lineageOnly = t.value === 'only'; mapView.reset(); app.render('map'); }
    else if (what === 'paper-template') app.ui.paper.template = t.value;
    else if (what === 'new-page-kind' && app.ui.newPage) { app.ui.newPage.kind = t.value; }
  });

  /* drops: files and folders are attached to the composer */
  let dragDepth = 0;
  document.addEventListener('dragenter', (ev) => { const z = ev.target.closest && ev.target.closest('[data-drop]'); if (z) { dragDepth++; z.classList.add('drag'); } });
  document.addEventListener('dragover', (ev) => { const z = ev.target.closest && ev.target.closest('[data-drop]'); if (z) { ev.preventDefault(); z.classList.add('drag'); } else if (ev.dataTransfer && Array.from(ev.dataTransfer.types || []).indexOf('Files') >= 0) ev.preventDefault(); });
  document.addEventListener('dragleave', (ev) => { const z = ev.target.closest && ev.target.closest('[data-drop]'); if (z && !z.contains(ev.relatedTarget)) { dragDepth = 0; z.classList.remove('drag'); } });
  document.addEventListener('drop', (ev) => {
    const z = ev.target.closest && ev.target.closest('[data-drop]');
    if (ev.dataTransfer && Array.from(ev.dataTransfer.types || []).indexOf('Files') >= 0) ev.preventDefault();   // never navigate away to a dropped file
    if (!z) return;
    dragDepth = 0;
    z.classList.remove('drag');
    if (composer().uploading) return;
    filesOfDrop(ev.dataTransfer).then((files) => { if (files.length) addFiles(files); });
  });

  /* tooltips for anything with data-tip: on hover and on keyboard focus */
  const tipEl = $('#tip');
  const showTip = (t) => {
    if (!t || !t.dataset.tip) { tipEl.classList.remove('on'); return; }
    tipEl.textContent = t.dataset.tip;
    tipEl.classList.add('on');
    const r = t.getBoundingClientRect();
    tipEl.style.left = Math.max(8, Math.min(window.innerWidth - tipEl.offsetWidth - 8, r.left + r.width / 2 - tipEl.offsetWidth / 2)) + 'px';
    tipEl.style.top = (r.top - tipEl.offsetHeight - 6 < 4 ? r.bottom + 6 : r.top - tipEl.offsetHeight - 6) + 'px';
  };
  document.addEventListener('mouseover', (ev) => showTip(ev.target.closest && ev.target.closest('[data-tip]')));
  document.addEventListener('focusin', (ev) => { const t = ev.target.closest && ev.target.closest('[data-tip]'); if (t && t.matches(':focus-visible')) showTip(t); });
  document.addEventListener('focusout', () => tipEl.classList.remove('on'));
  document.addEventListener('scroll', () => tipEl.classList.remove('on'), true);
  document.addEventListener('mousedown', () => tipEl.classList.remove('on'));

  /* ---------- keyboard ---------- */
  function moveInList(dir) {
    const r = app.route, s = store();
    let keys = [];
    if (r.view === 'inbox') keys = s.pending().map((it) => it.page.path).filter((k, i, a) => a.indexOf(k) === i);
    else if (r.view === 'experiments') keys = s.experiments.filter((x) => x.review !== 'rejected').map((x) => x.path);
    else if (r.view === 'wiki') keys = [].concat(...STAGES.concat(EXTRA_KINDS.filter((k) => k !== 'lesson' || app.ui.showLessons)).map((k) => s.pages.filter((x) => x.kind === k).map((x) => x.path)));
    else return false;
    const cur = r.view === 'inbox' ? (currentItem(s, r) || { page: {} }).page.path : r.sel;
    const i = keys.indexOf(cur), next = keys[i < 0 ? 0 : Math.max(0, Math.min(keys.length - 1, i + dir))];
    if (next) select(next);
    return true;
  }
  document.addEventListener('keydown', (ev) => {
    const r = app.route, s = store();
    const target = ev.target;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) && target.type !== 'checkbox' && target.type !== 'radio';
    trapTab(ev);
    if (ev.key === 'Escape') {
      if (overlayOpen()) { closeOverlay(); return; }
      if (app.ui.rejecting) { app.ui.rejecting = null; app.render('ui'); return; }
      if (app.ui.confirmDelete) { app.ui.confirmDelete = null; app.render('ui'); return; }
      if (typing) { target.blur(); return; }
      if (r.name === 'project' && r.sub) { go({ sub: '' }); return; }
      if (r.name === 'project' && r.sel && r.view === 'map') select('');
      return;
    }
    const mod = ev.metaKey || ev.ctrlKey;
    if (mod && ev.key.toLowerCase() === 's') {
      const code = $('form[data-form=code]'), edit = $('form[data-form=edit]');
      if (code || edit) { ev.preventDefault(); if (code && (!edit || code.contains(target))) saveCode(code); else saveEdit(edit.dataset.ref); }
      return;
    }
    if (mod && ev.key.toLowerCase() === 'k') { ev.preventDefault(); if (overlayOpen()) closeOverlay(); else openPalette(); return; }
    if (typing || mod || ev.altKey || overlayOpen()) return;
    const k = ev.key;
    if (k === '/') { ev.preventDefault(); openPalette(); return; }
    if (k === '?') { openHelp(); return; }
    if (k === 't' || k === 'T') { theme.toggle(); return; }
    if (k === 'z' || k === 'Z') { undo(); return; }
    if (r.name !== 'project' || !s || !s.loaded) return;
    if (target.closest && target.closest('.n-btn')) return;
    if (/^[1-6]$/.test(k)) { const view = VIEWS[+k - 1][0]; go({ view, sub: '', sel: view === 'paper' ? '' : r.sel }); return; }
    if (k === 'A' && r.view === 'inbox' && Object.keys(app.ui.checked || {}).length) { bulk('accept'); return; }
    const tgt = reviewTarget();
    if ((k === 'a' || k === 'A') && tgt && !r.sub) { decide(tgt.ref, 'accept'); return; }
    if ((k === 'r' || k === 'R') && tgt && !r.sub) { ev.preventDefault(); app.ui.rejecting = tgt.ref; app.render('ui'); return; }
    const cur = r.view === 'inbox' ? (tgt && tgt.page) : s.byPath.get(r.sel);
    if ((k === 'e' || k === 'E') && cur && editable(cur) && r.view !== 'experiments' && r.view !== 'overview') { if (r.view === 'inbox' && r.sel !== cur.path) go({ sel: cur.path, sub: 'edit' }); else go({ sub: r.sub === 'edit' ? '' : 'edit' }); return; }
    if ((k === 'h' || k === 'H') && cur && r.view !== 'experiments' && r.view !== 'overview') { if (r.sel !== cur.path) go({ sel: cur.path, sub: 'history' }); else go({ sub: r.sub === 'history' ? '' : 'history' }); return; }
    if (k === 'l' || k === 'L') { app.ui.logOpen = !app.ui.logOpen; app.render('ui'); return; }
    if ((k === 'x' || k === 'X') && r.view === 'inbox' && tgt) { const c = app.ui.checked; if (c[tgt.ref]) delete c[tgt.ref]; else c[tgt.ref] = true; app.render('check'); return; }
    if (k === 'j' || k === 'k' || k === 'J' || k === 'K') { const d = k.toLowerCase() === 'j' ? 1 : -1; if (r.view === 'map') mapView.key({ key: d > 0 ? 'ArrowDown' : 'ArrowUp' }); else moveInList(d); return; }
    if (r.view === 'map' && mapView.key(ev)) ev.preventDefault();
  });

  /* decisions still in their grace period go out when the page is left */
  window.addEventListener('pagehide', () => { const s = store(); if (s) { const waiting = s.unsent(); if (waiting.length) s.send(waiting, { keepalive: true }).catch(() => {}); } });
  /* nothing typed is lost without a word */
  window.addEventListener('beforeunload', (ev) => { const s = composer(); if (s.uploading) { ev.preventDefault(); ev.returnValue = ''; } });

  connection.on(() => app.render('connection'));
}
