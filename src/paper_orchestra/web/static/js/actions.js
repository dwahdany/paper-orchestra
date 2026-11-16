/* What the person can do: one table of actions (the `data-act` attributes of the markup),
   one handler each for clicks, submits, typing, changes, drops and keys. */

import { VIEWS, app, codeKey, go, keepRested, link, reasonsKey, requestForKey, requestKey, select } from './core.js';
import { api, connection, Unreachable } from './api.js';
import { $, $$, ago, bytes, cap, cutoffFrom, cutoffParts, cutoffWords, lineDiff, plain, plural, storage } from './util.js';
import { STAGES, EXTRA_KINDS, PHASE_DO, ROLE, agentName, phaseName, roleName } from './vocab.js';
import { dropDraft, isDirty, label, schedulePreview, touchDraft } from './page.js';
import { mapView } from './map.js';
import { currentItem } from './inbox.js';
import { EXAMPLE, clearComposer, composer, composerReady, composerSettings, keepComposer, loadDoctor, loadLibrary, totalSize } from './home.js';
import { clearToast, closeOverlay, openHelp, openNewPage, openPalette, overlayOpen, renderToast, say, sayError, trapTab } from './overlays.js';
import { dismissNotices, theme } from './shell.js';

const GRACE_MS = 5000;
/** How the page moves to something: at once for a person who asked for less motion. */
const glide = () => (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');
/** The reasons being typed are kept on this device until the rejection they belong to is made. */
function keepReasons() {
  if (!app.store) return;
  const typed = {};
  Object.keys(app.ui.reasons).forEach((ref) => { if (String(app.ui.reasons[ref] || '').trim()) typed[ref] = app.ui.reasons[ref]; });
  if (Object.keys(typed).length) storage.set(reasonsKey(app.store.pid), typed); else storage.remove(reasonsKey(app.store.pid));
}
/** The instruction being typed is kept on this device until it is sent, and whom it is for
    with it: a reload must not turn an instruction for the scout into one for everyone. */
function keepRequest() {
  if (!app.store) return;
  const rq = app.ui.request, pid = app.store.pid;
  if (!rq.text.trim()) { storage.remove(requestKey(pid)); storage.remove(requestForKey(pid)); return; }
  storage.set(requestKey(pid), rq.text);
  if (rq.for && rq.for !== 'all') storage.set(requestForKey(pid), rq.for); else storage.remove(requestForKey(pid));
}
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
    out.results.forEach((result, i) => { if (result.ok && result.worker_error) sayError(cap(entries[i].label) + ' is accepted and approved, but the run could not start now: ' + plain(result.worker_error.message) + ' It runs when you continue.'); });
    out.failed.forEach((failure) => { const page = failure.entry.item && failure.entry.item.page; if (page && page.kind === 'experiment' && page.id) s.invalidateExperiment(page.id); });   // its code may be what changed: read it again
    /* an experiment was rejected: the server says what rests on it (the findings based on its
       runs, the verdicts they carry). Kept for its page, which lists it with links, and said. */
    out.results.forEach((result, i) => {
      const deps = result.ok && result.dependents, entry = entries[i];
      if (!deps || !entry) return;
      keepRested(s, entry.target, deps);
      if (!out.failed.length && restCount(deps)) say(restSentence(entry.label, deps), { ms: 12000, action: { href: link({ view: 'experiments', sel: entry.target, sub: '' }), label: 'Show' } });
    });
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
/* The decisions still in their grace period, oldest first, one entry per toast. Undo takes
   back the newest; the one before it is then offered, for as long as its own period lasts. */
const waiting = [];
function later(entries, decision) {
  const batch = { entries, decision };
  waiting.push(batch);
  setTimeout(() => { forget(batch); flush(entries); }, GRACE_MS);
  return batch;
}
function forget(batch) { const i = waiting.indexOf(batch); if (i >= 0) waiting.splice(i, 1); }
const stillOpen = (batch) => batch.entries.some((entry) => !entry.sent && !entry.cancelled);
/** What the toast adds while earlier decisions can still be taken back as well. */
function earlier() {
  const n = waiting.filter(stillOpen).length - 1;
  return n > 0 ? (n === 1 ? 'The decision before it' : 'The ' + n + ' before it') + ' can still be taken back too.' : '';
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
  keepReasons();
  delete app.ui.checked[ref];
  if (options.immediate) { app.render('decide'); return flush([entry]); }
  later([entry], decision);
  say(describe([entry], decision), { undo: [entry], note: earlier() });
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
  keepReasons();
  later(entries, decision);
  say(describe(entries, decision), { undo: entries, note: earlier() });
  if (app.route.view === 'inbox') go({ sel: '', sub: '' }, true); else app.render('bulk');
}
function undo() {
  if (!store()) return false;
  /* the newest decision that has not been sent: the one the toast names, or the one before */
  let batch = null;
  for (let i = waiting.length - 1; i >= 0 && !batch; i--) if (stillOpen(waiting[i])) batch = waiting[i];
  if (!batch) return false;
  const n = store().retract(batch.entries);
  forget(batch);
  clearToast();
  if (!n) return false;
  app.ui.flash = null;
  const took = n === 1 ? 'Taken back: ' + batch.entries[0].label + '.' : 'Taken back: ' + n + ' decisions.';
  const before = waiting.filter(stillOpen).pop();
  if (before) say(took + ' ' + describe(before.entries, before.decision), { undo: before.entries });
  else say(took + ' Nothing was sent.');
  app.render('undo');
  return true;
}

/** How much rests on a rejected experiment, as the server counted it. */
const restCount = (deps) => ((deps || {}).findings || []).length + ((deps || {}).hypotheses || []).length;
/** "E1 is rejected. 2 findings and 2 verdicts rest on its runs and were not changed: its page lists them." */
function restSentence(name, deps) {
  const findings = (deps.findings || []).length, verdicts = (deps.hypotheses || []).length, one = findings + verdicts === 1;
  return cap(name) + ' is rejected. ' + [findings ? plural(findings, 'finding') : '', verdicts ? plural(verdicts, 'verdict on a hypothesis', 'verdicts on hypotheses') : ''].filter(Boolean).join(' and ')
    + (one ? ' rests' : ' rest') + ' on its runs and ' + (one ? 'was' : 'were') + ' not changed: its page lists ' + (one ? 'it' : 'them') + '.';
}

/** Reject a page that was accepted before. There is nothing optimistic about it and no
    short undo: the server is asked at once, and for an experiment it answers with what
    rests on its runs, which is said with the confirmation. */
async function rejectAccepted(path, reason) {
  const s = store(), page = s.byPath.get(path);
  if (!page) return;
  app.ui.rejecting = null;
  delete app.ui.reasons[path];
  keepReasons();
  try {
    const answer = await api.review(s.pid, [{ ref: path, decision: 'reject', reason: reason || null, base_updated: page.updated }]);
    const result = (answer.results || [])[0] || {};
    s.invalidate(path);
    if (page.kind === 'experiment' && page.id) s.invalidateExperiment(page.id);
    await s.refreshNow();
    if (!result.ok) sayError(result.status === 409 ? cap(label(page)) + ' changed after you read it, so it was not rejected. It is shown as it is now.' : cap(label(page)) + ' was not rejected: ' + plain((result.error && result.error.message) || 'the server refused it.'));
    else {
      const deps = result.dependents;
      if (deps) keepRested(s, path, deps);
      /* the page in front of the person now says that it is rejected, and lists what rests on it with links */
      say(restCount(deps) ? restSentence(label(page), deps) : 'Rejected ' + label(page) + ' after all. It is kept with your reason.', { ms: 12000 });
    }
  } catch (error) { sayError(cap(label(page)) + ' was not rejected: ' + message(error)); }
  app.render('decided');
}

/* ---------- editing a page ---------- */
const draftKey = (path) => 'po-draft:' + store().pid + ':' + path;
/** Save what is in the editor. Editing is not deciding: a proposal stays a proposal and a
    rejected page stays rejected, unless `accept` says that the save is also the decision
    ("Save and accept"). The toast says which of the two was done. */
async function saveEdit(path, accept) {
  const s = store(), page = s.byPath.get(path), draft = app.ui.drafts[draftKey(path)];
  if (!page || !draft || draft.saving) return;
  if (!draft.title.trim()) { draft.error = 'A page needs a title.'; app.render('edit'); return; }
  draft.saving = true; draft.error = null; draft.accepting = !!accept;
  app.render('saving');
  const fields = { summary: draft.summary };
  if (draft.state && draft.state !== (draft.base.state || '')) fields.state = draft.state;
  try {
    if (draft.proposal) {
      /* a suggested edit has no state of its own to save into: the person's version of it is applied, which decides it */
      const entry = { ref: draft.proposal, decision: 'accept', edits: { body: draft.body, fields: Object.assign({ title: draft.title }, fields) }, base_updated: null, target: path, label: label(page), sent: false };
      const out = await s.send([entry]);
      if (out.failed.length) {
        const result = out.failed[0].result;
        if (result.status === 409) { const view = await api.page(s.pid, path); draft.conflict = view.page; }
        else draft.error = plain((result.error && result.error.message) || 'The server refused it.');
        return;
      }
      dropDraft(s, path);
      go({ sub: '' }, true);
      say('Saved your version of the change to ' + label(page) + ', and accepted it.');
    } else {
      const body = { title: draft.title, body: draft.body, fields, base_updated: draft.base_updated };
      if (accept) body.accept = true;
      const view = await api.putPage(s.pid, path, body);
      const now = (view && view.page && view.page.review) || page.review;
      dropDraft(s, path);
      s.invalidate(path);
      await s.refreshNow();
      go({ sub: '' }, true);
      say(now === 'proposed' ? 'Saved. ' + label(page) + ' is still a proposal: it waits for your decision.'
        : now === 'rejected' ? 'Saved. ' + label(page) + ' stays rejected.'
          : accept ? 'Saved your version and accepted ' + label(page) + '.'
            : 'Saved. The previous version is in the history.');
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
/** Start the agents. Without `phase` the engine goes on by itself (what is approved runs
    first); with it, the phase the person chose is run as chosen. */
async function advance(phase) {
  const s = store();
  app.ui.goMenu = null;
  await flushAll();
  try {
    const answer = await api.advance(s.pid, phase ? { phase } : {});
    if (answer.worker) s.project.worker = answer.worker;
    if (app.ui.toast && app.ui.toast.kind !== 'error' && !app.ui.toast.undo) clearToast();   // "Stopping…" is no longer true
    if (app.route.view !== 'map' && app.route.view !== 'overview') go({ view: 'map', sel: '', sub: '' });
    s.refresh();
    app.render('advance');
  } catch (error) {
    sayError(error.status === 409 ? error.message : 'The agents were not started: ' + message(error));
    s.refresh();
    app.render('advance');
  }
}
/** Clear the name of a worker the server cannot verify. Nothing is signalled. */
async function forgetWorker() {
  const s = store();
  if (app.ui.forgetting) return;
  app.ui.forgetting = true;
  app.render('forget');
  try {
    const answer = await api.forgetWorker(s.pid);
    if (answer.worker) s.project.worker = answer.worker;
    await s.refreshNow();
    const still = s.project.worker && s.project.worker.state !== 'idle';
    say(still ? 'The worker was not forgotten: it is still named in the project\'s files.' : 'Forgotten. Nothing was signalled or stopped; the project can be continued. If a worker is in fact at work, it shows again.');
  } catch (error) { sayError('Not forgotten: ' + message(error)); s.refresh(); }
  app.ui.forgetting = false;
  app.render('forget');
  /* the button is gone: the keyboard goes to what can be done now */
  const next = $('.ov-actions [data-act="continue"], .ov-actions [data-go-main], .strip [data-act="continue"], .strip [data-go-main], [data-act="forget-worker"]');
  if (next && (!document.activeElement || document.activeElement === document.body)) next.focus({ preventScroll: true });
}
async function stopAgents() {
  const s = store();
  try {
    const answer = await api.stop(s.pid);
    if (answer.worker) s.project.worker = answer.worker;
    const kind = (answer.worker || {}).kind;
    say(answer.worker && answer.worker.state === 'idle' ? 'Nothing was running.'
      : kind === 'run' ? 'Stopping the run. The experiment is queued again: it runs when you continue, or now.'
      : kind === 'write' ? 'Stopping the write-up. What it has written is kept; it can be started again.'
      : 'Stopping. What was written so far is kept; the phase can be resumed.');
    s.refresh();
    app.render('stop');
  } catch (error) { sayError('Not stopped: ' + message(error)); }
}

/* ---------- experiments ---------- */
/** The plan of an experiment as the server last read it, or null when it has not been fetched. */
const planOf = (eid) => { const entry = store().exps.get(eid); return (entry && entry.data && entry.data.plan) || null; };
/** What an approval says. Network access is never taken from the plan: the request says
    yes or no itself. For a plan that asks for the network the answer is the person's own
    (`chosen.network`); until they have given it the field is left out, and nothing is sent. */
function approvalBody(eid, start) {
  const s = store(), row = s.expById.get(eid), entry = s.exps.get(eid), chosen = app.ui.approve[eid] || {};
  const plan = planOf(eid);
  const body = { start: !!start };
  if (chosen.backend) body.backend = chosen.backend;
  if (chosen.gpu != null) body.gpu = String(chosen.gpu).trim() || null;
  if (chosen.minutes != null && chosen.minutes !== '' && !Number.isNaN(+chosen.minutes)) body.timeout_seconds = Math.round(+chosen.minutes * 60);
  if (chosen.network === true || chosen.network === false) body.network = chosen.network;
  else if (plan && !plan.network) body.network = false;
  if (row) body.base_updated = row.updated;
  const digest = entry && entry.data && entry.data.code && entry.data.code.digest;
  if (digest) body.code_digest = digest;
  return body;
}
/** The plan asks for the network and the person has not said yes or no: the question is
    shown (the approval form is opened if it is not), and nothing is sent. Never a guess. */
function networkUnanswered(eid, body) {
  const plan = planOf(eid);
  if (!plan || !plan.network || body.network === true || body.network === false) return false;
  const row = store().expById.get(eid);
  app.ui.netNudge = eid;
  if (app.route.view === 'experiments' && row && row.review === 'accepted' && row.state && row.state !== 'planned') app.ui.reapprove = eid;
  sayError('The plan of ' + eid + ' asks for network access. Say whether the run may have it, then approve.');
  app.render('ui');
  setTimeout(() => { const q = $('.net-ask'); if (!q) return; q.scrollIntoView({ block: 'center', behavior: glide() }); const first = $('input', q); if (first) first.focus({ preventScroll: true }); }, 0);
  return true;
}
async function approve(eid, start, overrides) {
  const s = store();
  await flushAll();
  try {
    const body = Object.assign(approvalBody(eid, start), overrides || {});
    if (networkUnanswered(eid, body)) return;
    const answer = await api.approve(s.pid, eid, body);
    app.ui.reapprove = null;
    app.ui.netNudge = null;
    if (app.ui.approve[eid]) delete app.ui.approve[eid].network;      // an answer about the network is given per approval, not once for good
    delete app.ui.runOpen[eid];
    if (answer.worker) s.project.worker = answer.worker;
    s.invalidateExperiment(eid);
    await s.refreshNow();
    if (answer.worker_error) sayError(eid + ' is approved, but the run could not start now: ' + plain(answer.worker_error.message) + ' It runs when you continue.');
    else say(start ? eid + ' is approved; its run is starting.' : eid + ' is approved. It runs first when you continue, or now.');
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
  if (networkUnanswered(page.id, body)) return;
  const run = { start: true };
  ['backend', 'gpu', 'timeout_seconds', 'network', 'code_digest'].forEach((k) => { if (body[k] != null) run[k] = body[k]; });
  const done = decide(path, 'accept', { approve_run: run, immediate: true });
  if (done && done.then) done.then((out) => { if (out && !out.failed.length) { if (app.ui.approve[page.id]) delete app.ui.approve[page.id].network; s.invalidateExperiment(page.id); say(label(page) + ' is accepted; its run is starting.'); if (app.route.view !== 'experiments') go({ view: 'experiments', sel: path, sub: '' }); } });
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
  const cutoff = cutoffFrom(s.cutoffYear, s.cutoffMonth);
  if (cutoff.error) { s.error = cutoff.error; s.adv = true; keepComposer(); app.render('start'); return; }
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
    /* a file that was given another name in inputs/ (two of one name; a name the project keeps for itself) */
    const renamed = (project.uploads || []).filter((u) => u && u.sent && u.stored && u.stored !== u.sent).map((u) => u.sent + ' is kept as ' + u.stored);
    const also = (skipped ? ' ' + plural(skipped, 'file') + ' could not be attached.' : '') + (renamed.length ? ' ' + renamed.slice(0, 3).join('; ') + (renamed.length > 3 ? '; and ' + (renamed.length - 3) + ' more were renamed' : '') + '.' : '');
    if (project.worker_error) sayError('The project is created, but the agents could not be started: ' + plain(project.worker_error.message) + ' Start them from here once the cause is put right.' + also);
    else say('The project is created' + (project.worker ? ' and research has started.' : '.') + also);
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
  'save-accept': (el) => saveEdit(el.dataset.ref || app.route.sel, true),
  'conflict-mine': () => { const d = app.ui.drafts[draftKey(app.route.sel)]; if (!d || !d.conflict) return; d.base_updated = d.conflict.updated; d.conflict = null; saveEdit(app.route.sel, !!d.accepting); },
  'conflict-rebase': () => { const d = app.ui.drafts[draftKey(app.route.sel)]; if (!d || !d.conflict) return; const c = d.conflict; d.theirs = { rows: lineDiff(d.base.body, c.body || ''), by: (c.meta || {}).updated_by || c.author || '', at: c.updated }; d.base = { title: c.title, summary: c.summary || '', state: c.state || '', body: c.body || '' }; d.base_updated = c.updated; d.conflict = null; touchDraft(store(), app.route.sel, 'body', d.body); say('You are now editing on top of the newer version. What it changed stays above the text, so that you can carry it over: saving replaces that version with your text.'); app.render('conflict'); },
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
      s.forget(path);
      if (app.route.sel === path) go({ sel: '', sub: '' });
      await s.refreshNow();
      say('Deleted ' + (page ? label(page) : 'the page') + '. Its last version is kept under history/.');
    } catch (error) { app.ui.confirmDelete = null; sayError('Not deleted: ' + message(error)); app.render('delete'); }
  },
  'reload-page': (el) => { store().invalidate(el.dataset.ref); app.render('reload'); },
  'reload-experiment': (el) => { store().invalidateExperiment(el.dataset.eid); app.render('reload'); },
  'toggle-diff': (el) => { app.ui.asDiff[el.dataset.ref] = !app.ui.asDiff[el.dataset.ref]; app.render('diff'); },
  'jump-change': () => { const el = $('.prose-marked .diff-changed, .prose-marked ins, .prose-marked del'); if (el) { el.scrollIntoView({ block: 'center', behavior: glide() }); el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 1600); } },
  'close-panel': () => select(''),
  continue: () => advance(),
  /* the person's own choice of the phase that runs next: more research, above all */
  'continue-phase': (el) => { const phase = el.dataset.phase; if (PHASE_DO[phase]) advance(phase); },
  'go-menu': (el) => {
    const where = el.dataset.where || 'strip';
    app.ui.goMenu = app.ui.goMenu === where ? null : where;
    app.render('ui');
    if (app.ui.goMenu) setTimeout(() => { const first = $('.go.open .go-i'); if (first) first.focus(); }, 0);
  },
  'forget-worker': () => forgetWorker(),
  'notices-seen': () => { dismissNotices(store()); app.render('ui'); },
  'show-problems': () => { if (app.route.view !== 'overview') go({ view: 'overview', sel: '', sub: '' }); setTimeout(() => { const el = $('#problems'); if (el) el.scrollIntoView({ block: 'start', behavior: glide() }); }, 60); },
  'cutoff-open': () => {
    if (app.ui.cutoffEdit) { app.ui.cutoffEdit = null; return; }
    const parts = cutoffParts(((store().project.settings || {}).research_cutoff));
    app.ui.cutoffEdit = { year: parts && parts.raw == null ? parts.year : '', month: parts && parts.raw == null ? parts.month : 0, error: null, saving: false };
    app.render('ui');
  },
  'cutoff-none': () => saveCutoff(null),
  stop: () => stopAgents(),
  'limit-edit': () => { const limits = (store().project.settings || {}).limits || {}; app.ui.limitEdit = { value: limits.model_usd > 0 ? String(Math.ceil(limits.model_usd * 1.5)) : '' }; app.render('limit'); setTimeout(() => { const el = $('#limit-usd'); if (el) el.select(); }, 0); },
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
  'rq-del': async (el) => {
    const s = store();
    try { await api.deleteRequest(s.pid, el.dataset.ref); await s.refreshNow(); say('The instruction is withdrawn. No further phase is given it.'); }
    catch (error) {
      /* withdrawn already (in another window: that writes no event): the list is read again, and the row goes */
      await s.refreshNow();
      if (error.status === 404 && !s.requests.some((r) => r.id === el.dataset.ref)) say('That instruction was withdrawn already. No further phase is given it.');
      else sayError('Not withdrawn: ' + message(error));
    }
    app.render('request');
  },
  'feed-more': async () => { const s = store(); app.ui.feedLimit += 60; if (s.activity.length < app.ui.feedLimit && s.moreActivity) { try { await s.loadEarlierActivity(); } catch (error) { sayError(message(error)); } } app.render('ui'); },
  'new-page': () => openNewPage(),
  'show-lessons': () => { app.ui.showLessons = true; app.render('ui'); },
  'bulk-accept': () => bulk('accept'),
  'bulk-reject': () => { app.ui.rejecting = app.ui.rejecting === 'bulk' ? null : 'bulk'; app.render('ui'); },
  approve: (el) => approve(el.dataset.eid, !!el.dataset.start),
  reapprove: (el) => { app.ui.reapprove = el.dataset.eid || null; app.render('ui'); if (app.ui.reapprove) setTimeout(() => { const f = $('#approve'); if (f) f.scrollIntoView({ block: 'start', behavior: glide() }); }, 0); },
  /* what the button offers becomes the choice of the approval form, so that a question still
     to be answered (network access) is asked with that choice in place */
  rerun: (el) => { const eid = el.dataset.eid; const chosen = {}; if (el.dataset.backend) chosen.backend = el.dataset.backend; if (el.dataset.minutes) chosen.minutes = String(+el.dataset.minutes); const row = store().expById.get(eid); app.ui.approve[eid] = chosen; approve(eid, true, { base_updated: row ? row.updated : undefined }); },
  'run-now': async (el) => {
    const s = store(), eid = el.dataset.eid;
    if (app.ui.starting[eid]) return;
    const again = !!(s.expById.get(eid) || {}).interrupted;
    /* for a run that was interrupted the server first settles what is left of it, which can take some seconds */
    app.ui.starting[eid] = true;
    app.render('run');
    await flushAll();
    try { const answer = await api.run(s.pid, eid); if (answer.worker) s.project.worker = answer.worker; delete app.ui.runOpen[eid]; s.invalidateExperiment(eid); await s.refreshNow(); say(again ? eid + ' is run again. What was left of the interrupted run is settled first.' : 'The run of ' + eid + ' is starting.'); }
    catch (error) { sayError(eid + ' was not started: ' + message(error)); s.invalidateExperiment(eid); s.refresh(); }
    delete app.ui.starting[eid];
    app.render('run');
  },
  'cancel-run': async (el) => {
    const s = store(), eid = el.dataset.eid;
    const row = s.expById.get(eid) || {};
    const waiting = ['approved', 'queued'].indexOf(row.state) >= 0;   // approved, not started: the approval is taken back
    try { await api.cancel(s.pid, eid); say(waiting ? eid + ' will not run: its approval is taken back. Approve it again when you want it run.' : row.interrupted ? 'The interrupted run of ' + eid + ' is cancelled. Approve it again when you want it run.' : 'Cancelling the run of ' + eid + '. It is recorded as cancelled.'); s.invalidateExperiment(eid); s.refresh(); }
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
    if (!app.ui.codeEdit[eid + '/' + name]) app.ui.codeEdit[eid + '/' + name] = { text: file.text, error: null, saving: false };
    app.render('ui');
    setTimeout(() => { const t = $('#code-text'); if (t) { t.focus(); t.scrollIntoView({ block: 'nearest' }); } }, 0);
  },
  'code-cancel': (el) => { delete app.ui.codeEdit[el.dataset.key]; if (store()) storage.remove(codeKey(store().pid, el.dataset.key)); app.render('ui'); },
};
/** Run an action by name (the palette does). */
app.act = (name, el) => { if (ACT[name]) return ACT[name](el || { dataset: {} }); return undefined; };

/** Change the project's research cutoff: "2024-06", "2024", or `null` for none. */
async function saveCutoff(value) {
  const s = store(), ui = app.ui.cutoffEdit || (app.ui.cutoffEdit = { year: '', month: 0, error: null, saving: false });
  if (ui.saving) return;
  ui.saving = true; ui.error = null;
  app.render('cutoff');
  try {
    await api.patchProject(s.pid, { settings: { research_cutoff: value } });
    app.ui.cutoffEdit = null;
    await s.refreshNow();
    const parts = cutoffParts(value);
    say(parts ? 'The research cutoff is set: only literature published before ' + cutoffWords(parts) + ' is in scope.' : 'The project has no research cutoff now.');
  } catch (error) { ui.error = message(error); ui.saving = false; }
  app.render('cutoff');
}

async function saveCode(form) {
  const s = store(), eid = form.dataset.eid, name = form.dataset.name, key = eid + '/' + name, edit = app.ui.codeEdit[key];
  if (!edit || edit.saving) return;
  edit.saving = true; edit.error = null;
  app.render('saving');
  try {
    const answer = await api.putCode(s.pid, eid, name, edit.text);
    delete app.ui.codeEdit[key];
    storage.remove(codeKey(s.pid, key));
    s.invalidateExperiment(eid);
    await s.refreshNow();
    const problems = answer.problems || [];
    if (problems.length) sayError('Saved ' + name + '. The check found: ' + problems.join(' ')); else say('Saved ' + name + '. It parses; the earlier version is kept.');
  } catch (error) { edit.error = message(error); }
  edit.saving = false;
  app.render('saved');
}

/* What a second click of a double click may do again. Everything else is done once: after
   the first click the page has changed under the pointer (the next proposal has moved up
   into the place of the accepted one, a form has opened where the button was), so a second
   click would decide something the person never looked at. */
const REPEATABLE = { 'zoom-in': 1, 'zoom-out': 1, 'feed-more': 1, 'code-tab': 1 };

export function installActions() {
  document.addEventListener('click', (ev) => {
    if (ev.detail > 1) {
      const again = ev.target.closest('[data-act], button[type=submit], input[type=submit]');
      if (again && !REPEATABLE[again.dataset.act]) { ev.preventDefault(); return; }
    }
    /* the list of phases beside Continue closes when something else is pressed */
    if (app.ui.goMenu && !(ev.target.closest && ev.target.closest('.go'))) { app.ui.goMenu = null; app.render('ui'); }
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
    if (kind === 'reject') {
      const ref = form.dataset.ref, reason = (form.elements.reason.value || '').trim();
      /* a proposal is decided with the short undo; a page that was accepted before is rejected at once */
      if (s.pending().some((it) => it.ref === ref)) decide(ref, 'reject', { reason }); else rejectAccepted(ref, reason);
    }
    else if (kind === 'bulk-reject') bulk('reject', (form.elements.reason.value || '').trim());
    else if (kind === 'edit') saveEdit(form.dataset.ref, false);
    else if (kind === 'cutoff') {
      const ui = app.ui.cutoffEdit;
      if (!ui) return;
      const made = cutoffFrom(ui.year, ui.month);
      if (made.error) { ui.error = made.error; app.render('cutoff'); return; }
      saveCutoff(made.value);
    }
    else if (kind === 'code') saveCode(form);
    else if (kind === 'start') startProject();
    else if (kind === 'request') {
      const rq = app.ui.request, text = rq.text.trim();
      if (!text || rq.sending) return;
      const target = rq.about && app.route.sel && s.byPath.has(app.route.sel) ? app.route.sel : null;
      const role = rq.for && rq.for !== 'all' ? rq.for : null;
      rq.sending = true; rq.error = null;
      app.render('request');
      api.addRequest(s.pid, text, target, role).then(async (answer) => {
        const sent = (answer && answer.request && answer.request.for) || role || 'all';
        const phase = (ROLE[sent] || {}).phase;
        rq.text = ''; rq.for = 'all';
        storage.remove(requestKey(s.pid));
        storage.remove(requestForKey(s.pid));
        await s.refreshNow();
        say(!phase ? 'Added for everyone: every phase that runs is given it, until the analysis that ends this round.'
          : phase === 'research' ? 'Added for the scout. It waits for a research phase: choose “Search the literature again” beside Continue.'
            : 'Added for ' + roleName(sent) + ': it is given to the next ' + (phase === 'analyze' ? 'analysis' : phaseName(phase) + ' phase') + '.');
      }, (error) => { rq.error = message(error); })
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
    } else if (what === 'opt') { const s = composer(); s[t.dataset.name] = t.value; keepComposer(); if (t.dataset.name === 'model_usd' || t.dataset.name === 'cutoffYear' || s.error) { s.error = null; app.render('start'); } }
    else if (what === 'reason') { app.ui.reasons[t.dataset.ref] = t.value; keepReasons(); }
    else if (what === 'request') { app.ui.request.text = t.value; keepRequest(); }
    else if (what === 'code') { const edit = app.ui.codeEdit[t.dataset.key]; if (edit) { edit.text = t.value; edit.restored = false; if (store()) storage.set(codeKey(store().pid, t.dataset.key), t.value); } }
    /* drawn again while typing, never on blur: a render between the press and the release of a
       button would swallow the click that follows the typing */
    else if (what === 'ap-minutes') { (app.ui.approve[t.dataset.eid] = app.ui.approve[t.dataset.eid] || {}).minutes = t.value; app.render('backend'); }
    else if (what === 'ap-gpu') { (app.ui.approve[t.dataset.eid] = app.ui.approve[t.dataset.eid] || {}).gpu = t.value; app.render('backend'); }
    else if (what === 'paper-model') app.ui.paper.model = t.value;
    else if (what === 'limits' && app.ui.limits) app.ui.limits[t.dataset.name] = t.value;
    else if (what === 'cutoff' && app.ui.cutoffEdit) app.ui.cutoffEdit[t.dataset.name] = t.value;
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
    else if (what === 'ap-net-choice') { (app.ui.approve[t.dataset.eid] = app.ui.approve[t.dataset.eid] || {}).network = t.value === 'yes'; if (app.ui.netNudge === t.dataset.eid) app.ui.netNudge = null; app.render('backend'); }
    else if (what === 'opt') { const s = composer(); s[t.dataset.name] = t.value; keepComposer(); if (t.tagName === 'SELECT') app.render('start'); }
    else if (what === 'opt-review') { const s = composer(); s.auto[t.dataset.kind] = !t.checked; keepComposer(); }
    else if (what === 'request-about') { app.ui.request.about = t.checked; }
    else if (what === 'request-for') { app.ui.request.for = t.value; keepRequest(); app.render('request'); }
    else if (what === 'cutoff' && app.ui.cutoffEdit) { app.ui.cutoffEdit[t.dataset.name] = t.value; }
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

  /* A <select> whose list is open: the render loop leaves the page alone until the choice
     is made or given up (the list would close under the pointer if its markup were
     written again while the agents are at work). */
  const isSelect = (el) => !!el && el.tagName === 'SELECT';
  document.addEventListener('mousedown', (ev) => { if (isSelect(ev.target) || isSelect(ev.target.parentElement)) app.choosingAt = Date.now(); }, true);
  document.addEventListener('keydown', (ev) => { if (isSelect(ev.target) && (ev.key === ' ' || ev.key === 'F4' || (ev.altKey && (ev.key === 'ArrowDown' || ev.key === 'ArrowUp')))) app.choosingAt = Date.now(); else if (isSelect(ev.target) && ev.key === 'Escape') app.choosingAt = 0; }, true);
  document.addEventListener('change', (ev) => { if (isSelect(ev.target)) app.choosingAt = 0; }, true);
  document.addEventListener('focusout', (ev) => { if (isSelect(ev.target)) app.choosingAt = 0; }, true);

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
    if (app.ui.goMenu && ev.key === 'Tab') {
      /* the list does not stay open behind a keyboard that has left it: it closes, and Tab
         moves on from the button that opened it (the key's own action is left to the browser) */
      app.ui.goMenu = null;
      app.render('ui');
      const more = $('.go .go-more');
      if (more) more.focus();
      return;
    }
    if (app.ui.goMenu && (ev.key === 'ArrowDown' || ev.key === 'ArrowUp')) {
      const items = $$('.go.open .go-i');
      if (items.length) { const i = items.indexOf(document.activeElement); items[(i + (ev.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length].focus(); ev.preventDefault(); }
      return;
    }
    if (ev.key === 'Escape') {
      if (overlayOpen()) { closeOverlay(); return; }
      if (app.ui.goMenu) { app.ui.goMenu = null; app.render('ui'); const more = $('.go .go-more'); if (more) more.focus(); return; }
      if (app.ui.rejecting) {
        const ref = app.ui.rejecting;
        app.ui.rejecting = null;
        app.render('ui');
        /* the keyboard goes back to what opened the question, not to the top of the page */
        const asked = $$('[data-act="reject-ask"]').find((el) => el.dataset.ref === ref && el.offsetParent !== null);
        if (asked) asked.focus();
        return;
      }
      if (app.ui.confirmDelete) { app.ui.confirmDelete = null; app.render('ui'); return; }
      if (typing) { target.blur(); return; }
      if (r.name === 'project' && r.sub) { go({ sub: '' }); return; }
      if (r.name === 'project' && r.sel && r.view === 'map') select('');
      return;
    }
    const mod = ev.metaKey || ev.ctrlKey;
    if (mod && ev.key.toLowerCase() === 's') {
      const code = $('form[data-form=code]'), edit = $('form[data-form=edit]');
      if (code || edit) { ev.preventDefault(); if (code && (!edit || code.contains(target))) saveCode(code); else saveEdit(edit.dataset.ref, false); }
      return;
    }
    if (mod && ev.key.toLowerCase() === 'k') { ev.preventDefault(); if (overlayOpen()) closeOverlay(); else openPalette(); return; }
    if (typing || mod || ev.altKey || overlayOpen()) return;
    const k = ev.key;
    /* A key that is held down repeats. Moving through a list may repeat; deciding may not:
       A held a moment too long would accept one proposal after the other, unread. */
    if (ev.repeat && !/^(Arrow|[jJkK+\-=]$)/.test(k)) return;
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
    /* (the editor that opens takes the keyboard at once: without this the letter of the key
       itself would be typed into the page's text) */
    if ((k === 'e' || k === 'E') && cur && editable(cur) && r.view !== 'experiments' && r.view !== 'overview') { ev.preventDefault(); if (r.view === 'inbox' && r.sel !== cur.path) go({ sel: cur.path, sub: 'edit' }); else go({ sub: r.sub === 'edit' ? '' : 'edit' }); return; }
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
