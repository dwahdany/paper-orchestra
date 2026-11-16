/* The frame around every view: the top bar, the strip that says where the engine stands,
   the quiet status bar at the bottom of a project, and the banner for a server that is gone. */

import { VIEWS, app, hrefOf, link } from './core.js';
import { connection } from './api.js';
import { clock, duration, hm, html, listOf, money, parseStamp, plain, plural, raw, tf } from './util.js';
import { KIND, STAGES, icon, phaseName } from './vocab.js';

const MARK = raw('<svg class="brand-mark" width="18" height="18" viewBox="-10 -10 20 20" aria-hidden="true"><circle r="8"/><circle class="core" r="3"/></svg>');

export const theme = {
  get() { try { return localStorage.getItem('po-theme') || ''; } catch (e) { return ''; } },
  effective() { return this.get() || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); },
  toggle() {
    const next = this.effective() === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('po-theme', next); } catch (e) { /* the choice lasts for this visit */ }
    document.documentElement.setAttribute('data-theme', next);
    app.render('theme');
  },
};

/* ---------- what the engine is doing, in words ---------- */
/** One reading of state.json and the worker's status, shared by the strip, the top bar,
    the status bar and the overview. */
export function engineView(store) {
  const p = store.project || {};
  const state = p.state || {};
  const worker = p.worker || { state: 'idle' };
  const phases = state.phases || [];
  const last = phases[phases.length - 1] || null;
  const busy = worker.state && worker.state !== 'idle';
  const kind = busy ? worker.kind || 'advance' : null;
  const stopping = worker.state === 'stopping';
  let mode = state.status || 'idle';
  if (busy && kind === 'advance') mode = 'running';
  else if (busy && kind === 'run') mode = 'experiment';
  else if (busy && kind === 'write') mode = 'writing';
  else if (mode === 'running') mode = 'stopped';              // state.json says running, nothing runs: settled at the next look
  let budget = mode === 'checkpoint' && last && last.status === 'interrupted' && last.error ? plain(last.error) : null;
  /* The phase stopped at the limit for models, and the limit has been raised since: the
     stop is over. (The record of the phase keeps the sentence of the moment it stopped.) */
  const limits = (p.settings || {}).limits || {}, spent = state.spent || {};
  const lifted = !!budget && /model budget/i.test(last.error) && limits.model_usd != null && (spent.model_usd || 0) < limits.model_usd;
  if (lifted) budget = null;
  const died = mode === 'stopped' && !!state.error;
  const exit = worker.last_exit || null;
  const crashed = !busy && exit && !exit.stopped && exit.code !== 0 && exit.code != null && (exit.kind === 'advance' || exit.code < 0) && mode !== 'failed' && !died;
  return {
    mode, stopping, kind, budget, lifted, died, crashed, exit,
    round: state.round || 0,
    /* a worker that has just started has not written its phase yet: it is the one it was asked for */
    phase: busy && kind === 'advance' && state.status !== 'running' ? worker.phase || state.next_phase || state.phase : state.phase || (busy ? worker.phase : null),
    next: state.next_phase || null,
    error: state.error || null,
    started: busy ? worker.started : null,
    last,
    fresh: !phases.length && mode === 'idle',
  };
}

/** A worker that ended without leaving an account in state.json. */
export const exitText = (exit) => 'The ' + (exit.kind || 'agent') + ' worker ' + (exit.code < 0 ? 'was killed (signal ' + -exit.code + ')' : 'ended with exit code ' + exit.code) + ' before it could record why. What it wrote is kept; its last words are in the log.';

/** Why the agents stopped without leaving an account in state.json: the error the worker
    reported just before it ended, or what is known of its exit. */
export function crashText(store, exit) {
  const ended = parseStamp(exit.at) || Date.now();
  const said = store.activity.slice().reverse().find((ev) => ev.type === 'error' && Math.abs((parseStamp(ev.ts) || 0) - ended) < 120000 && !/worker (was killed|ended|is gone)/.test(ev.summary || ''));
  return said ? said.summary : exitText(exit);
}

export function waitingSentence(store) {
  const items = store.pending();
  if (!items.length) return 'Nothing is waiting for you.';
  const by = {};
  let edits = 0;
  items.forEach((it) => { if (it.type === 'edit') edits++; else by[it.kind] = (by[it.kind] || 0) + 1; });
  const parts = Object.keys(by).sort((a, b) => STAGES.indexOf(a) - STAGES.indexOf(b)).map((k) => by[k] + ' ' + (by[k] === 1 ? (KIND[k] || { one: k }).one : (KIND[k] || { many: k + 's' }).many).toLowerCase());
  if (edits) parts.push(plural(edits, 'suggested edit'));
  return listOf(parts) + (items.length === 1 ? ' is' : ' are') + ' waiting for you.';
}
/** What the main button does next. The run phase runs what is approved; with nothing
    approved it goes straight on, to the analysis of results nobody has read yet when
    there are any, and the button says so instead of promising a run. */
export function continueLabel(next, store) {
  if (next === 'run' && store) {
    const ready = store.experiments.filter((x) => x.review === 'accepted' && (x.state === 'approved' || x.state === 'queued')).length;
    if (ready === 1) return 'Run the approved experiment';
    if (!ready) return store.pages.some((p) => p.kind === 'result' && !(p.meta || {}).analysed) ? 'Continue to analysis' : 'Continue without a run';
  }
  return next === 'run' ? 'Run the approved experiments' : next === 'research' ? 'Continue to research' : next ? 'Continue to ' + phaseName(next) : 'Continue';
}
/** Why the button does not say "run" when the next phase is the run phase. */
export function continueTip(next, store) {
  if (next !== 'run' || !store) return '';
  const label = continueLabel(next, store);
  if (label === 'Continue to analysis') return 'No experiment is approved to run. The results that are there have not been analysed yet: that comes next.';
  if (label === 'Continue without a run') return 'No experiment is approved to run. Approve one under Experiments first, or go on without a run.';
  return '';
}
const PAUSE = raw('<svg class="ic" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M5.5 3.5v9M10.5 3.5v9"/></svg>');
const RING = raw('<svg class="ic" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.2"/></svg>');

/* ---------- top bar ---------- */
export function topbar(route) {
  const right = html`<button class="ibtn" data-act="palette" data-tip="Find a page, run an action (Ctrl K)" aria-label="Find a page or run an action">${icon('search')}</button>
    <button class="ibtn" data-act="theme" data-tip="Light or dark (T)" aria-label="Switch between light and dark">${icon(theme.effective() === 'dark' ? 'sun' : 'moon')}</button>
    <button class="ibtn" data-act="help" data-tip="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">${icon('help')}</button>`;
  const store = app.store;
  if (route.name !== 'project' || !store || !store.loaded) {
    const nav = [['library', 'Projects'], ['start', 'New project'], ['setup', 'Setup']];
    const here = route.name === 'project' ? '' : route.name;
    return html`<a class="brand" href="#/library">${MARK}<span>PaperOrchestra</span></a><nav class="views" aria-label="Sections">
      ${nav.map((n) => html`<a href="#/${n[0]}"${here === n[0] ? raw(' aria-current="page"') : ''}>${n[1]}</a>`)}</nav><div class="top-r">${right}</div>`;
  }
  const p = store.project, e = engineView(store), n = store.pending().length;
  const round = html`<span class="ts-long">Round ${e.round || 1} · </span>`;
  const status = e.mode === 'running' ? html`<span class="live-dot" aria-hidden="true"></span>${round}${phaseName(e.phase) || 'starting'}`
    : e.mode === 'experiment' ? html`<span class="live-dot" aria-hidden="true"></span><span class="ts-long">Experiment </span>running`
    : e.mode === 'writing' ? html`<span class="live-dot" aria-hidden="true"></span>Writing the paper`
    : e.mode === 'failed' ? html`${icon('warn')}${cap1(phaseName(e.next || e.phase))} failed`
    : e.died || e.crashed ? html`${icon('warn')}The agents stopped`
    : e.mode === 'stopped' ? html`${PAUSE}Stopped by you`
    : e.fresh ? html`${RING}Not started`
    : e.budget ? html`${icon('warn')}Budget reached`
    : html`${PAUSE}${round}checkpoint`;
  const cls = e.mode === 'running' || e.mode === 'experiment' || e.mode === 'writing' ? 'running' : e.mode === 'failed' || e.died || e.crashed ? 'failed' : e.budget ? 'budget' : e.mode;
  const limits = (p.settings || {}).limits || {};
  const spent = (p.state || {}).spent || {};
  return html`<a class="brand" href="#/library" data-tip="All projects">${MARK}<span>PaperOrchestra</span></a><span class="top-sep" aria-hidden="true">/</span>
    <a class="top-proj" href="${link({ view: 'overview', sel: '', sub: '' })}" data-tip="${p.title}">${p.title}</a>
    <nav class="views" aria-label="Views of this project">${VIEWS.map((v, i) => html`<a href="${hrefOf({ name: 'project', pid: route.pid, view: v[0], sel: v[0] === 'paper' ? '' : route.sel })}"${route.view === v[0] ? raw(' aria-current="page"') : ''} data-tip="${v[1]} (${i + 1})">${v[1]}${v[0] === 'inbox' && n ? html`<span class="count" aria-label="${n} waiting">${n}</span>` : ''}</a>`)}</nav>
    <div class="top-r"><a class="top-status st-${cls}" href="${link({ view: 'overview', sel: '', sub: '' })}" data-tip="Where things stand">${status}</a>
      <span class="top-spend" data-tip="Spent on models, of the limit for this project">${money(spent.model_usd || 0)}<span> / ${money(limits.model_usd)}</span></span>${right}</div>`;
}
const cap1 = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '');

/* ---------- the strip ---------- */
/** Who an event is from: the person, an agent, or the engine. */
export const whoOf = (ev) => (ev.type === 'review.decided' || ev.type === 'request.created' || (ev.data || {}).by === 'human' ? 'you' : ev.agent || 'engine');
/** The events with what belongs to one run on one line: queued, running, the result record
    and the end of the run are the steps of "E1 ran on this machine: succeeded after 1.8 s".
    A line is `{run: true, key, eid, n, backend, status, seconds, error, ts, parts}`. */
export function foldRuns(events) {
  const out = [], open = new Map(), byKey = new Map();
  const start = (eid, ev) => { const line = { run: true, key: eid + '#next', eid, n: null, backend: '', status: 'queued', seconds: null, error: '', ts: ev.ts, seq: ev.seq, parts: [] }; open.set(eid, line); out.push(line); return line; };
  const number = (line, n) => { line.n = n; line.key = line.eid + '#' + n; byKey.set(line.key, line); };
  events.forEach((ev) => {
    const d = ev.data || {};
    if (ev.type === 'experiment.state' && d.by === 'system' && d.id) {
      if (d.state === 'queued' || d.state === 'running') {
        const line = open.get(d.id) || start(d.id, ev);
        if (line.n == null) line.status = d.state;
        line.parts.push(ev);
        return;
      }
      const line = open.get(d.id);
      if (line && line.n != null) { line.parts.push(ev); open.delete(d.id); return; }
    }
    if (ev.type === 'run.started' && d.experiment) {
      let line = open.get(d.experiment);
      if (!line || (line.n != null && line.n !== d.run)) line = start(d.experiment, ev);
      number(line, d.run);
      line.backend = d.backend || '';
      line.status = 'running';
      line.parts.push(ev);
      return;
    }
    if (ev.type === 'run.finished' && d.experiment) {
      let line = byKey.get(d.experiment + '#' + d.run);
      if (!line) { line = start(d.experiment, ev); number(line, d.run); }
      line.status = d.requeued ? 'requeued' : d.status || 'finished';
      line.seconds = d.seconds;
      line.error = d.error || '';
      line.parts.push(ev);
      return;
    }
    if (ev.type === 'page.created' && d.kind === 'result' && d.by === 'system') {
      const m = /(?:^|\/)([A-Za-z]+\d+)-run(\d+)\.md$/.exec(d.path || '');
      const line = m && byKey.get(m[1] + '#' + m[2]);
      if (line) { line.parts.push(ev); return; }
    }
    out.push(ev);
  });
  return out;
}
/** What a folded run says. */
export function runSentence(line) {
  const where = line.backend === 'local' ? ' on this machine' : line.backend === 'modal' ? ' on Modal' : line.backend ? ' on ' + line.backend : '';
  const which = line.n > 1 ? ' (run ' + line.n + ')' : '';
  if (line.n == null) return line.eid + (line.status === 'running' ? ' is starting' : ' is queued to run');
  if (line.status === 'running') return line.eid + ' is running' + where + which;
  if (line.status === 'requeued') return line.eid + ' was stopped' + where + ' and is queued again' + which;
  const after = line.seconds == null ? '' : ' after ' + duration(line.seconds);
  const why = line.status === 'failed' && line.error ? '. ' + (line.error.length > 150 ? line.error.slice(0, 149) + '…' : line.error) : '';
  return line.eid + ' ran' + where + which + ': ' + line.status + after + why;
}
function activityLines(store, limit) {
  const lines = foldRuns(store.activity.filter((ev) => ev.type !== 'run.log')).slice(-limit);
  return lines.map((ev) => html`<div><time>${hm(ev.ts)}</time><b>${ev.run ? 'runner' : whoOf(ev)}</b><span>${ev.run ? runSentence(ev) : ev.summary}</span></div>`);
}
/** Stop, unless the work was started in a terminal on a machine this server cannot signal. */
function stopButton(store, e) {
  const w = store.project.worker || {};
  if (w.adopted && w.pid == null) return html`<span class="strip-foreign" data-tip="Someone started it with the command line${w.host ? ' on ' + w.host : ''}. This page cannot stop it.">Started in a terminal: stop it there</span>`;
  return html`<button class="btn" data-act="stop"${e.stopping ? raw(' disabled') : ''}>${icon('stop')}${e.stopping ? 'Stopping…' : 'Stop'}</button>`;
}
function workerLogHTML(store) {
  const log = store.workerLog;
  if (!log) return '';
  if (log.text == null) return html`<p class="quiet">Reading what the worker wrote…</p>`;
  if (!log.text.trim()) return html`<p class="quiet">The worker wrote nothing to <code>worker.log</code>.</p>`;
  return html`<pre class="strip-pre" data-follow>${log.text.split('\n').slice(-60).join('\n')}</pre>`;
}
export function strip(store) {
  const e = engineView(store), n = store.pending().length;
  const open = app.ui.activityOpen;
  const toggle = (what) => html`<button class="btn btn-quiet" data-act="toggle-activity" aria-expanded="${tf(open)}">${open ? 'Hide' : 'Show'} ${what} ${icon(open ? 'down' : 'right')}</button>`;
  if (e.mode === 'running') {
    const since = (parseStamp(e.started) || 0) - 2000;
    const last = store.activity.slice().reverse().find((ev) => /^(agent\.activity|page\.created|phase\.started|proposal\.created|run\.started|run\.finished|experiment\.state)$/.test(ev.type) && (parseStamp(ev.ts) || 0) >= since);
    return html`<div class="strip s-running" role="status"><span class="live-dot" aria-hidden="true"></span><span class="strip-kicker">Round ${e.round || 1} · ${phaseName(e.phase) || 'starting'}</span>
      <span class="strip-text">${e.stopping ? 'Stopping: the phase ends at its next step and can be resumed.' : last ? (String(last.summary || '').toLowerCase().startsWith('the ' + String(last.agent || 'engine').toLowerCase() + ' ') ? last.summary : html`<b>${last.agent || 'engine'}</b> ${last.summary}`) : 'Starting…'}</span>
      ${e.started ? html`<span class="strip-clock" data-since="${e.started}">${clock((Date.now() - (parseStamp(e.started) || Date.now())) / 1000)}</span>` : ''}
      <span class="strip-actions">${toggle('activity')}${stopButton(store, e)}</span></div>
      ${open ? html`<div class="strip-log" data-follow>${activityLines(store, 40)}</div>` : ''}`;
  }
  if (e.mode === 'experiment' || e.mode === 'writing') {
    const running = store.experiments.find((x) => x.state === 'running' || x.state === 'queued');
    return html`<div class="strip s-running" role="status"><span class="live-dot" aria-hidden="true"></span><span class="strip-kicker">${e.mode === 'writing' ? 'Writing the paper' : 'Experiment'}</span>
      <span class="strip-text">${e.stopping ? 'Stopping…' : e.mode === 'writing' ? 'The write-up is being produced from the wiki.' : running ? html`<b>${running.id}</b> ${running.title}` : 'A run is starting.'}</span>
      ${e.started ? html`<span class="strip-clock" data-since="${e.started}">${clock((Date.now() - (parseStamp(e.started) || Date.now())) / 1000)}</span>` : ''}
      <span class="strip-actions">${e.mode === 'writing' ? html`<a class="btn" href="${link({ view: 'paper', sel: '', sub: '' })}">Open the paper</a>` : running ? html`<a class="btn" href="${link({ view: 'experiments', sel: running.path, sub: '' })}">Watch the run</a>` : ''}
        ${stopButton(store, e)}</span></div>`;
  }
  if (e.mode === 'failed' || e.died || e.crashed) {
    const phase = e.next || e.phase;
    const title = e.mode === 'failed' ? cap1(phaseName(phase)) + ' failed' : 'The agents stopped unexpectedly';
    const text = e.mode === 'failed' ? e.error : e.died ? e.error : crashText(store, e.exit);
    return html`<div class="strip s-failed" role="alert"><span class="strip-mark">${icon('warn')}</span><span class="strip-kicker">${title}</span>
      <span class="strip-text">${text}</span>
      <span class="strip-actions">${toggle('log')}<a class="btn" href="#/setup">Open Setup</a><button class="btn btn-primary" data-act="continue">${e.mode === 'failed' ? 'Retry ' + phaseName(phase) : e.died && phase ? 'Resume ' + phaseName(phase) : e.fresh || !e.next ? 'Start again' : 'Try again: ' + continueLabel(e.next, store).replace(/^Continue to /, '').replace(/^Continue /, '').replace(/^Run /, 'run ')}</button></span></div>
      ${open ? html`<div class="strip-log"><p class="strip-fix">What was written so far is kept. Put the cause right (credentials are checked under Setup), then retry: the phase resumes where it stopped.</p>${workerLogHTML(store)}${activityLines(store, 12)}</div>` : ''}`;
  }
  if (e.fresh) {
    return html`<div class="strip s-idle"><span class="strip-mark">${RING}</span><span class="strip-kicker">Not started</span><span class="strip-text">Research has not started. Nothing has been spent.</span>
      <span class="strip-actions"><button class="btn btn-primary" data-act="continue">Start research ${icon('right')}</button></span></div>`;
  }
  if (e.mode === 'stopped') {
    return html`<div class="strip s-checkpoint"><span class="strip-mark">${PAUSE}</span><span class="strip-kicker">Stopped · round ${e.round || 1}, during ${phaseName(e.next || e.phase)}</span>
      <span class="strip-text">You stopped the agents. What they had written is kept; the phase resumes where it stopped.</span>
      <span class="strip-actions">${n ? html`<a class="btn" href="${link({ view: 'inbox', sel: '', sub: '' })}">Review ${n}</a>` : ''}<button class="btn btn-primary" data-act="continue">Resume ${phaseName(e.next || e.phase)} ${icon('right')}</button></span></div>`;
  }
  if (e.budget) {
    const editing = app.ui.limitEdit;
    return html`<div class="strip s-budget" role="status"><span class="strip-mark">${icon('warn')}</span><span class="strip-kicker">Budget reached · round ${e.round || 1}</span>
      <span class="strip-text">${e.budget}</span>
      <span class="strip-actions">${editing ? html`<form class="limit-form" data-form="limit"><label class="sr" for="limit-usd">New limit for models, in dollars</label><span class="quiet">Models, up to $</span><input id="limit-usd" name="usd" data-input="limit" inputmode="decimal" value="${editing.value}" size="6"><button class="btn btn-primary" type="submit">Set the limit</button><button class="btn btn-quiet" type="button" data-act="limit-cancel">Cancel</button></form>`
        : html`<button class="btn" data-act="limit-edit">Raise the limit…</button><button class="btn btn-primary" data-act="continue">${continueLabel(e.next, store)} ${icon('right')}</button>`}</span></div>`;
  }
  return html`<div class="strip s-checkpoint"><span class="strip-mark">${PAUSE}</span>
    <span class="strip-kicker">Checkpoint · round ${e.round || 1}${e.phase ? (e.lifted ? ', during ' : ', after ') + phaseName(e.phase) : ''}</span><span class="strip-text">${e.lifted ? 'The limit for models is raised. ' + cap1(phaseName(e.next || e.phase)) + ' stopped at the old one and can go on where it stopped. ' : ''}${waitingSentence(store)}</span>
    <span class="strip-actions">${n ? html`<a class="btn" href="${link({ view: 'inbox', sel: '', sub: '' })}">Review ${n}</a>` : ''}
      <button class="btn btn-primary" data-act="continue" data-tip="${[continueTip(e.next, store), n ? 'Proposals you have not decided stay in the inbox.' : ''].filter(Boolean).join(' ')}">${e.lifted ? 'Resume ' + phaseName(e.next || e.phase) : continueLabel(e.next, store)} ${icon('right')}</button></span></div>`;
}

/* ---------- the status bar ---------- */
export function statusbar(store) {
  const p = store.project, e = engineView(store), n = store.pending().length;
  const limits = (p.settings || {}).limits || {}, spent = (p.state || {}).spent || {};
  const running = store.experiments.find((x) => x.state === 'running');
  const run = running && running.last_run && running.last_run.status === 'running' ? running.last_run : null;
  const state = e.mode === 'running' ? html`<span class="sb-live"><span class="live-dot" aria-hidden="true"></span><b>Running</b> round ${e.round || 1}${e.phase ? ' · ' + phaseName(e.phase) : ''}</span>`
    : e.mode === 'experiment' ? html`<span class="sb-live"><span class="live-dot" aria-hidden="true"></span><b>Running</b> an experiment</span>`
    : e.mode === 'writing' ? html`<span class="sb-live"><span class="live-dot" aria-hidden="true"></span><b>Writing</b> the paper</span>`
    : e.mode === 'failed' ? html`<span class="sb-neg">${icon('warn')}<b>Failed</b> ${phaseName(e.next || e.phase)}</span>`
    : e.died || e.crashed ? html`<span class="sb-neg">${icon('warn')}<b>Stopped</b> unexpectedly</span>`
    : e.mode === 'stopped' ? html`<span>${PAUSE}<b>Stopped</b> during ${phaseName(e.next || e.phase)}</span>`
    : e.fresh ? html`<span>${RING}<b>Not started</b></span>`
    : html`<span class="sb-check">${PAUSE}<b>${e.budget ? 'Budget reached' : 'Checkpoint'}</b> round ${e.round || 1}${e.phase ? html` · ${phaseName(e.phase)}${e.next && e.next !== e.phase ? html` → ${phaseName(e.next)}` : ''}` : ''}</span>`;
  const live = !connection.ok ? html`<span class="sb-neg" data-tip="The server is not answering">offline</span>` : store.live === 'reconnecting' ? html`<span class="sb-prop" data-tip="The live connection dropped; it is being opened again">reconnecting…</span>` : html`<span class="sb-ok" data-tip="Changes arrive as they happen">live</span>`;
  const frac = (a, b) => (b > 0 ? Math.max(0, Math.min(1, a / b)) : 0);
  return html`<a class="sb-i sb-state" href="${link({ view: 'overview', sel: '', sub: '' })}">${state}</a>
    <a class="sb-i${n ? ' sb-wait' : ''}" href="${link({ view: 'inbox', sel: '', sub: '' })}"><b>${n}</b> waiting</a>
    ${running ? html`<a class="sb-i sb-run" href="${link({ view: 'experiments', sel: running.path, sub: '' })}"><span class="live-dot" aria-hidden="true"></span><b>${running.id}</b>${run ? html` run ${run.number} · <span data-since="${run.started}">${clock((Date.now() - (parseStamp(run.started) || Date.now())) / 1000)}</span>${running.meta && running.meta.est_minutes ? html` of about ${running.meta.est_minutes} min` : ''}` : ' running'}</a>` : ''}
    <span class="sb-gap"></span>
    <span class="sb-i" data-tip="Spent on models, of the limit for this project">models <b>${money(spent.model_usd || 0)}</b> <span class="sb-of">/ ${money(limits.model_usd)}</span><progress class="sb-meter${frac(spent.model_usd || 0, limits.model_usd) > 0.85 ? ' hot' : ''}" max="1" value="${frac(spent.model_usd || 0, limits.model_usd)}" aria-label="Model spend"></progress></span>
    <span class="sb-i" data-tip="Estimated compute spend of the runs, of the limit for this project">experiments <b>${money(spent.experiment_usd || 0)}</b> <span class="sb-of">/ ${money(limits.experiment_usd)}</span></span>
    <span class="sb-i sb-conn">${live}</span>`;
}

/* ---------- the server is gone ---------- */
export function banner() {
  if (connection.ok) return html``;
  return html`<div class="banner-in">${icon('warn')}<span><b>The server is not answering.</b> Nothing you typed is lost: drafts stay on this device. It is tried again every few seconds.</span><button class="btn btn-s" data-act="retry-connection">Retry now</button></div>`;
}

