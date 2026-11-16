/* The frame around every view: the top bar, the strip that says where the engine stands,
   the quiet status bar at the bottom of a project, and the banner for a server that is gone. */

import { VIEWS, app, hrefOf, link } from './core.js';
import { connection, RESTARTED } from './api.js';
import { clock, dateTime, duration, hm, html, listOf, money, parseStamp, plain, plural, raw, storage, tf } from './util.js';
import { KIND, PHASE_DO, STAGES, icon, phaseName } from './vocab.js';

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
/** One reading of the project view, shared by the strip, the top bar, the status bar and the
    overview. Where the project stands is the engine's own reading (`state`), and the server
    says where that reading came from (`state_source`) and what is wrong with state.json
    (`state_problem`, a sentence): nothing about either is worked out here. */
export function engineView(store) {
  const p = store.project || {};
  const state = p.state || {};
  const worker = p.worker || { state: 'idle' };
  const phases = state.phases || [];
  const last = phases[phases.length - 1] || null;
  const busy = !!worker.state && worker.state !== 'idle';
  const kind = busy ? worker.kind || 'advance' : null;
  const stopping = worker.state === 'stopping';
  /* The project's files name a worker that the server cannot show to be this project's: it
     is not ours to stop, and nothing can be started beside it until it is gone or forgotten. */
  const unverified = busy && worker.verified === false;
  let mode = state.status || 'idle';
  if (unverified) mode = 'unverified';
  else if (busy && kind === 'advance') mode = 'running';
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
  const source = typeof p.state_source === 'string' ? p.state_source : '';
  const stateNote = typeof p.state_problem === 'string' ? p.state_problem : '';
  /* Nothing on file says where the project stands: said as that, in the server's sentence,
     and never as a project that has not started. */
  const unknown = source === 'unknown' && !busy;
  const exit = worker.last_exit || null;
  const crashed = !busy && exit && !exit.stopped && exit.code !== 0 && exit.code != null && (exit.kind === 'advance' || exit.code < 0) && mode !== 'failed' && !died;
  return {
    mode, stopping, kind, budget, lifted, died, crashed, exit, unverified,
    /* agents, a run or the write-up are at work, and this server can stop them */
    working: mode === 'running' || mode === 'experiment' || mode === 'writing',
    note: unverified ? worker.note || '' : '',
    round: state.round || 0,
    /* a worker that has just started has not written its phase yet: it is the one it was asked for */
    phase: busy && !unverified && kind === 'advance' && state.status !== 'running' ? worker.phase || state.next_phase || state.phase : state.phase || (busy ? worker.phase : null),
    next: state.next_phase || null,
    error: state.error || null,
    started: busy ? worker.started : null,
    last,
    /* no phase has run: the server says so (a server that does not say is taken at its list of phases) */
    fresh: mode === 'idle' && !unknown && (source ? source === 'not_started' : !phases.length),
    unknown, stateNote,
    /* state.json did not say, and the engine read round, phases and what is next from the events instead */
    rebuilt: source === 'events' && !!stateNote,
  };
}

/** A worker that ended without leaving an account in state.json. */
export const exitText = (exit) => 'The ' + (exit.kind || 'agent') + ' worker ' + (exit.code < 0 ? 'was killed (signal ' + -exit.code + ')' : 'ended with exit code ' + exit.code) + ' before it could record why. What it wrote is kept; its last words are in the log.';

/** Why the agents stopped without leaving an account in state.json: the error the worker
    reported just before it ended, or what is known of its exit. */
export function crashText(store, exit) {
  const ended = parseStamp(exit.at) || Date.now();
  /* not the note that a lost state.json was rebuilt: that is said among the project's problems, and is not why a worker ended */
  const said = store.activity.slice().reverse().find((ev) => ev.type === 'error' && !(ev.data || {}).rebuilt && Math.abs((parseStamp(ev.ts) || 0) - ended) < 120000 && !/worker (was killed|ended|is gone)/.test(ev.summary || ''));
  return said ? plain(said.summary) : exitText(exit);
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
/* ---------- what happens next ---------- */
/** What pressing the main button does, in words. The answer itself is the server's:
    `next_step` of the project view, which the engine works out by the rules its own
    `advance` goes by (`will_run`: the phases a press runs, in order; `then`: the phase its
    checkpoint names as next; `experiments`: what a run phase is owed; `results`: how many
    result records have no analysis yet; `checkpoint_of`: a phase whose checkpoint is still
    owed; `needs_phase`: a press would be refused until a phase is named). Nothing of that is
    worked out here: this only finds the words for it.

    `{known, blocked, steps, back, label, says, tip, href, pick, waiting, results}`: whether
    the server said what a press does; whether no press is offered at all (nothing says where
    the project stands, and the server could not say what a press would do either: its
    sentence says what to put right); the phases the press runs; the phase that was next and is
    still next afterwards, when something else comes first; the words on the button; a sentence for beside it when
    the button cannot say it all; the same with the fine print; where a press would do
    nothing, the address the button leads to instead; whether the button opens the list of
    phases, because one has to be named; the experiments that wait for a run, and how many
    results wait for an analysis. */
export function nextStep(store, e) {
  e = e || engineView(store);
  const p = store.project || {};
  const said = p.next_step && typeof p.next_step === 'object' ? p.next_step : null;
  const list = (value) => (Array.isArray(value) ? value.filter((x) => typeof x === 'string' && x) : []);
  const ids = said ? list(said.experiments) : [], results = said ? +said.results || 0 : 0;
  const resume = e.mode === 'stopped' || e.died || e.lifted;
  const out = { known: false, steps: [], back: null, label: 'Continue', says: '', tip: '', href: '', pick: false, waiting: ids, results };
  /* The server did not say (it could not tell, or it is an older one), or the state has just
     moved (it came with the event stream) and the project view, which carries the forecast,
     has not been read since: the button then promises nothing about what comes first. A
     press does what the engine finds either way. */
  if (!said || (store.behind && !e.working && !e.fresh)) {
    out.settling = !!said;
    out.blocked = !said && e.unknown;
    out.label = e.fresh && !e.crashed ? 'Start research' : e.mode === 'failed' ? 'Retry' : resume ? 'Resume' : e.crashed ? 'Try again' : 'Continue';
    out.tip = said ? 'Reading what a press will do…' : '';
    return out;
  }
  out.known = true;
  if (said.needs_phase) {
    /* nothing on file says where the project stands: a press without a phase would be refused */
    out.pick = true;
    out.label = 'Choose the phase to go on with';
    return out;
  }
  const run = list(said.will_run), then = typeof said.then === 'string' && said.then ? said.then : null;
  const owed = typeof said.checkpoint_of === 'string' ? said.checkpoint_of : '';
  if (!run.length) {
    /* a phase had finished when its worker ended, without its checkpoint: a press writes that, and no more */
    if (owed) {
      out.label = 'Continue to the checkpoint';
      out.says = 'The ' + phaseName(owed) + ' phase had finished when its worker ended, before it could pause. A press writes its checkpoint and runs nothing' + (then ? '; the next phase is then ' + phaseName(then) + '.' : '.');
      out.tip = out.says;
    }
    return out;
  }
  const next = e.next || (e.mode === 'stopped' || e.mode === 'failed' ? e.phase : null);
  /* a run phase with nothing to run (the server lists no experiment for it) is not what the press is for */
  const noRun = run[0] === 'run' && !ids.length;
  const shown = noRun && run.length > 1 ? run.slice(1) : run;
  const first = shown[0];
  out.steps = shown;
  /* the press runs something else first, and the phase that was next is the one the server
     names as next afterwards: it is said that it stays so */
  out.back = then && then === next && first !== next ? then : null;
  const still = (phase) => (e.mode === 'failed' ? 'After that, the ' + phaseName(phase) + ' phase is tried again.'
    : resume ? 'After that, the ' + phaseName(phase) + ' phase is resumed.'
      : 'After that, the next phase is still ' + phaseName(phase) + '.');
  const one = ids.length === 1;
  if (first === 'run' && ids.length) {
    /* left at "running" by a worker that is gone (the server marks them): settled and run again with the others */
    const again = ids.every((id) => (store.expById.get(id) || {}).interrupted);
    const analysed = shown[1] === 'analyze';
    out.label = (again ? 'Run the interrupted experiment' + (one ? '' : 's') + ' again' : one ? 'Run the approved experiment' : 'Run the ' + ids.length + ' approved experiments') + (analysed ? (one ? ', then analyse it' : ', then analyse them') : '');
    if (out.back) {
      out.says = listOf(ids) + (one ? ' runs first' + (analysed ? ' and is analysed. ' : '. ') : ' run first' + (analysed ? ' and are analysed. ' : '. ')) + still(out.back);
      out.tip = out.says + ' If ' + (one ? 'it cannot be run' : 'none of them can be run') + ', the ' + phaseName(out.back) + ' phase goes ahead instead, and its checkpoint says why.';
    }
  } else if (first === 'analyze' && results) {
    out.label = results === 1 ? 'Analyse the new result' : 'Analyse the ' + results + ' new results';
    out.says = (noRun ? 'No experiment is approved to run. ' : '') + (results === 1 ? 'One result has' : results + ' results have') + ' not been analysed yet: that comes first.' + (out.back ? ' ' + still(out.back) : '');
    out.tip = out.says;
  } else if (noRun && run.length === 1) {
    /* nothing is approved and nothing waits to be analysed: the run phase would say so and pause again */
    const could = store.experiments.some((x) => x.review === 'proposed' || (x.review === 'accepted' && (x.state === 'planned' || !x.state)));
    out.says = could ? 'No experiment is approved to run. Approve one first, or choose another phase.' : 'There is no experiment to run. Choose another phase from the list beside the button.';
    out.tip = out.says;
    if (could) { out.label = 'Approve an experiment to run'; out.href = link({ view: 'experiments', sel: '', sub: '' }); out.steps = []; } else out.label = 'Continue without a run';
  } else {
    out.label = e.fresh && !e.crashed ? 'Start research' : e.mode === 'failed' ? 'Retry ' + phaseName(first) : resume ? 'Resume ' + phaseName(first) : e.crashed ? (e.fresh ? 'Start again' : 'Try again: ' + phaseName(first)) : 'Continue to ' + phaseName(first);
    if (noRun) { out.says = 'No experiment is approved to run: the press goes straight on to ' + phaseName(first) + '.'; out.tip = out.says; }
  }
  return out;
}
/** The phases the person can choose to run in place of what comes next, where choosing them
    makes sense: `[{phase, title, sub}]`. More research is always among them: nothing but this
    choice sends a project back to the literature. Whether there is something to run or to
    analyse is the server's word (`step.waiting`, `step.results`); where it did not say, both
    are offered and the server answers the choice. */
export function phaseChoices(store, e, step) {
  if (e.fresh || step.blocked) return [];
  const accepted = (kind, test) => store.pages.some((p) => p.kind === kind && p.review === 'accepted' && (!test || test(p)));
  const only = step.steps.length === 1 && !step.href ? step.steps[0] : null;    // what the button does by itself
  const list = [];
  const add = (phase, when) => { if (when && only !== phase) list.push({ phase, title: PHASE_DO[phase][0], sub: PHASE_DO[phase][1] }); };
  const told = step.known || step.settling;
  add('research', true);
  add('hypothesize', accepted('topic') || accepted('question') || accepted('source'));
  add('design', accepted('hypothesis', (p) => p.state !== 'dropped'));
  add('run', told ? step.waiting.length > 0 && step.steps[0] !== 'run' : true);
  add('analyze', told ? step.results > 0 && step.steps.indexOf('analyze') < 0 : true);
  return list;
}
/** Why the person may want the literature searched again, as the server said it: the
    analyst's recommendation, which the checkpoint of the analysis carries as a field
    (`recommends`, with the journal page that says so in `recommends_page`; the server takes
    both away once that page is rejected, gone or says otherwise), and instructions for the
    scout that wait for a research phase (which does not come round by itself). */
export function researchReasons(store, e) {
  const out = [];
  const p = store.project || {}, cp = p.checkpoint || null;
  if (e.next === 'research') return out;
  if (cp && cp.recommends === 'research' && e.mode === 'checkpoint') {
    const page = typeof cp.recommends_page === 'string' ? store.byPath.get(cp.recommends_page) || null : null;
    out.push({ kind: 'recommended', text: 'The journal page of this round recommends another look at the literature first. It is yours to decide.', brief: 'Recommended in the journal page of this round. It is yours to decide.', page });
  }
  const forScout = store.requests.filter((r) => r.state === 'open' && r.for === 'scout').length;
  if (forScout) out.push({ kind: 'requests', text: (forScout === 1 ? 'One instruction for the scout waits' : forScout + ' instructions for the scout wait') + ' for a research phase, which does not come round by itself.', page: null });
  return out;
}
/** The control that continues a project: the main button, and beside it the list of phases
    the person can choose instead. `where` names the control whose list is open. Where a
    phase has to be named before anything can go on, the one button opens that list. */
export function continueControl(store, e, where, large) {
  const step = nextStep(store, e), choices = phaseChoices(store, e, step), open = app.ui.goMenu === where;
  if (step.blocked) return '';
  const cls = 'btn btn-primary' + (large ? ' btn-l' : '');
  const tip = step.tip || (where === 'strip' && store.pending().length ? 'Proposals you have not decided stay in the inbox.' : '');
  const why = choices.length ? researchReasons(store, e) : [];
  const menu = open && choices.length ? html`<div class="go-menu" role="menu" aria-label="${step.pick ? 'Phases to go on with' : 'Run another phase instead'}">
      <p class="go-h">${step.pick ? 'Go on with the phase you choose' : 'Run a phase of your choice instead'}</p>
      ${choices.map((c) => html`<button class="go-i" role="menuitem" data-act="continue-phase" data-phase="${c.phase}"><b>${c.title}</b><span>${c.sub}</span>${c.phase === 'research' ? why.map((w) => html`<span class="go-why">${icon('again')}${w.text}</span>`) : ''}</button>`)}
      <p class="go-f">Only a choice made here changes which phase runs next.</p></div>` : '';
  if (step.pick && choices.length) {
    return html`<span class="go go-pick${open ? ' open' : ''}" data-menu="${where}"><button class="${cls} go-more" data-act="go-menu" data-where="${where}" data-go-main aria-haspopup="menu" aria-expanded="${tf(open)}" data-tip="Nothing on file says where the project stands, so the agents go on with the phase you name">${step.label} ${icon('down')}</button>${menu}</span>`;
  }
  const main = step.href ? html`<a class="${cls}" href="${step.href}" data-go-main data-tip="${tip}">${step.label} ${icon('right')}</a>`
    : html`<button class="${cls}" data-act="continue" data-tip="${tip}">${step.label} ${icon('right')}</button>`;
  if (!choices.length) return main;
  return html`<span class="go${open ? ' open' : ''}" data-menu="${where}">${main}<button class="${cls} go-more" data-act="go-menu" data-where="${where}" aria-haspopup="menu" aria-expanded="${tf(open)}" aria-label="Choose another phase to run" data-tip="Choose another phase to run">${icon('down')}</button>${menu}</span>`;
}
/** What stands beside the control: what a press does when the button cannot say all of it,
    and why the literature might be searched again, with that choice next to the reason.
    `brief`: the checkpoint's own words stand right above (the overview shows them), so a
    reason they give already is named in a few words, with its choice. */
export function continueNotes(store, e, brief) {
  const step = nextStep(store, e), offered = phaseChoices(store, e, step).some((c) => c.phase === 'research');
  const why = offered ? researchReasons(store, e) : [];
  if (!step.says && !why.length) return '';
  return html`${step.says ? html`<p class="go-says">${step.says}</p>` : ''}${why.map((w) => html`<p class="go-rec" data-reason="${w.kind}">${icon('again')}<span>${(brief && w.brief) || w.text}${w.page ? html` <a class="linkish" href="${link({ view: 'wiki', sel: w.page.path, sub: '' })}" data-sel="${w.page.path}">Read the journal</a>` : ''}</span>
    <button class="btn btn-s" data-act="continue-phase" data-phase="research">${PHASE_DO.research[0]}</button></p>`)}`;
}
/** What the engine says when state.json did not tell where the project stands and it read
    that from the project's events instead: the server's sentence, which asks the person to
    look at the round and the next phase before they continue. It stands beside the control. */
export const stateNoteHTML = (e) => (e.rebuilt ? html`<p class="go-warn" data-state-problem>${icon('warn')}<span>${plain(e.stateNote)}</span></p>` : '');

/* ---------- what the project's files had to be read around ---------- */
const noticeKey = (pid) => 'po-notices-seen:' + pid;
/** What the person should know about the project's files: what the store read leniently
    (`problems` of the project view: `{path, message}`, of which the server lists the first
    few and says how many there are in `problems_total`), what the engine says about a
    state.json it had to read around (`state_problem`, when the state was rebuilt from the
    events), and what the engine said when it rebuilt a lost state.json or carried spending
    over from a lost usage ledger. The engine's notes are events; each is shown until the
    person has put it aside. */
export function problemsOf(store) {
  const p = store.project || {};
  const files = (Array.isArray(p.problems) ? p.problems : []).map((x) => (typeof x === 'string' ? { path: '', message: x } : { path: (x && x.path) || '', message: (x && x.message) || '' })).filter((x) => x.message);
  const all = Number.isInteger(p.problems_total) && p.problems_total > files.length ? p.problems_total : files.length;
  const state = p.state_source === 'events' && typeof p.state_problem === 'string' ? p.state_problem : '';
  const seen = +storage.get(noticeKey(store.pid), 0) || 0;
  const notes = [];
  const latest = (test) => store.activity.slice().reverse().find(test);
  [latest((ev) => ev.type === 'error' && (ev.data || {}).rebuilt), latest((ev) => ev.type === 'budget.warning' && (ev.data || {}).carried != null)].forEach((ev) => {
    if (ev && (ev.seq || 0) > seen) notes.push({ seq: ev.seq || 0, at: ev.ts, message: ev.summary || '' });
  });
  return { files, all, state, notes, total: all + notes.length + (state ? 1 : 0) };
}
export function dismissNotices(store) {
  const top = problemsOf(store).notes.reduce((max, n) => Math.max(max, n.seq), 0);
  if (top) storage.set(noticeKey(store.pid), top);
}
export function problemsHTML(store) {
  const { files, all, state, notes, total } = problemsOf(store);
  if (!total) return '';
  const item = (x) => { const page = x.path && store.byPath.get(x.path); return html`<li>${x.path ? (page ? html`<a class="prob-path" href="${link({ view: 'wiki', sel: page.path, sub: '' })}" data-sel="${page.path}"><code>${x.path}</code></a>` : html`<code class="prob-path">${x.path}</code>`) : ''} <span>${plain(x.message)}</span></li>`; };
  const shown = files.slice(0, 4), rest = files.slice(4);
  return html`<section class="ov-problems" id="problems" aria-label="Problems in the project's files"><div class="note note-warn">${icon('warn')}<div class="prob-body">
    ${state ? html`<p class="prob-note" data-state-problem>${plain(state)}</p>` : ''}
    ${notes.map((n) => html`<p class="prob-note"><b>${dateTime(n.at) ? cap1(dateTime(n.at)) + '.' : ''}</b> ${plain(n.message)}</p>`)}
    ${notes.length ? html`<p class="prob-act"><button class="linkish" data-act="notices-seen">I have read this</button></p>` : ''}
    ${files.length ? html`<p><b>${plural(all, 'problem')} in the project's files${all > files.length ? ', of which the first ' + files.length + ' are listed' : ''}.</b> The project opens all the same: this is what was read around, as the server reports it.</p>
      <ul class="problems">${shown.map(item)}</ul>
      ${rest.length ? html`<details class="prob-more" data-keep="problems"><summary>${rest.length} more</summary><ul class="problems">${rest.map(item)}</ul></details>` : ''}` : ''}
  </div></div></section>`;
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
  /* the words are one piece: the bar's gap belongs between the mark and the words, not inside them */
  const status = e.unverified ? html`${icon('warn')}Worker not verified`
    : e.unknown ? html`${icon('warn')}State unknown`
    : e.mode === 'running' ? html`<span class="live-dot" aria-hidden="true"></span><span>${round}${phaseName(e.phase) || 'starting'}</span>`
    : e.mode === 'experiment' ? html`<span class="live-dot" aria-hidden="true"></span><span><span class="ts-long">Experiment </span>running</span>`
    : e.mode === 'writing' ? html`<span class="live-dot" aria-hidden="true"></span>Writing the paper`
    : e.mode === 'failed' ? html`${icon('warn')}${cap1(phaseName(e.next || e.phase))} failed`
    : e.died || e.crashed ? html`${icon('warn')}The agents stopped`
    : e.mode === 'stopped' ? html`${PAUSE}Stopped by you`
    : e.fresh ? html`${RING}Not started`
    : e.budget ? html`${icon('warn')}Budget reached`
    : html`${PAUSE}<span>${round}checkpoint</span>`;
  const cls = e.working ? 'running' : e.mode === 'failed' || e.died || e.crashed ? 'failed' : e.budget || e.unverified || e.unknown ? 'budget' : e.mode;
  const probs = problemsOf(store).total;
  const limits = (p.settings || {}).limits || {};
  const spent = (p.state || {}).spent || {};
  return html`<a class="brand" href="#/library" data-tip="All projects">${MARK}<span>PaperOrchestra</span></a><span class="top-sep" aria-hidden="true">/</span>
    <a class="top-proj" href="${link({ view: 'overview', sel: '', sub: '' })}" data-tip="${p.title}">${p.title}</a>
    <nav class="views" aria-label="Views of this project">${VIEWS.map((v, i) => html`<a href="${hrefOf({ name: 'project', pid: route.pid, view: v[0], sel: v[0] === 'paper' ? '' : route.sel })}"${route.view === v[0] ? raw(' aria-current="page"') : ''} data-tip="${v[1]} (${i + 1})">${v[1]}${v[0] === 'inbox' && n ? html`<span class="count" aria-label="${n} waiting">${n}</span>` : ''}</a>`)}</nav>
    <div class="top-r">${probs ? html`<a class="top-prob" href="${link({ view: 'overview', sel: '', sub: '' })}" data-act="show-problems" data-tip="${plural(probs, 'problem')} in the project's files: see the overview" aria-label="${plural(probs, 'problem')} in the project's files">${icon('warn')}<span>${probs}</span></a>` : ''}
      <a class="top-status st-${cls}" href="${link({ view: 'overview', sel: '', sub: '' })}" data-tip="Where things stand">${status}</a>
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
  /* The person's approval of a run is two records, written in the same moment: how the
     experiment may run, then "is now approved". They are one thing the person did, and
     one line: "You approved experiment E4 … to run on this machine, for up to 300 s, …". */
  const lines = [];
  out.forEach((ev) => {
    const prev = lines[lines.length - 1], d = ev.data || {}, pd = (prev && prev.data) || {};
    if (prev && !prev.run && !ev.run && ev.type === 'experiment.state' && d.by === 'human' && d.state === 'approved'
        && prev.type === 'page.updated' && pd.by === 'human' && pd.approval && pd.id === d.id) {
      const said = /^You set how (.+) may run: (.+)$/.exec(prev.summary || '');
      lines[lines.length - 1] = Object.assign({}, ev, { summary: said ? 'You approved ' + said[1] + ' to run ' + said[2].replace(/\bon local\b/, 'on this machine') : prev.summary });
      return;
    }
    lines.push(ev);
  });
  return lines;
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
  return lines.map((ev) => html`<div><time>${hm(ev.ts)}</time><b>${ev.run ? 'runner' : whoOf(ev)}</b><span>${ev.run ? runSentence(ev) : plain(ev.summary)}</span></div>`);
}
/** Stop, for a worker that is this server's to stop. */
function stopButton(store, e) {
  return html`<button class="btn" data-act="stop"${e.stopping ? raw(' disabled') : ''}>${icon('stop')}${e.stopping ? 'Stopping…' : 'Stop'}</button>`;
}
/** For a worker the server cannot verify there is no Stop: it is not signalled. It can be
    forgotten, which clears its name from the project's files and stops nothing. */
export const forgetButton = (large) => html`<button class="btn${large ? ' btn-l' : ''}" data-act="forget-worker"${app.ui.forgetting ? raw(' disabled') : ''} data-tip="Clears the worker's name from the project's files. Nothing is signalled or stopped; if a worker is in fact at work, it shows again.">${app.ui.forgetting ? 'Forgetting…' : 'Forget this worker'}</button>`;
export const UNVERIFIED = 'The project\'s files name a worker that this server did not start and cannot show to be this project\'s.';
function workerLogHTML(store) {
  const log = store.workerLog;
  if (!log) return '';
  if (log.problem) return html`<p class="quiet">${plain(log.problem)}</p>`;
  if (log.text == null) return html`<p class="quiet">Reading what the worker wrote…</p>`;
  if (!log.text.trim()) return html`<p class="quiet">The worker wrote nothing to <code>worker.log</code>.</p>`;
  return html`<pre class="strip-pre" data-follow>${log.text.split('\n').slice(-60).join('\n')}</pre>`;
}
export function strip(store) {
  const e = engineView(store), n = store.pending().length;
  const open = app.ui.activityOpen;
  const toggle = (what) => html`<button class="btn btn-quiet" data-act="toggle-activity" aria-expanded="${tf(open)}">${open ? 'Hide' : 'Show'} ${what} ${icon(open ? 'down' : 'right')}</button>`;
  /* what a press does where the button cannot say all of it, the reasons for more research,
     and, before either, what the engine says about a state it had to read from the events */
  const notes = continueNotes(store, e), warn = stateNoteHTML(e);
  const sub = String(notes) || String(warn) ? html`<div class="strip-sub">${warn}${notes}</div>` : '';
  if (e.unverified) {
    return html`<div class="strip s-unverified" role="status"><span class="strip-mark">${icon('warn')}</span><span class="strip-kicker">A worker this server cannot verify</span>
      <span class="strip-text">${plain(e.note) || UNVERIFIED}</span>
      <span class="strip-actions">${forgetButton()}</span></div>`;
  }
  if (e.unknown && !e.crashed) {
    return html`<div class="strip s-unverified s-unknown" role="status"><span class="strip-mark">${icon('warn')}</span><span class="strip-kicker">Where it stands is not known</span>
      <span class="strip-text">${plain(e.stateNote) || 'Nothing on file says which round and phase this project is in. Choose the phase to go on with.'}</span>
      <span class="strip-actions">${continueControl(store, e, 'strip')}</span></div>`;
  }
  if (e.mode === 'running') {
    const since = (parseStamp(e.started) || 0) - 2000;
    const last = store.activity.slice().reverse().find((ev) => /^(agent\.activity|page\.created|phase\.started|proposal\.created|run\.started|run\.finished|experiment\.state)$/.test(ev.type) && (parseStamp(ev.ts) || 0) >= since);
    return html`<div class="strip s-running" role="status"><span class="live-dot" aria-hidden="true"></span><span class="strip-kicker">Round ${e.round || 1} · ${phaseName(e.phase) || 'starting'}</span>
      <span class="strip-text">${e.stopping ? 'Stopping: the phase ends at its next step and can be resumed.' : last ? (String(last.summary || '').toLowerCase().startsWith('the ' + String(last.agent || 'engine').toLowerCase() + ' ') ? plain(last.summary) : html`<b>${last.agent || 'engine'}</b> ${plain(last.summary)}`) : 'Starting…'}</span>
      ${e.started ? html`<span class="strip-clock" data-since="${e.started}">${clock((Date.now() - (parseStamp(e.started) || Date.now())) / 1000)}</span>` : ''}
      <span class="strip-actions">${toggle('activity')}${stopButton(store, e)}</span></div>
      ${open ? html`<div class="strip-log" data-follow>${activityLines(store, 40)}</div>` : ''}`;
  }
  if (e.mode === 'experiment' || e.mode === 'writing') {
    const running = store.experiments.find((x) => (x.state === 'running' && !x.interrupted) || x.state === 'queued');
    return html`<div class="strip s-running" role="status"><span class="live-dot" aria-hidden="true"></span><span class="strip-kicker">${e.mode === 'writing' ? 'Writing the paper' : 'Experiment'}</span>
      <span class="strip-text">${e.stopping ? 'Stopping…' : e.mode === 'writing' ? 'The write-up is being produced from the wiki.' : running ? html`<b>${running.id}</b> ${running.title}` : 'A run is starting.'}</span>
      ${e.started ? html`<span class="strip-clock" data-since="${e.started}">${clock((Date.now() - (parseStamp(e.started) || Date.now())) / 1000)}</span>` : ''}
      <span class="strip-actions">${e.mode === 'writing' ? html`<a class="btn" href="${link({ view: 'paper', sel: '', sub: '' })}">Open the paper</a>` : running ? html`<a class="btn" href="${link({ view: 'experiments', sel: running.path, sub: '' })}">Watch the run</a>` : ''}
        ${stopButton(store, e)}</span></div>`;
  }
  if (e.mode === 'failed' || e.died || e.crashed) {
    const phase = e.next || e.phase;
    const title = e.mode === 'failed' ? cap1(phaseName(phase)) + ' failed' : 'The agents stopped unexpectedly';
    const text = e.mode === 'failed' ? plain(e.error) : e.died ? plain(e.error) : crashText(store, e.exit);
    return html`<div class="strip s-failed" role="alert"><span class="strip-mark">${icon('warn')}</span><span class="strip-kicker">${title}</span>
      <span class="strip-text">${text}</span>
      <span class="strip-actions">${toggle('log')}<a class="btn" href="#/setup">Open Setup</a>${continueControl(store, e, 'strip')}</span></div>${sub}
      ${open ? html`<div class="strip-log"><p class="strip-fix">What was written so far is kept. Put the cause right (credentials are checked under Setup), then retry: the phase resumes where it stopped.</p>${workerLogHTML(store)}${activityLines(store, 12)}</div>` : ''}`;
  }
  if (e.fresh) {
    return html`<div class="strip s-idle"><span class="strip-mark">${RING}</span><span class="strip-kicker">Not started</span><span class="strip-text">Research has not started. Nothing has been spent.</span>
      <span class="strip-actions">${continueControl(store, e, 'strip')}</span></div>${sub}`;
  }
  if (e.mode === 'stopped') {
    return html`<div class="strip s-checkpoint"><span class="strip-mark">${PAUSE}</span><span class="strip-kicker">Stopped · round ${e.round || 1}, during ${phaseName(e.next || e.phase)}</span>
      <span class="strip-text">You stopped the agents. What they had written is kept; the phase resumes where it stopped.</span>
      <span class="strip-actions">${n ? html`<a class="btn" href="${link({ view: 'inbox', sel: '', sub: '' })}">Review ${n}</a>` : ''}${continueControl(store, e, 'strip')}</span></div>${sub}`;
  }
  if (e.budget) {
    const editing = app.ui.limitEdit;
    return html`<div class="strip s-budget" role="status"><span class="strip-mark">${icon('warn')}</span><span class="strip-kicker">Budget reached · round ${e.round || 1}</span>
      <span class="strip-text">${e.budget}</span>
      <span class="strip-actions">${editing ? html`<form class="limit-form" data-form="limit"><label class="sr" for="limit-usd">New limit for models, in dollars</label><span class="quiet">Models, up to $</span><input id="limit-usd" name="usd" data-input="limit" inputmode="decimal" value="${editing.value}" size="6"><button class="btn btn-primary" type="submit">Set the limit</button><button class="btn btn-quiet" type="button" data-act="limit-cancel">Cancel</button></form>`
        : html`<button class="btn" data-act="limit-edit">Raise the limit…</button>${continueControl(store, e, 'strip')}`}</span></div>${sub}`;
  }
  return html`<div class="strip s-checkpoint"><span class="strip-mark">${PAUSE}</span>
    <span class="strip-kicker">Checkpoint · round ${e.round || 1}${e.phase ? (e.lifted ? ', during ' : ', after ') + phaseName(e.phase) : ''}</span><span class="strip-text">${e.lifted ? 'The limit for models is raised. ' + cap1(phaseName(e.next || e.phase)) + ' stopped at the old one and can go on where it stopped. ' : ''}${waitingSentence(store)}</span>
    <span class="strip-actions">${n ? html`<a class="btn" href="${link({ view: 'inbox', sel: '', sub: '' })}">Review ${n}</a>` : ''}
      ${continueControl(store, e, 'strip')}</span></div>${sub}`;
}

/* ---------- the status bar ---------- */
export function statusbar(store) {
  const p = store.project, e = engineView(store), n = store.pending().length;
  const limits = (p.settings || {}).limits || {}, spent = (p.state || {}).spent || {};
  const running = store.experiments.find((x) => x.state === 'running' && !x.interrupted);
  const run = running && running.last_run && running.last_run.status === 'running' ? running.last_run : null;
  /* what comes next, as the server says a press of Continue would have it; where it has not said, the phase the state names */
  const step = nextStep(store, e);
  const to = step.steps.length ? step.steps.map(phaseName).join(', ') + (step.back ? ', then ' + phaseName(step.back) : '') : phaseName(e.next);
  const state = e.unverified ? html`<span class="sb-prop">${icon('warn')}<b>Worker not verified</b></span>`
    : e.unknown ? html`<span class="sb-prop">${icon('warn')}<b>State unknown</b></span>`
    : e.mode === 'running' ? html`<span class="sb-live"><span class="live-dot" aria-hidden="true"></span><b>Running</b> round ${e.round || 1}${e.phase ? ' · ' + phaseName(e.phase) : ''}</span>`
    : e.mode === 'experiment' ? html`<span class="sb-live"><span class="live-dot" aria-hidden="true"></span><b>Running</b> an experiment</span>`
    : e.mode === 'writing' ? html`<span class="sb-live"><span class="live-dot" aria-hidden="true"></span><b>Writing</b> the paper</span>`
    : e.mode === 'failed' ? html`<span class="sb-neg">${icon('warn')}<b>Failed</b> ${phaseName(e.next || e.phase)}</span>`
    : e.died || e.crashed ? html`<span class="sb-neg">${icon('warn')}<b>Stopped</b> unexpectedly</span>`
    : e.mode === 'stopped' ? html`<span>${PAUSE}<b>Stopped</b> during ${phaseName(e.next || e.phase)}</span>`
    : e.fresh ? html`<span>${RING}<b>Not started</b></span>`
    : html`<span class="sb-check">${PAUSE}<b>${e.budget ? 'Budget reached' : 'Checkpoint'}</b> round ${e.round || 1}${e.phase ? html` · ${phaseName(e.phase)}${to && to !== phaseName(e.phase) ? html` → ${to}` : ''}` : ''}</span>`;
  const live = connection.denied ? html`<span class="sb-neg" data-tip="${RESTARTED}">new link needed</span>` : !connection.ok ? html`<span class="sb-neg" data-tip="The server is not answering">offline</span>` : store.live === 'reconnecting' ? html`<span class="sb-prop" data-tip="The live connection dropped; it is being opened again">reconnecting…</span>` : html`<span class="sb-ok" data-tip="Changes arrive as they happen">live</span>`;
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
  /* The server answers, and no longer lets this page in: it was restarted and has a new
     access token. Nothing is tried again by itself; the way back in is the new link, and
     once that was opened in this browser a look (or coming back to this window) is enough. */
  if (connection.denied) return html`<div class="banner-in" data-denied>${icon('warn')}<span><b>${RESTARTED}</b> The link carries the server's new access token; once it was opened in this browser, this page works again. Nothing you typed is lost: drafts stay on this device.</span><button class="btn btn-s" data-act="retry-connection">Try again</button></div>`;
  /* what becomes of the agents and the runs when the server ends is the server's own word
     (`stops_workers_on_exit` of its last answer to /api/health); where it never said, both
     are named */
  const stops = (app.ui.health || {}).stops_workers_on_exit;
  const workers = stops === true ? 'This server was started to stop agents and experiments when it ends: if it has ended, they were stopped with it, and what they had written is kept.'
    : stops === false ? 'Agents and experiments that were running go on without the server, and the app finds them again when it is back.'
      : 'What becomes of agents and experiments that were running depends on how the server was started: as a rule they go on without it. The app finds them again when it is back.';
  return html`<div class="banner-in">${icon('warn')}<span><b>The server is not answering.</b> Nothing you typed is lost: drafts stay on this device. <span data-workers="${stops === true ? 'stopped' : stops === false ? 'go-on' : 'unknown'}">${workers}</span> It is tried again every few seconds.</span><button class="btn btn-s" data-act="retry-connection">Retry now</button></div>`;
}

