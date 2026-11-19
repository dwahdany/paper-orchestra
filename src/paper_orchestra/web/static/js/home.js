/* The screens outside a project: the composer that starts one, the library, the setup check. */

import { app, defaults } from './core.js';
import { api } from './api.js';
import { MONTH_NAMES, ago, bytes, checkpointDid, codeSpans, cutoffFrom, cutoffParts, cutoffWords, html, money, plain, plural, raw, setHTML, storage } from './util.js';
import { KIND, STAGES, computeName, credentialName, effortName, icon, phaseName, runtimeName } from './vocab.js';
import { stripMapSmall } from './overview.js';
import { DEMO_COST, agentFields } from './shell.js';
import { capsWords } from './autopilot.js';

/* ---------- start: one box for the idea ---------- */
const EXAMPLE = 'Learning-rate warmup is assumed to matter most for large batches. I suspect that with adaptive optimisers it is small batches that need it, because early second-moment estimates are noisy.';
const FILE_NOTE = { md: 'notes · read as text', txt: 'text', csv: 'table · read as text', tsv: 'table · read as text', json: 'data · read as text', png: 'image · shown to the model', jpg: 'image · shown to the model', jpeg: 'image · shown to the model', gif: 'image · shown to the model', webp: 'image · shown to the model', pdf: 'paper · read as a PDF', ipynb: 'notebook · read as text', py: 'code · read as text', tex: 'LaTeX · read as text', bib: 'references · read as text' };
const COMPOSER_KEY = 'po-composer';
/** How a kind is reviewed when nothing else is said: the server's default (`settings.review`
    of /api/defaults), "review" or "auto"; `null` until the defaults are known. */
const reviewDefault = (kind) => { const d = defaults(); const policy = d && d.settings && d.settings.review; return policy && typeof policy[kind] === 'string' ? policy[kind] : null; };
/** The kinds the review options offer, in the order research flows: those the server's
    defaults name, without the records and notes whose review is not the person's business. */
function reviewKinds() {
  const d = defaults(), policy = (d && d.settings && d.settings.review) || {};
  return STAGES.filter((kind) => kind !== 'idea' && kind !== 'source' && kind !== 'result' && policy[kind] != null);
}
/** The values a setting accepts, as the server lists them (`efforts`, `compute_backends` of
    /api/defaults). Until they are known the list is empty, and the one choice is the
    project's default; a choice kept from before a reload is offered until then. */
function accepted(name, kept) {
  const d = defaults();
  const list = d && Array.isArray(d[name]) ? d[name].filter((x) => typeof x === 'string' && x) : [];
  return kept && list.indexOf(kept) < 0 && !d ? list.concat([kept]) : list;
}
/** How the agents of a new project are run when the composer says nothing: the server's
    `settings.agents`, whose runtime follows the server's environment; or `null`. */
const agentDefaults = () => { const d = defaults(); const a = d && d.settings && d.settings.agents; return a && typeof a === 'object' ? a : null; };
/** What the composer's choices about the agents come to: the person's where one was made, the
    server's default otherwise; `null` until the defaults are known. */
/** What a person should know before they press Start when the agents are Claude agents
    with a shell: on which model, on whose account, and that fast mode costs more. Said on
    the page itself, not only inside the closed Options. */
function agentsWord(a) {
  const fast = a.fast && a.credential === 'login';
  return 'The agents are Claude agents with a shell on ' + a.model + (fast ? ' in fast mode' : '') + ', billed to '
    + (a.credential === 'login' ? 'your own Claude account' : 'the account of the server\'s federation settings')
    + (fast ? '; fast mode costs several times the usual price, and the start of each agent alone can cost about a dollar' : '')
    + '. Change it under Options.';
}
function agentChoice(s) {
  const a = agentDefaults();
  if (!a) return null;
  const flag = (name) => (typeof s[name] === 'boolean' ? s[name] : !!a[name]);
  return { runtime: s.runtime || a.runtime, model: String(s.model || '').trim() || a.model, credential: s.credential || a.credential, fast: flag('fast'), ultracode: flag('ultracode'), sandbox: flag('sandbox') };
}
/** The limits a new project gets when the composer says nothing: the server's, or `null`. */
const limitDefaults = () => { const d = defaults(); return (d && d.settings && d.settings.limits) || null; };

/** The composer's state. The text and the options survive a reload; files cannot (the
    browser does not let a page keep them), and the composer says so when that happens. */
export function composer() {
  if (!app.ui.start) {
    const saved = storage.get(COMPOSER_KEY, {}) || {};
    /* a cutoff kept from before it had a field for the year and one for the month */
    const old = cutoffParts(saved.cutoff);
    app.ui.start = {
      text: saved.text || '', files: [], adv: !!saved.adv, lostFiles: saved.fileCount || 0,
      title: saved.title || '', effort: saved.effort || '', compute: saved.compute || '',
      cutoffYear: saved.cutoffYear || (old && old.raw == null ? old.year : ''), cutoffMonth: +saved.cutoffMonth || (old && old.raw == null ? old.month : 0),
      model_usd: saved.model_usd || '', exp_minutes: saved.exp_minutes || '', exp_usd: saved.exp_usd || '',
      /* how the agents are run: empty, or `null` for a tick, where the person chose nothing and the server's default holds */
      runtime: saved.runtime || '', model: saved.model || '', credential: saved.credential || '',
      fast: typeof saved.fast === 'boolean' ? saved.fast : null, ultracode: typeof saved.ultracode === 'boolean' ? saved.ultracode : null, sandbox: typeof saved.sandbox === 'boolean' ? saved.sandbox : null,
      /* autopilot: `null` where the person chose nothing and the server's default holds */
      autopilot: typeof saved.autopilot === 'boolean' ? saved.autopilot : null,
      /* kind -> true: accepted as it is written; false: waits for the person. Only what the person changed. */
      auto: saved.auto || {}, uploading: false, progress: 0, error: null, request: null,
    };
  }
  return app.ui.start;
}
export function keepComposer() {
  const s = composer();
  storage.set(COMPOSER_KEY, { text: s.text, adv: s.adv, title: s.title, effort: s.effort, cutoffYear: s.cutoffYear, cutoffMonth: s.cutoffMonth, compute: s.compute, model_usd: s.model_usd, exp_minutes: s.exp_minutes, exp_usd: s.exp_usd, runtime: s.runtime, model: s.model, credential: s.credential, fast: s.fast, ultracode: s.ultracode, sandbox: s.sandbox, autopilot: s.autopilot, auto: s.auto, fileCount: s.files.length });
}
export function clearComposer() { app.ui.start = null; storage.remove(COMPOSER_KEY); }
export const composerReady = (s) => (s.text.trim().length > 0 || s.files.length > 0) && !s.uploading;
export const totalSize = (s) => s.files.reduce((a, f) => a + f.size, 0);
/** The settings the person changed, as the partial object the API merges into its defaults. */
export function composerSettings(s) {
  const settings = {};
  /* the effort is the tool loop's models': agents with a shell do not take it, and the form does not offer it then */
  if (s.effort && (agentChoice(s) || {}).runtime !== 'sdk') settings.models = { effort: s.effort };
  const cutoff = cutoffFrom(s.cutoffYear, s.cutoffMonth);
  if (cutoff.value) settings.research_cutoff = cutoff.value;
  if (s.compute) settings.compute = s.compute;
  const limits = {};
  if (String(s.model_usd).trim() !== '') limits.model_usd = Number(String(s.model_usd).replace(/[$,\s]/g, ''));
  if (String(s.exp_minutes).trim() !== '') limits.experiment_seconds = Math.round(Number(String(s.exp_minutes).replace(/[^\d.]/g, '')) * 60);
  if (String(s.exp_usd).trim() !== '') limits.experiment_usd = Number(String(s.exp_usd).replace(/[$,\s]/g, ''));
  if (Object.keys(limits).length) settings.limits = limits;
  /* only where the person's choice differs from what the project would do anyway */
  const review = {};
  Object.keys(s.auto || {}).forEach((kind) => { const wanted = s.auto[kind] ? 'auto' : 'review'; if (wanted !== (reviewDefault(kind) || 'review')) review[kind] = wanted; });
  if (Object.keys(review).length) settings.review = review;
  /* how the agents are run: only what the person chose. A runtime that is not named is the
     server's to choose, from its environment (PAPER_ORCHESTRA_AGENT_RUNTIME). */
  const agents = {};
  if (s.runtime) agents.runtime = s.runtime;
  if (String(s.model || '').trim()) agents.model = String(s.model).trim();
  if (s.credential) agents.credential = s.credential;
  ['fast', 'ultracode', 'sandbox'].forEach((name) => { if (typeof s[name] === 'boolean') agents[name] = s[name]; });
  if (Object.keys(agents).length) settings.agents = agents;
  /* who decides: only where the person said so (the server's default may be autopilot) */
  if (typeof s.autopilot === 'boolean') settings.autonomy = { mode: s.autopilot ? 'autopilot' : 'supervised' };
  return settings;
}

const ext = (name) => (String(name).split('.').length > 1 ? String(name).split('.').pop().toLowerCase().slice(0, 5) : 'file');

export const startView = {
  render(root, route) {
    const s = composer();
    if (route.q && route.q.adv) s.adv = true;
    const ready = composerReady(s);
    const n = s.files.length;
    /* every number the composer shows before the person types one is the server's default */
    const limits = limitDefaults();
    const d = defaults(), models = (d && d.settings && d.settings.models) || {};
    const typed = Number(String(s.model_usd).replace(/[$,\s]/g, ''));
    const limit = String(s.model_usd).trim() !== '' && typed > 0 ? money(typed) : limits && limits.model_usd != null ? money(limits.model_usd) : '';
    const hint = (value, unit) => (value == null ? 'the default' : value + ' ' + unit);
    const cutoff = cutoffFrom(s.cutoffYear, s.cutoffMonth), cutoffNow = cutoff.value ? cutoffParts(cutoff.value) : null;
    /* a choice kept from an earlier visit that the server no longer lists is not sent behind a list that does not show it */
    if (d) { if (s.effort && accepted('efforts').indexOf(s.effort) < 0) s.effort = ''; if (s.compute && accepted('compute_backends').indexOf(s.compute) < 0) s.compute = ''; }
    if (d) { if (s.runtime && accepted('agent_runtimes').indexOf(s.runtime) < 0) s.runtime = ''; if (s.credential && accepted('agent_credentials').indexOf(s.credential) < 0) s.credential = ''; }
    /* how the agents are run: the server's default named as the default, the person's choice beside it */
    const agents = agentChoice(s), agentBase = agentDefaults();
    /* who decides: the server's default (it may be autopilot) until the person ticks */
    const autonomy = d && d.autonomy, autopilot = typeof s.autopilot === 'boolean' ? s.autopilot : !!autonomy && autonomy.mode === 'autopilot';
    /* why autopilot would approve no run of a project with this runtime and backend: the server's sentence, or nothing */
    const runsWait = (((d && d.autopilot_runs_wait) || {})[(agents && agents.runtime) || 'classic'] || {})[s.compute || (d && d.settings && d.settings.compute) || ''] || '';
    const offer = (list, kept, base, name) => [['', base ? 'Default (' + name(base) + ')' : 'As the project default']].concat(accepted(list, kept).map((v) => [v, name(v)]));
    const kinds = reviewKinds();
    const waits = (kind) => (s.auto[kind] != null ? !s.auto[kind] : reviewDefault(kind) !== 'auto');
    setHTML(root, html`<div class="start"><div class="start-col">
      <h1>What do you want to find out?</h1>
      <p class="start-lede">Describe the idea in a few sentences and add anything you already have. Agents research it and fill a wiki with sources, questions, hypotheses and experiments. They only suggest: nothing is accepted, and nothing runs, without you.</p>
      <form class="composer${s.uploading ? ' busy' : ''}" data-form="start" data-drop>
        <label class="sr" for="idea-text">Your idea</label><textarea id="idea-text" name="text" data-input="start-text" rows="5" placeholder="What do you suspect, and why? What would convince you?"${s.uploading ? raw(' readonly') : ''}>${s.text}</textarea>
        ${n ? html`<ul class="files" aria-label="Attached files">${s.files.map((f, i) => html`<li class="filechip"><span class="ft">${ext(f.name)}</span><span class="fn" data-tip="${f.name}">${f.name}</span><span class="fs">${bytes(f.size)} · ${FILE_NOTE[ext(f.name)] || 'file'}</span>
          <button class="ibtn" type="button" data-act="file-del" data-i="${i}" aria-label="Remove ${f.name}"${s.uploading ? raw(' disabled') : ''}>${icon('x')}</button></li>`)}</ul>` : ''}
        ${s.lostFiles && !n ? html`<p class="composer-note">${icon('warn')}Your text was kept across the reload. The ${plural(s.lostFiles, 'file')} you had attached could not be: a page cannot keep files. Add ${s.lostFiles === 1 ? 'it' : 'them'} again. <button class="linkish" type="button" data-act="lost-ok">Dismiss</button></p>` : ''}
        ${s.uploading ? html`<div class="upload" role="status"><progress id="upload-bar" max="1" value="${s.progress}" aria-label="Upload progress"></progress><span id="upload-say">${s.progress >= 1 ? 'Sent. The project is being created…' : 'Sending ' + (n ? plural(n, 'file') + ', ' + bytes(totalSize(s)) : 'the idea') + '…'}</span><button class="btn btn-quiet btn-s" type="button" data-act="upload-cancel">Cancel</button></div>` : ''}
        ${s.error ? html`<div class="note note-neg composer-error" role="alert">${icon('warn')}<span><b>The project was not created.</b> ${s.error} Your text and files are still here.</span></div>` : ''}
        <div class="composer-bar"><button class="btn btn-quiet" type="button" data-act="file-add"${s.uploading ? raw(' disabled') : ''}>${icon('plus')}Add files</button><span class="quiet drop-hint">or drop notes, data, figures, papers, code, a folder</span>
          <input type="file" id="file-input" class="sr" multiple tabindex="-1" aria-label="Files to attach">
          <span class="composer-go">${s.text || n ? '' : html`<button class="btn btn-quiet" type="button" data-act="start-example">Use an example</button>`}
            <button class="btn btn-primary btn-l" type="submit"${ready ? '' : raw(' disabled')}>Start research ${icon('right')}</button></span></div>
        <div class="drop-veil" aria-hidden="true">Drop to attach</div></form>
      <p class="start-est">${ready ? 'First it reads ' + (n ? 'your ' + plural(n, 'file') + ' and ' : '') + 'the idea, then looks for prior work, and pauses at the first checkpoint.' + (limit ? ' It stops when ' + limit + ' has been spent on models.' : ' It stops at the project\'s limit for model spend.') : 'A few sentences are enough. You can edit the idea later; it stays yours and no agent changes it.'}${agents && agents.runtime === 'sdk' ? html`<span class="start-agents" data-agents-say="${agents.credential}">${agentsWord(agents)}</span>` : ''}</p>
      <details class="adv"${s.adv ? raw(' open') : ''}><summary data-act="adv">Options <span class="quiet">title, model effort, a research cutoff, limits, what needs your review, how the agents work</span></summary><div class="adv-grid">
        <label><span>Title</span><input id="opt-title" data-input="opt" data-name="title" value="${s.title}" placeholder="Named from the idea" autocomplete="off"></label>
${agents && agents.runtime === 'sdk' ? '' : html`        <label><span>Model effort</span><select id="opt-effort" data-input="opt" data-name="effort">${[['', models.effort ? 'Default (' + effortName(models.effort) + ')' : 'As the project default']].concat(accepted('efforts', s.effort).map((v) => [v, effortName(v)])).map(([v, t]) => html`<option value="${v}"${s.effort === v ? raw(' selected') : ''}>${t}</option>`)}</select></label>`}
        <div class="adv-cutoff" role="group" aria-labelledby="opt-cutoff-l"><span id="opt-cutoff-l">Only literature before</span><span class="cutoff-fields">
          <label class="sr" for="opt-cutoff-month">Month of the research cutoff</label><select id="opt-cutoff-month" data-input="opt" data-name="cutoffMonth"><option value="0"${+s.cutoffMonth ? '' : raw(' selected')}>any month</option>${MONTH_NAMES.map((name, i) => html`<option value="${i + 1}"${+s.cutoffMonth === i + 1 ? raw(' selected') : ''}>${name}</option>`)}</select>
          <label class="sr" for="opt-cutoff-year">Year of the research cutoff</label><input id="opt-cutoff-year" data-input="opt" data-name="cutoffYear" value="${s.cutoffYear}" placeholder="year" inputmode="numeric" maxlength="4" autocomplete="off"></span>
          <span class="quiet cutoff-hint" data-cutoff>${cutoff.error ? cutoff.error : cutoffNow ? 'Papers published in ' + cutoffWords(cutoffNow) + ' or later are left out.' : 'Optional. Without a year, literature of any date is in scope.'}</span></div>
        <label><span>Experiments run on</span><select id="opt-compute" data-input="opt" data-name="compute">${[['', d && d.settings && d.settings.compute ? 'Default (' + computeName(d.settings.compute, true) + ')' : 'As the project default']].concat(accepted('compute_backends', s.compute).map((v) => [v, computeName(v)])).map(([v, t]) => html`<option value="${v}"${s.compute === v ? raw(' selected') : ''}>${t}</option>`)}</select></label>
        <label><span>Model spend limit</span><input id="opt-model-usd" data-input="opt" data-name="model_usd" value="${s.model_usd}" placeholder="${limits && limits.model_usd != null ? Number(limits.model_usd).toFixed(2) + ' dollars' : 'the default'}" inputmode="decimal" autocomplete="off"></label>
        <label><span>Time limit per run</span><input id="opt-exp-min" data-input="opt" data-name="exp_minutes" value="${s.exp_minutes}" placeholder="${limits && limits.experiment_seconds != null ? hint(Math.round(limits.experiment_seconds / 6) / 10, 'minutes') : 'the default'}" inputmode="decimal" autocomplete="off"></label>
        <label><span>Experiment spend limit</span><input id="opt-exp-usd" data-input="opt" data-name="exp_usd" value="${s.exp_usd}" placeholder="${limits && limits.experiment_usd != null ? Number(limits.experiment_usd).toFixed(2) + ' dollars' : 'the default'}" inputmode="decimal" autocomplete="off"></label>
        <fieldset class="adv-review"><legend>Needs your review before it counts</legend>${kinds.length ? kinds.map((kind) => html`<label class="tick"><input type="checkbox" data-input="opt-review" data-kind="${kind}"${waits(kind) ? raw(' checked') : ''}> ${KIND[kind].many}</label>`) : html`<span class="quiet">As the project default.</span>`}
          ${reviewDefault('source') === 'auto' ? html`<label class="tick"><input type="checkbox" disabled> Sources <span class="quiet">verified by code, accepted on creation</span></label>` : ''}</fieldset>
        ${agents ? html`<fieldset class="adv-agents"><legend>How the agents work</legend>${agentFields(agents, { id: 'opt-agent', input: 'opt', tick: 'opt-agent',
          runtimes: offer('agent_runtimes', s.runtime, agentBase.runtime, runtimeName), credentials: offer('agent_credentials', s.credential, agentBase.credential, credentialName),
          picked: { runtime: s.runtime, model: s.model, credential: s.credential }, modelHint: agentBase.model })}</fieldset>` : ''}
        ${autonomy ? html`<fieldset class="adv-agents adv-auto"><legend>Who decides</legend><label class="tick"><input type="checkbox" id="opt-autopilot" data-input="opt-autopilot"${autopilot ? raw(' checked') : ''}> <span>Autopilot: go on without me <span class="quiet">${autopilot ? (runsWait ? 'Pages are accepted, edits applied and checkpoints passed without you. Runs on this machine are not approved without you: ' + runsWait + '.' : 'Pages are accepted, edits applied, runs approved and checkpoints passed without you.') + ' One stretch is held to ' + capsWords(autonomy) + '; the caps are changed on the project\'s overview.' : 'Off: every proposal, edit, run and checkpoint waits for you.'}</span></span></label>
          ${autopilot && agents && agents.runtime === 'sdk' && agents.sandbox === false ? html`<p class="note note-warn">${icon('warn')}<span>Autopilot is refused for agents that work outside the sandbox: the project would not be created.</span></p>` : ''}</fieldset>` : ''}</div></details>
      <div class="start-route" aria-label="What happens next"><div class="smap smap-route">
        ${[['Idea', 'yours'], ['Sources', 'verified papers'], ['Topics', 'what is known'], ['Questions', 'what is open'], ['Hypotheses', 'you accept or reject'], ['Experiments', 'you approve each run'], ['Results', 'written by code'], ['Findings', 'traced to results']].map((x, i) => html`<div class="smap-st${i ? ' none' : ' here'}"><div class="smap-dot"></div><div class="smap-name">${x[0]}</div><div class="smap-hint">${x[1]}</div></div>`)}</div></div>
    </div></div>`);
  },
};
export { EXAMPLE };

/* ---------- library ---------- */
/** The projects of the workspace as the server lists them: each row with its live counts
    (what waits for the person, edit proposals included), and beside them the directories
    that hold a project which cannot be opened, each with the server's reason. */
export function loadLibrary(force) {
  const lib = app.ui.library || (app.ui.library = { projects: null, unreadable: [], error: null, loading: false, at: 0 });
  if (lib.loading || (!force && lib.projects && Date.now() - lib.at < 4000)) return;
  lib.loading = true;
  api.projects().then((data) => {
    lib.projects = data.projects || [];
    lib.unreadable = data.unreadable || [];
    lib.error = null;
    lib.at = Date.now();
  }, (error) => { lib.error = error; }).finally(() => {
    lib.loading = false;
    const now = JSON.stringify([lib.projects, lib.unreadable, lib.error && lib.error.message]);   // looked at again every few seconds: drawn again only when it changed
    if (now !== lib.drawn && app.route.name === 'library') { lib.drawn = now; app.render('library', true); }
  });
}

/** What a project holds, in a line under its small map: "14 sources · 5 hypotheses · 1
    experiment · 2 findings"; "idea only" where nothing but the idea is there. */
export function countsLine(by) {
  if (!STAGES.slice(1).some((k) => (by || {})[k])) return 'idea only';
  return ['source', 'hypothesis', 'experiment', 'finding'].map((k) => plural(by[k] || 0, KIND[k].one.toLowerCase(), KIND[k].many.toLowerCase())).join(' · ');
}
/** How many proposals wait in a project, in words: the live number the row carries. */
function waitingWords(row) {
  /* with the runs autopilot left for the person: they wait as a proposal does (the Inbox counts them too) */
  const n = (row.pending || 0) + (row.held || 0);
  /* (what the agents asked the person to choose is counted apart: it holds nothing up) */
  const asked = ((row.counts || {}).decisions || {}).open || 0;
  const decide = asked ? asked + (asked === 1 ? ' decision' : ' decisions') + ' to answer' : '';
  if (!n) return decide;
  const pending = (row.counts || {}).pending;
  /* while the pages of a project that changed are still to be counted, the number is the last checkpoint's */
  return n + ' waiting for you' + (pending ? '' : ' at the last checkpoint') + (decide ? ' · ' + decide : '');
}
/** Where a row says the project stands. The row carries the engine's reading and where it
    came from (`state_source`): a project whose state.json is lost is listed where its events
    say it stands, and "Not started" is said only of one the server says has not started. */
export function rowState(row) {
  const running = row.worker && row.worker !== 'idle';
  const waiting = waitingWords(row);
  const and = (text) => (waiting ? text + ' · ' + waiting : text);
  /* Nothing on file says where it stands: said as that, with the server's sentence on the row.
     What it needs is in that sentence (a phase to be named, or a file to be put right): the
     row does not guess which. */
  if (!running && row.state_source === 'unknown') return ['pause', 'State unknown', and('needs you')];
  /* the project's files name a worker this server cannot verify: said as that, not as "running" */
  if (running && row.worker_verified === false) return ['pause', 'Worker not verified', and('named in its files; see the project')];
  if (running || row.status === 'running') return ['live', 'Round ' + (row.round || 1) + ' · ' + phaseName(row.phase), and('running now')];
  if (row.status === 'failed') return ['neg', 'Round ' + (row.round || 1) + ' · ' + phaseName(row.next_phase || row.phase) + ' failed', and('needs you')];
  if (row.status === 'stopped') return [row.error ? 'neg' : 'pause', 'Round ' + (row.round || 1) + ' · stopped', and(row.error ? 'the agents stopped unexpectedly' : 'stopped by you')];
  if (row.status === 'checkpoint') return ['pause', 'Round ' + (row.round || 1) + ' · checkpoint', waiting || 'ready to continue'];
  if (row.state_source ? row.state_source === 'not_started' : !row.round) return ['idle', 'Not started', waiting || 'ready'];
  /* at rest in a round, with no checkpoint on record */
  return ['pause', 'Round ' + (row.round || 1) + (row.phase ? ' · after ' + phaseName(row.phase) : ''), waiting || 'ready to continue'];
}

export const libraryView = {
  render(root) {
    loadLibrary(false);
    const lib = app.ui.library;
    const head = html`<header class="view-head lib-head"><div><h1>Projects</h1><p>Each project is a folder of plain files${app.ui.health && app.ui.health.workspace ? html` in <code>${app.ui.health.workspace}</code>` : ''}. Open one to see its map.</p></div>
      <a class="btn btn-primary" href="#/start">${icon('plus')}New project</a></header>`;
    const unreadable = (lib.unreadable || []);
    /* a folder that holds a project which cannot be opened is listed all the same, with why */
    const broken = unreadable.length ? html`<section class="lib-unreadable" aria-label="Projects that cannot be opened"><h2>${unreadable.length === 1 ? 'A project that cannot be opened' : unreadable.length + ' projects that cannot be opened'}</h2>
      ${unreadable.map((x) => html`<div class="lib-bad" data-dir="${x.dir || ''}">${icon('warn')}<div><code class="lib-dir">${x.dir || '?'}/</code><p>${plain(x.message || 'It cannot be opened.')}</p></div></div>`)}
      <p class="quiet">Nothing in these folders was changed. Put right what the message names, and the project is listed again.</p></section>` : '';
    let body;
    if (!lib.projects && lib.error) body = html`<div class="note note-neg" role="alert">${icon('warn')}<span>The projects could not be listed: ${lib.error.message} <button class="linkish" data-act="reload-library">Try again</button></span></div>`;
    else if (!lib.projects) body = html`<div class="skeleton" aria-busy="true" aria-label="Loading the projects"><i></i><i></i><i class="short"></i></div>`;
    else if (!lib.projects.length) body = unreadable.length ? broken : html`<div class="blank lib-blank"><h2>No project yet</h2><p>A project starts from an idea: a few sentences, and whatever notes, data or papers you already have.</p><p><a class="btn btn-primary btn-l" href="#/start">Start the first one ${icon('right')}</a></p></div>`;
    else {
      body = html`<div class="lib-list">${lib.projects.map((row) => {
        const counts = row.counts || null, st = rowState(row);
        const by = counts ? counts.by_kind || {} : null, pending = counts ? counts.pending || {} : null;
        /* what the last phase did; how much waits now is said live in the next column */
        const did = row.checkpoint && (row.checkpoint.did || row.checkpoint.summary) ? checkpointDid(row.checkpoint) : '';
        const fresh = row.state_source ? row.state_source === 'not_started' : row.status === 'idle' && !row.round;
        const say = row.state_source === 'unknown' ? plain(row.state_problem || 'Nothing on file says where this project stands.')
          : row.status === 'failed' || (row.status === 'stopped' && row.error) ? plain(row.error) : did ? 'After ' + phaseName(row.checkpoint.phase) + ': ' + did : fresh ? 'The idea is in place; research has not started.' : '';
        /* state.json did not say, and the engine read the events instead: the row stands where they say, and says so */
        const rebuilt = row.state_source === 'events' && row.state_problem ? plain(row.state_problem) : '';
        /* two projects of one title are told apart by their folders */
        const twin = row.title && lib.projects.filter((x) => x.title === row.title).length > 1;
        const waits = pending && pending.total ? plural(pending.pages || 0, 'proposed page') + (pending.proposals ? ' and ' + plural(pending.proposals, 'suggested edit') : '') + ' wait for a decision' : '';
        return html`<a class="lib-row" href="#/p/${encodeURIComponent(row.id)}/map"><div class="lib-main"><h2>${row.title || row.dir}</h2>${twin && row.dir ? html`<code class="lib-dir">${row.dir}/</code>` : ''}<p>${say}</p></div>
          <div class="lib-map">${stripMapSmall(by, pending ? pending.by_kind : null)}<span class="lib-counts">${by ? countsLine(by) : ' '}</span></div>
          <div class="lib-st st-${st[0]}"><span class="lib-st1">${st[0] === 'live' ? html`<span class="live-dot"></span>` : ''}${st[1]}</span><span class="lib-st2"${waits ? html` data-tip="${waits}"` : ''}>${st[2]}</span>${rebuilt ? html`<span class="lib-st3" data-state-problem data-tip="${rebuilt}">${icon('warn')}read from its events</span>` : ''}</div>
          <div class="lib-num"${row.demonstration ? html` data-demonstration data-tip="${DEMO_COST}"` : ''}><span>${money((row.spent || {}).model_usd || 0)}</span>${row.demonstration ? html`<span class="lib-demo">in its story</span>` : ''}${row.autonomy === 'autopilot' ? html`<span class="lib-auto" data-autonomy="autopilot" data-tip="Autopilot is on: pages, edits, runs and checkpoints go on without you, up to the caps of one stretch">autopilot</span>` : ''}<span class="quiet">${ago(row.updated)}</span></div></a>`;
      })}</div>${broken}`;
    }
    setHTML(root, html`<div class="library"><div class="lib-wrap">${head}${body}</div></div>`);
  },
};

/* ---------- setup ---------- */
export function loadDoctor(force) {
  const doc = app.ui.doctor || (app.ui.doctor = { data: null, error: null, loading: false, at: 0 });
  if (doc.loading || (!force && doc.data)) return;
  doc.loading = true;
  api.doctor().then((data) => { doc.data = data; doc.error = null; doc.at = Date.now(); }, (error) => { doc.error = error; })
    .finally(() => { doc.loading = false; if (app.route.name === 'setup') app.render('setup', true); });
}
const MARKS = { ok: ['check', 'Usable'], warn: ['warn', 'Works, limited'], missing: ['x', 'Not set up'], optional: ['minus', 'Optional, not set up'] };
const item = (status, name, note, detail, fix, more) => html`<div class="doc-i s-${status}"><span class="doc-m">${icon(MARKS[status][0])}</span><div class="doc-b"><h3>${name}${note ? html`<span class="quiet">${note}</span>` : ''}</h3>
  ${String(detail || '').split('\n').filter((l) => l.trim()).map((l) => html`<p>${l}</p>`)}
  ${fix ? html`<div class="doc-fix"><code>${fix}</code><button class="ibtn" data-act="copy" data-text="${fix}" data-tip="Copy" aria-label="Copy the command">${icon('copy')}</button></div>` : ''}${more || ''}</div><span class="doc-s">${MARKS[status][1]}</span></div>`;

export const setupView = {
  render(root) {
    loadDoctor(false);
    const doc = app.ui.doctor, d = doc.data;
    const head = html`<header class="view-head lib-head"><div><h1>Setup check</h1><p>What this machine can use. Keys are read from the environment and never shown, stored or sent anywhere but their own service.</p></div>
      <button class="btn" data-act="recheck"${doc.loading ? raw(' disabled') : ''}>${doc.loading ? 'Checking…' : 'Check again'}</button></header>`;
    if (!d) {
      setHTML(root, html`<div class="setup"><div class="setup-wrap">${head}${doc.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The check could not be run: ${doc.error.message}</span></div>` : html`<div class="skeleton" aria-busy="true" aria-label="Checking"><i></i><i></i><i class="short"></i></div>`}</div></div>`);
      return;
    }
    const prov = d.provider || {}, local = (d.compute || {}).local || {}, modal = (d.compute || {}).modal || {};
    /* Where the default model cannot be used, the server says in a line what works all the
       same (`without_credentials`); the app adds what only it knows: the server read its
       environment when it started, so a credential set now is seen after a restart. */
    const without = d.ok === false && typeof d.without_credentials === 'string' && d.without_credentials ? html`<p class="doc-note" data-without-credentials>${codeSpans(d.without_credentials)}</p><p class="doc-note">After you set a credential, stop the app and start it again: credentials are read when it starts.</p>` : '';
    const models = html`<section class="doc-g"><h2>Models</h2>
      ${item(prov.usable ? ((prov.warnings || []).length ? 'warn' : 'ok') : 'missing', cap(prov.provider || 'Provider') + ' · ' + (d.model || prov.model || ''), prov.credential_source ? 'credentials from ' + prov.credential_source : '', prov.usable ? ((prov.warnings || []).join('\n') || 'The research agents can use this model.') : prov.error || 'No credentials were found.', prov.usable ? '' : 'export ANTHROPIC_API_KEY=…', without)}
      ${(d.other_providers || []).map((o) => item(o.credential_source ? 'ok' : 'optional', cap(o.provider) === 'Openai' ? 'OpenAI' : cap(o.provider), o.sdk_installed ? 'package installed' : 'package not installed', o.credential_source ? 'Credentials from ' + o.credential_source + '. Research agents need a Claude model; other providers serve the paper pipeline only.' : 'No credentials. Only needed if you write papers with this provider.', ''))}</section>`;
    const compute = html`<section class="doc-g"><h2>Where experiments run</h2>
      ${item(local.available ? ((local.jobs || {}).network_isolation === false ? 'warn' : 'ok') : 'missing', 'This machine', local.python ? local.python : '', ((local.jobs || {}).summary || '') + ((local.problems || []).length ? '\n' + local.problems.join('\n') : '') + (local.error ? '\n' + local.error : ''), '')}
      ${item(modal.available ? 'ok' : 'missing', 'Modal', modal.modal_version ? 'modal ' + modal.modal_version : 'package not installed', modal.available ? (modal.jobs || {}).summary || 'Sandboxes can be started.' : ((modal.problems || []).join('\n') || modal.error || 'Not available.'), modal.available ? '' : 'modal setup')}
      ${item(local.pdflatex ? 'ok' : 'optional', 'LaTeX', local.pdflatex || '', local.pdflatex ? 'A paper can be compiled on this machine.' : 'pdflatex was not found. Only needed for the optional paper step, and Modal can compile instead.', '')}</section>`;
    const lit = html`<section class="doc-g"><h2>Literature</h2>
      ${item(d.semantic_scholar_api_key ? 'ok' : 'warn', 'Semantic Scholar', d.semantic_scholar_api_key ? 'API key found' : 'no API key', d.semantic_scholar_api_key ? 'Papers are verified at the rate your key allows.' : 'Papers are still verified, at the slow rate of the public interface. A key makes the research phase faster.', d.semantic_scholar_api_key ? '' : 'export SEMANTIC_SCHOLAR_API_KEY=…')}</section>`;
    /* What agents with a shell (`settings.agents.runtime: sdk`) need on this machine: the
       package and its command line, and one of the two credentials a project can choose. */
    const ag = d.agents || null;
    const agentsSec = !ag ? '' : html`<section class="doc-g" data-setup="agents"><h2>Agents with a shell <span class="quiet">for projects whose agents are Claude Agent SDK agents</span></h2>
      ${ag.error ? item('optional', 'Claude Agent SDK', '', 'This could not be checked: ' + ag.error, '')
        : html`${item(ag.package && ag.command_line ? 'ok' : 'optional', 'Claude Agent SDK', ag.package ? 'package installed' : 'package not installed', ag.package && ag.command_line ? 'Agents with a shell can be started. Which model they use, and whose account, is set per project: on its overview under "How the agents work".' : ag.package ? 'The package is installed, but the Claude Code command line the agents run in was not found.' : 'Only needed for projects whose agents are Claude agents with a shell.', ag.package ? '' : "uv sync --extra agents")}
          ${item(ag.login ? 'ok' : 'optional', 'Your Claude Code login', ag.login ? 'found' : ag.login === false ? 'not found' : 'not kept in a file here', ag.login ? 'Projects set to the credential "login" work on your own Claude account; fast mode works there.' : ag.login === false ? 'Projects set to the credential "login" cannot start their agents until you have signed in.' : 'On this system the login is not kept in a file; the first agent\'s start shows whether you are signed in.', ag.login === false ? 'claude' : '')}
          ${item(ag.federation ? 'ok' : 'optional', 'Workload Identity Federation', ag.federation ? 'settings found' : 'not set up', ag.federation ? 'Projects set to the credential "federation" work on the worker\'s account.' : (ag.federation_missing || []).length ? 'Not set for this server: ' + ag.federation_missing.join(', ') + '. Only needed for projects set to the credential "federation".' : 'The file that ANTHROPIC_IDENTITY_TOKEN_FILE names is not there.', '')}`}</section>`;
    const pk = d.packages || {};
    setHTML(root, html`<div class="setup"><div class="setup-wrap">${head}${models}${agentsSec}${compute}${lit}
      <p class="quiet setup-foot">Checked ${ago(new Date(doc.at).toISOString())} · PaperOrchestra ${d.version || ''} · Python ${d.python || ''}${app.ui.health && app.ui.health.workspace ? html` · workspace <code>${app.ui.health.workspace}</code>` : ''} · server at ${location.host}
        ${Object.keys(pk).length ? html`<br>Packages: ${Object.keys(pk).filter((k) => pk[k]).map((k) => k + ' ' + pk[k]).join(', ')}` : ''}</p></div></div>`);
  },
};
const cap = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '');
