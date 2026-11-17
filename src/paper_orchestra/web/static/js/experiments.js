/* Experiments: the table of all of them, and one experiment with its approval, plan, code,
   runs, live log, metrics and artifacts. Nothing here runs anything by itself: every run
   starts from a button the person presses. */

import { app, link } from './core.js';
import { api, connection, Unreachable } from './api.js';
import { $$, ago, cap, clock, dateTime, duration, html, join, minutes, money, parseStamp, plain, plural, raw, setHTML, tf, whyNot } from './util.js';
import { TONE, agentName, glyph, icon, lapsed, pageGlyph, shownState, successWords } from './vocab.js';
import { afterRender, approvalChoice, backendName, label, modelAccessNote, networkAsk, pageName, pageTip, planProblemsHTML, proposedAt, reviewBar, reviewTag, stateTag, unacceptForm, unacceptLink, who } from './page.js';

const GROUPS = [
  /* (also one that reads approved and is no longer what was approved: it will not run until it is approved again) */
  ['Needs your approval', (x) => x.review !== 'rejected' && (x.review === 'proposed' || x.state === 'planned' || !x.state || lapsed(x))],
  /* the page says "running" and nothing is running it: its worker is gone, and it waits for the person */
  ['Interrupted', (x) => x.review === 'accepted' && x.state === 'running' && !!x.interrupted],
  ['Approved, queued and running', (x) => x.review === 'accepted' && !lapsed(x) && (x.state === 'approved' || x.state === 'queued' || (x.state === 'running' && !x.interrupted))],
  ['Failed', (x) => x.review === 'accepted' && x.state === 'failed'],
  ['Finished', (x) => x.review === 'accepted' && (x.state === 'succeeded' || x.state === 'cancelled')],
  ['Rejected', (x) => x.review === 'rejected'],
];
/** The heading an experiment is listed under, in the table and in the rail. */
export const groupName = (x) => (GROUPS.find((g) => g[1](x)) || [''])[0];
/** Where an experiment stands, in the few words a narrow list has room for: its state as
    it is shown, and with a state that reads as "no result" the earlier run that succeeded. */
export function standingWords(x) {
  if (x.review === 'proposed') return 'proposed · ' + agentName(x.author);
  if (x.review === 'rejected') return 'rejected';
  const state = shownState(x) || 'planned', won = successWords(x);
  return won ? state + '; ' + won : state;
}
const IMAGE = /\.(png|jpe?g|gif|webp|svg)$/i;
/** Whether an agent carried a run out: the run's own word (`carried_out_by` of its record),
    which stays true when the project's runtime is changed afterwards. */
const byAgent = (run) => !!run && run.carried_out_by === 'agent';
/** Whether a run approved now for `backend` would be carried out by an agent: the project's
    agents have a shell, and an agent works on this machine (a run on Modal runs there as ever). */
const agentWould = (store, backend) => ((store.project.settings || {}).agents || {}).runtime === 'sdk' && backend === 'local';

const whereOf = (x) => { const m = x.meta || {}; return m.backend ? backendName(m.backend, m.gpu) : m.mode === 'gpu' ? 'not chosen · needs a GPU' : m.mode === 'repo' ? 'not chosen · your repository' : 'not chosen'; };
const metricLine = (run) => { const m = (run && (run.metrics_text || run.metrics)) || {}; return Object.keys(m).slice(0, 2).map((k) => k + ' ' + m[k]).join(' · '); };
const sinceClock = (stamp) => html`<span data-since="${stamp}">${clock((Date.now() - (parseStamp(stamp) || Date.now())) / 1000)}</span>`;

function subline(store, x) {
  const run = x.last_run || {};
  if (x.review === 'rejected') return html`rejected${(x.meta || {}).decision_reason ? ': ' + x.meta.decision_reason : ''}`;
  if (x.review === 'proposed') return html`waiting for you · proposed ${ago(proposedAt(x))}`;
  if (x.state === 'planned' || !x.state) return (x.meta || {}).state_reason ? html`<span class="t-prop">back to planned · ${plain(x.meta.state_reason)}</span>` : html`accepted · runs only after you approve it`;
  if (lapsed(x)) return html`<span class="t-prop">will not run · ${whyNot(x.will_not_run)} · approve it again</span>`;
  if (x.state === 'approved') return html`approved · runs first when you continue, or now`;
  if (x.state === 'queued') return html`queued · runs first when you continue, or now`;
  if (x.state === 'running' && x.interrupted) return html`<span class="t-prop">interrupted · the worker that ran it is gone · run it again, or cancel it</span>`;
  if (x.state === 'running') return html`<span class="ex-tail" data-tail="${x.id}">${store.tails[x.id] || 'run ' + (run.number || '') + ' started ' + ago(run.started)}</span>`;
  if (x.state === 'succeeded') return html`${metricLine(run) || 'succeeded'}`;
  /* (an earlier run that succeeded is said with a state that would read as "no result") */
  const won = successWords(x);
  if (x.state === 'failed') return html`<span class="t-neg">${(x.meta && x.meta.state_reason) || run.error || 'failed'}</span>${won ? ' · ' + won : ''}`;
  if (x.state === 'cancelled') return html`${run.finished ? 'cancelled ' + ago(run.finished) : 'cancelled before it ran'}${won ? '; ' + won : ''}`;
  return '';
}
function stateCell(x) {
  if (x.review === 'rejected') return html`<span class="tag tag-rejected">rejected</span>`;
  if (x.review === 'proposed') return html`<span class="tag tag-proposed">proposed</span>`;
  /* (an interrupted run has its own mark and word: the page says "running", and nothing runs it) */
  const state = shownState(x) || 'planned', won = successWords(x);
  return html`<span class="st-word t-${TONE[state] || 'open'}">${glyph('experiment', 'accepted', state, 14)}${state}</span>${won ? html`<span class="st-also">${won}</span>` : ''}`;
}
function timeCell(x) {
  const run = x.last_run || {}, m = x.meta || {};
  if (x.state === 'running' && run.started && !x.interrupted) return sinceClock(run.started);
  if (run.seconds != null && (x.state === 'succeeded' || x.state === 'failed' || x.state === 'cancelled')) return duration(run.seconds);
  return m.est_minutes != null ? 'est. ' + minutes(m.est_minutes) : '';
}
function costCell(x) {
  const run = x.last_run || {}, m = x.meta || {};
  if (run.cost_usd != null) return money(run.cost_usd);
  if ((x.runs && run.backend === 'local') || m.backend === 'local') return 'no cost';
  if (m.est_usd != null && m.est_usd !== '') return 'est. ' + money(m.est_usd);
  return '';
}
const testsOf = (store, x) => ((x.relations || {}).tests || []).map((ref) => store.page(ref)).filter(Boolean);

function budgetLine(store) {
  const limits = (store.project.settings || {}).limits || {}, spent = (store.project.state || {}).spent || {};
  return html`<span class="ex-budget">Experiment budget <b>${money(spent.experiment_usd || 0)}</b> of ${money(limits.experiment_usd)} · limit <b>${minutes((limits.experiment_seconds || 0) / 60)}</b> per run</span>`;
}

/* ---------- the log of a run: fetched in pieces, followed while the run is live ---------- */
/* What stands in a log where a job printed a credential (the server and the job backends put
   it there). It is a mark, not an error, and is shown as the plain text it is. */
const MASK = '[redacted]';
const logs = new Map();
let following = null, pollTimer = 0;
const MAX_LOG = 400000;
const logKey = (store, eid, number) => store.pid + '/' + eid + '/' + number;
async function fetchLog(store, eid, number, entry) {
  if (entry.loading) return;
  entry.loading = true;
  try {
    const piece = await api.runLog(store.pid, eid, number, entry.next == null ? -65536 : entry.next);
    const text = piece.text || '';
    /* a log that is a link out of the project, or not a file, is not read: the server says so */
    entry.problem = piece.problem || '';
    if (piece.reset || entry.next == null) entry.text = text; else entry.text += text;
    if (entry.text.length > MAX_LOG) entry.text = entry.text.slice(-MAX_LOG);
    entry.cut = entry.next == null && piece.offset > 0 ? piece.offset : entry.cut || 0;
    entry.next = piece.next;
    const first = !entry.loaded;
    entry.loaded = true;
    entry.error = null;
    $$('[data-log="' + CSS.escape(eid + '/' + number) + '"]').forEach((el) => {
      const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      if (piece.reset || el.dataset.empty) { el.textContent = entry.problem ? plain(entry.problem) : entry.text; delete el.dataset.empty; } else if (text) el.append(text);
      if (stick) el.scrollTop = el.scrollHeight;
      const mask = el.closest('.log') && el.closest('.log').querySelector('.log-mask');
      if (mask) mask.hidden = entry.text.indexOf(MASK) < 0;
    });
    /* the log of a run that has ended is also where its failure quotes from: drawn again,
       after the text is in place (a render before it would be given the same text twice) */
    if (first && !entry.live) app.render('log', true);
    const lines = entry.text.trimEnd().split('\n');
    const tail = lines[lines.length - 1];
    if (tail) { store.tails[eid] = tail; $$('[data-tail="' + CSS.escape(eid) + '"]').forEach((el) => { el.textContent = tail; }); }
  } catch (error) { entry.error = error; }
  entry.loading = false;
}
function logOf(store, eid, run) {
  const key = logKey(store, eid, run.number);
  let entry = logs.get(key);
  if (!entry) { entry = { text: '', next: null, loaded: false, loading: false, live: false }; logs.set(key, entry); fetchLog(store, eid, run.number, entry); }
  const live = run.status === 'running';
  if (entry.live && !live) fetchLog(store, eid, run.number, entry);   // the last lines, once it has ended
  else if (entry.error instanceof Unreachable && connection.ok && !entry.loading) {
    /* the server was gone when the log was asked for, and is back */
    entry.error = null;
    fetchLog(store, eid, run.number, entry);
  }
  entry.live = live;
  if (live) { following = { store, eid, number: run.number, entry }; if (!pollTimer) pollTimer = setInterval(poll, 1200); }
  return entry;
}
function poll() {
  const f = following;
  if (!f || f.store.closed || app.store !== f.store || !document.querySelector('[data-log="' + CSS.escape(f.eid + '/' + f.number) + '"]')) { clearInterval(pollTimer); pollTimer = 0; following = null; return; }
  fetchLog(f.store, f.eid, f.number, f.entry);
}
/** A `run.log` event arrived: the open log reads on at once, without a render. */
app.hooks.runLog = (event) => {
  const data = (event && event.data) || {};
  if (data.experiment && data.lines && data.lines.length) $$('[data-tail="' + CSS.escape(data.experiment) + '"]').forEach((el) => { el.textContent = data.lines[data.lines.length - 1]; });
  if (following && following.eid === data.experiment) fetchLog(following.store, following.eid, following.number, following.entry);
};

function logHTML(store, eid, run) {
  const entry = logOf(store, eid, run), live = run.status === 'running', open = app.ui.logOpen;
  return html`<div class="log${open ? ' open' : ''}${live ? ' live' : ''}"><div class="log-bar"><span>${live ? 'Live log' : 'log.txt'} · run ${run.number}</span>${live ? html`<span class="log-follow">following</span>` : ''}
    <span class="log-mask quiet" data-tip="Where the job printed a credential, the log shows [redacted] in its place. The value is never sent to this page."${entry.text.indexOf(MASK) < 0 ? raw(' hidden') : ''}>credentials are masked as [redacted]</span>
    <a class="linkish log-open" href="${api.fileUrl(store.pid, 'experiments/' + eid + '/runs/' + run.number + '/log.txt')}" target="_blank" rel="noopener">Open the whole log</a>
    <button class="btn btn-quiet btn-s" data-act="toggle-log" aria-expanded="${tf(open)}">${open ? 'Collapse' : 'Expand'} <kbd>L</kbd></button></div>
    <pre class="log-body" tabindex="0" aria-label="Log of run ${run.number}" data-log="${eid}/${run.number}"${entry.loaded ? '' : raw(' data-empty="1"')} data-follow>${entry.loaded ? (entry.problem ? plain(entry.problem) : entry.text || (live ? '' : 'The run wrote nothing to its log.')) : entry.error ? 'The log could not be read: ' + entry.error.message : 'Reading the log…'}</pre></div>`;
}

/* ---------- pieces of one experiment ---------- */
const codeHTML = (text) => html`<pre class="code code-n" tabindex="0"><code>${String(text).replace(/\n$/, '').split('\n').map((l) => html`<span class="ln">${l || ' '}</span>\n`)}</code></pre>`;
/** A unified diff as coloured lines; each line also starts with its sign. */
export const diffText = (text) => html`<pre class="diff">${String(text || '').replace(/\n$/, '').split('\n').map((line) => html`<span class="diff-line ${line.startsWith('+++') || line.startsWith('---') ? 'diff-file' : line.startsWith('@@') ? 'diff-hunk' : line.startsWith('+') ? 'diff-add' : line.startsWith('-') ? 'diff-del' : 'diff-context'}">${line}</span>\n`)}</pre>`;

function approveHTML(store, page, data) {
  const eid = page.id, plan = data.plan || {}, options = data.options || [];
  const limits = (store.project.settings || {}).limits || {};
  const chosen = approvalChoice(eid, data);
  const runnable = options.filter((o) => o.can_run);
  /* the longest one run may take is the project's limit; a plan that names no time gets that */
  const capSeconds = limits.experiment_seconds || plan.timeout_seconds || 60;
  const capMin = Math.max(1, Math.floor(capSeconds / 60));
  const planMin = Math.max(1, Math.round((plan.timeout_seconds || capSeconds) / 60));
  const limitMin = chosen.minutes != null && chosen.minutes !== '' ? chosen.minutes : Math.min(planMin, capMin);
  /* network access is never taken from the plan: off until the person turns it on, and a
     plan that asks for it is answered explicitly (`networkAsk`) */
  const network = chosen.network === true;
  const gpu = chosen.gpu != null ? chosen.gpu : plan.gpu || '';
  const pick = options.find((o) => o.backend === chosen.backend);
  const proposed = page.review === 'proposed';
  const row = (o) => {
    const on = o.backend === chosen.backend;
    return html`<label class="backend${on ? ' on' : ''}${o.can_run ? '' : ' off'}"><input type="radio" name="backend-${eid}" value="${o.backend}" data-input="backend" data-eid="${eid}"${on ? raw(' checked') : ''}${o.can_run ? '' : raw(' disabled')}>
      <span class="be-name">${backendName(o.backend, o.backend === 'modal' ? gpu : '')}${plan.backend === o.backend && o.can_run ? html`<span class="flag">suggested</span>` : ''}</span>
      <span class="be-desc">${o.can_run ? (agentWould(store, o.backend) ? 'An agent with a shell carries it out here. ' : '') + ((o.notes || []).join(' ') || (o.backend === 'local' ? (agentWould(store, 'local') ? '' : 'A subprocess on this machine.') : 'A sandbox made for this job.')) : 'Cannot run this plan.'}</span>
      <span class="be-num">${o.est_minutes != null ? 'about ' + minutes(o.est_minutes) : 'not estimated'}</span><span class="be-num">${o.est_usd ? 'about ' + money(o.est_usd) : o.backend === 'local' ? 'no cost' : o.est_usd === 0 ? '$0.00' : 'not estimated'}</span>
      ${o.can_run ? '' : html`<span class="be-flag">${icon('warn')}<span>${(o.reasons || []).join(' ')}</span></span>`}</label>`;
  };
  const sum = pick ? backendName(pick.backend, pick.backend === 'modal' ? gpu : '') + ', for at most ' + minutes(+limitMin || planMin) + (pick.max_usd ? ' and ' + money(pick.max_usd * ((+limitMin || planMin) * 60) / Math.max(1, pick.timeout_seconds || planMin * 60)) : '') : '';
  return html`<section class="approve" id="approve" data-eid="${eid}"><h2>Where should it run?</h2>
    ${proposed ? html`<p class="approve-say"><b>Proposed</b> by ${who(page.author)}${page.round ? ' in round ' + page.round : ''}, ${ago(proposedAt(page))}. It is not part of the wiki until you accept it, and it will not run until you approve it.</p>` : ''}
    ${!proposed && page.state === 'planned' && (page.meta || {}).state_reason ? html`<p class="note note-warn approve-back" role="status">${icon('warn')}<span><b>It is planned again and waits for your approval.</b> ${plain(page.meta.state_reason)}</span></p>` : ''}
    ${data.plan_error ? html`<p class="note note-warn">${icon('warn')}<span>The plan cannot be read as it is: ${data.plan_error}</span></p>` : ''}
    <div class="backends" role="radiogroup" aria-label="Backend">${options.map(row)}</div>
    ${data.options_error ? html`<p class="note note-warn">${icon('warn')}<span>${data.options_error}</span></p>` : ''}
    <div class="approve-fields">
      <label><span>Time limit</span><input type="number" id="ap-min-${eid}" min="1" max="${capMin}" step="1" value="${limitMin}" data-input="ap-minutes" data-eid="${eid}" inputmode="numeric"><span class="quiet">${agentWould(store, chosen.backend) ? 'minutes for all of the agent\'s work (reading, running and repairing the code, not one execution of it), then it is stopped' : 'minutes, then it is stopped'} (at most ${capMin} in this project)</span></label>
      ${chosen.backend === 'modal' || plan.mode === 'gpu' ? html`<label><span>GPU</span><input id="ap-gpu-${eid}" value="${gpu}" placeholder="none (CPU)" data-input="ap-gpu" data-eid="${eid}" size="8" autocomplete="off"><span class="quiet">for example T4, L4, A10G</span></label>` : ''}
      ${plan.network ? '' : html`<label class="tick"><input type="checkbox" id="ap-net-${eid}" data-input="ap-network" data-eid="${eid}"${network ? raw(' checked') : ''}> Allow network access<span class="quiet">${chosen.backend === 'local' ? 'off unless you tick this; this machine cannot block it either way' : 'off unless you tick this'}</span></label>`}
    </div>
    ${plan.network ? '' : modelAccessNote(!!data.model_access && chosen.backend === 'local')}
    ${networkAsk(eid, plan, !!data.model_access && chosen.backend === 'local')}
    <div class="approve-opts"><span><b>Mode</b> ${plan.mode || 'script'}</span>${(plan.packages || []).length ? html`<span><b>Packages</b> ${plan.packages.join(', ')}</span>` : ''}${plan.repo ? html`<span><b>Repository</b> ${plan.repo}</span>` : ''}
      <span><b>Left for experiments</b> ${money(Math.max(0, (limits.experiment_usd || 0) - (((store.project.state || {}).spent || {}).experiment_usd || 0)))}</span></div>
    ${app.ui.rejecting === page.path ? html`<form class="reject-form" data-form="reject" data-ref="${page.path}"><label class="sr" for="why-x-${eid}">Reason</label><input id="why-x-${eid}" name="reason" data-input="reason" data-ref="${page.path}" value="${app.ui.reasons[page.path] || ''}" placeholder="Why? The agents read this reason." autocomplete="off"><button class="btn btn-neg" type="submit">Reject</button><button class="btn btn-quiet" type="button" data-act="reject-cancel">Cancel</button></form>`
      : html`<div class="approve-go">${runnable.length ? (pick && agentWould(store, pick.backend)
        ? html`<p data-approves="agent">Approving lets an agent carry out this experiment once on ${sum}: it works in a shell in the run's own folder, starts from the code below (or writes it from the plan), may change and rerun it, and must leave <code>metrics.json</code>. The time limit is for all of its work, and its shell reaches the web whether or not you allow network access. Nothing else is approved.</p>`
        : html`<p>Approving lets the runner execute <code>${(plan.command || ['python', plan.entrypoint || 'main.py']).join(' ')}</code> once on ${sum}. Nothing else is approved.</p>`) : html`<p><b>No backend can run this plan as it is.</b> You can still accept the plan and change it, or the code, before approving a run.</p>`}
        <div class="review-actions">${proposed
          ? html`<button class="btn btn-primary" data-act="accept-run" data-ref="${page.path}"${runnable.length ? '' : raw(' disabled')}>${icon('play')}Accept and run</button>
              <button class="btn" data-act="accept" data-ref="${page.path}">Accept, keep planned <kbd>A</kbd></button>
              <button class="btn" data-act="edit" data-ref="${page.path}">Edit <kbd>E</kbd></button>
              <button class="btn btn-neg-quiet" data-act="reject-ask" data-ref="${page.path}">Reject… <kbd>R</kbd></button>`
          : html`<button class="btn btn-primary" data-act="approve" data-eid="${eid}" data-start="1"${runnable.length ? '' : raw(' disabled')}>${icon('play')}Approve and run now</button>
              <button class="btn" data-act="approve" data-eid="${eid}"${runnable.length ? '' : raw(' disabled')} data-tip="It runs first, and is analysed, when you next continue the project">Approve, run when I continue</button>
              <button class="btn" data-act="edit" data-ref="${page.path}">Edit the plan <kbd>E</kbd></button>
              ${app.ui.reapprove === eid ? html`<button class="btn btn-quiet" data-act="reapprove" data-eid="">Cancel</button>` : ''}`}</div></div>`}
  </section>`;
}

/** The button that takes an approval back or cancels a run. Which of the two a press does
    is the server's word (`cancel_does` of the experiment: `withdraw` for one that is approved
    and has not started, `cancel` for one that waits in the queue or runs), and the button
    says it; the press tells the server what the person was told (`data-does`), so that it
    does that or nothing. Where the server has not said, the button promises neither. */
function cancelButton(eid, does, opt) {
  const known = does === 'withdraw' || does === 'cancel';
  const label = does === 'withdraw' ? 'Take the approval back' : does === 'cancel' ? opt.label || 'Cancel the run' : opt.label || 'Do not run it';
  const tip = does === 'withdraw' ? 'It becomes a plan again: nothing is cancelled and nothing is recorded as a run. Approve it again when you want it run.'
    : does === 'cancel' ? opt.tip || 'Its run is cancelled and recorded as that; it can be approved again'
      : opt.tip || 'Its approval is taken back, or its run is cancelled: the answer says which';
  const busy = !!app.ui.cancelling[eid];
  return html`<button class="btn ${opt.cls || 'btn-quiet'}" data-act="cancel-run" data-eid="${eid}"${known ? html` data-does="${does}"` : ''} data-tip="${tip}"${busy ? raw(' disabled') : ''}>${opt.stop ? icon('stop') : ''}${label}</button>`;
}

export function failureKind(run) {
  const k = run.error_kind;
  if (run.status === 'cancelled') return 'cancelled';
  if (run.timed_out || k === 'timeout') return 'timeout';
  if (k === 'budget') return 'budget';
  if (['backend', 'setup', 'worker', 'internal'].indexOf(k) >= 0) return 'backend';
  if (['exit', 'metrics', 'code', 'patch', 'command', 'declared'].indexOf(k) >= 0) return 'program';
  return run.error && run.exit_code == null ? 'backend' : 'program';
}
const choice = (act, attrs, title, sub, primary) => html`<button class="choice${primary ? ' choice-primary' : ''}" data-act="${act}"${raw(Object.entries(attrs).map(([k, v]) => ' data-' + k + '="' + String(v).replace(/[^\w.-]/g, '') + '"').join(''))}><b>${title}</b><span>${sub}</span></button>`;

function failureHTML(store, page, data, run) {
  const eid = page.id, kind = failureKind(run), plan = data.plan || {};
  const limits = (store.project.settings || {}).limits || {};
  const others = (data.options || []).filter((o) => o.can_run && o.backend !== run.backend);
  const same = (data.options || []).find((o) => o.backend === run.backend);
  const cost = (o) => (o.est_minutes != null ? 'about ' + minutes(o.est_minutes) : 'time not estimated') + ' · ' + (o.est_usd ? money(o.est_usd) : o.backend === 'local' ? 'no cost' : 'cost not estimated');
  const elsewhere = others.map((o, i) => choice('rerun', { eid, backend: o.backend }, 'Re-run on ' + backendName(o.backend, plan.gpu), cost(o), i === 0 && kind === 'timeout'));
  const again = same && same.can_run ? choice('rerun', { eid, backend: run.backend }, 'Run it again', backendName(run.backend, plan.gpu) + ' · unchanged', false) : '';
  const edit = choice('edit-code', { eid }, 'Edit the code', 'A changed file is parsed before it can run.', false);
  const entry = logOf(store, eid, run);
  const tail = entry.loaded ? entry.text.trimEnd().split('\n').slice(-12).join('\n') : '';
  let title, say, actions;
  if (kind === 'timeout') {
    const usedSeconds = (run.plan && run.plan.timeout_seconds) || plan.timeout_seconds || 0;
    const usedMin = Math.round(usedSeconds / 60);
    const capMin = Math.floor((limits.experiment_seconds || 0) / 60);
    const more = Math.min(capMin, Math.max(usedMin * 2, usedMin + 1));
    title = 'Stopped at the time limit';
    say = byAgent(run) ? html`<p>This is a limit, not a bug: the agent was stopped after ${usedSeconds < 90 ? duration(usedSeconds) : minutes(usedMin)}. The limit is for all of its work (writing, running and repairing the code), not for one execution.</p>`
      : html`<p>This is a limit, not a bug: the run was stopped after ${usedSeconds < 90 ? duration(usedSeconds) : minutes(usedMin)}. The runner did not attempt a repair because no code is wrong.</p>`;
    actions = html`${elsewhere}${more > usedMin && same && same.can_run ? choice('rerun', { eid, backend: run.backend, minutes: more }, 'Raise the limit to ' + minutes(more), backendName(run.backend, plan.gpu) + ' · ' + (same.est_usd ? money(same.est_usd) : run.backend === 'local' ? 'no cost' : 'cost not estimated'), !others.length)
      : html`<div class="choice choice-off"><b>The limit cannot be raised here</b><span>${minutes(usedMin)} is the most one run may take in this project. Raise the limit for one run on the overview first.</span></div>`}${edit}`;
  } else if (kind === 'budget') {
    title = 'Refused: over the experiment budget';
    say = html`<p>The run did not start. Raise the experiment budget of the project, or change the plan so that it costs less.</p>`;
    actions = html`${elsewhere}${again}`;
  } else if (kind === 'backend') {
    title = 'The backend could not run it';
    say = html`<p>This is the backend's own failure, not the experiment's: the job did not get to run its code to an end. Nothing in the code was repaired.</p>`;
    actions = html`${again}${elsewhere}${html`<a class="choice" href="#/setup"><b>Open Setup</b><span>What this machine can use, and what is missing.</span></a>`}`;
  } else if (kind === 'cancelled') {
    /* stopped together with the agents: the experiment is queued again, the page offers "Run now" above */
    title = run.requeued ? 'Stopped when you stopped the agents' : 'Cancelled by you';
    say = run.requeued
      ? html`<p>This run was under way${run.seconds != null ? ' for ' + duration(run.seconds) : ''} when you pressed Stop. The experiment is queued again: it runs first when you continue, or now. What the run had written is kept below.</p>`
      : html`<p>The run was stopped at your request${run.seconds != null ? ' after ' + duration(run.seconds) : ''}. Nothing was repaired or retried, and what it had written is kept below.</p>`;
    actions = run.requeued ? '' : html`${again}${elsewhere}`;
  } else {
    const repairs = run.repairs || [];
    title = byAgent(run) ? 'The agent did not bring the run to a result' : 'Failed with a programming error';
    say = byAgent(run) ? html`<p>The agent that carried out this run repairs the code while it works, so no separate repair is made afterwards. Its log below says what it tried.</p>` : html`<p>${repairs.length ? html`The runner repaired the code ${repairs.length === 1 ? 'once' : repairs.length + ' times'} before this run (${repairs.length}${limits.repair_rounds != null ? ' of ' + limits.repair_rounds : ''} repairs used), and it failed again.` : 'The code ended with an error. The runner has not repaired it.'}</p>`;
    actions = html`${edit}${again}${elsewhere}`;
  }
  /* a run the person cancelled is their own doing: it is said plainly, not as an alarm */
  if (kind === 'cancelled') return html`<div class="failure k-cancelled"><h3 class="fail-t">${icon('stop')}${title}</h3>${say}${actions ? html`<div class="choices">${actions}</div>` : ''}${repairsHTML(run, false)}</div>`;
  return html`<div class="failure k-${kind}"><h3 class="fail-t">${icon('warn')}${title}</h3>
    <pre class="code code-err" tabindex="0" data-follow>${kind === 'program' && tail ? tail : plain(run.error) || 'No error message was recorded.'}</pre>
    ${say}
    <div class="choices">${actions}</div>
    ${repairsHTML(run, false)}</div>`;
}
/** What the runner changed in the code before a run, with the diffs. A run that succeeded
    only after a repair says so: its numbers come from code the person has not read yet. */
function repairsHTML(run, succeeded) {
  const repairs = run.repairs || [];
  if (!repairs.length) return '';
  return html`<div class="repair${succeeded ? ' repair-ok' : ''}"><h4>${succeeded ? html`${icon('warn')}The runner changed the code before this run` : 'What the runner changed before this run'}</h4>
    ${succeeded ? html`<p>The run before it failed with a programming error, and the runner repaired the code ${repairs.length === 1 ? 'once' : repairs.length + ' times'}. These metrics come from the repaired code: read what changed before you rely on them.</p>` : ''}
    ${repairs.map((r) => html`<details class="repair-i" data-keep="repair:${run.experiment}#${run.number}/${r.round}"${(repairs.length === 1 && !succeeded ? raw(' open') : '')}><summary>Repair ${r.round}${r.after_run ? ' after run ' + r.after_run : ''}${(r.files || []).length ? ' · ' + r.files.join(', ') : ''}</summary><p>${r.reason}</p>${r.diff ? diffText(r.diff) : ''}</details>`)}</div>`;
}

/** The metrics the plan names under "Metrics and success criteria". A name can stand for a
    family: `spike_<configuration>` and `final_loss_` name every metric that starts so. */
function namedByPlan(data, keys) {
  const body = String((data && data.page && data.page.body) || '');
  const section = (body.split(/^#+[ \t]*metrics[^\n]*$/im)[1] || '').split(/^#+[ \t]/m)[0];
  const tests = (section.match(/`[^`\n]+`/g) || []).map((span) => {
    const name = span.slice(1, -1).trim();
    return new RegExp('^' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/<[^>]*>/g, '.+') + (/_$/.test(name) ? '.+' : '') + '$');
  });
  return keys.filter((k) => tests.some((t) => t.test(k)));
}
/** A long table would push everything else down: what the plan names comes first (or the
    first eight), the rest opens on demand. */
function metricsHTML(run, data) {
  const m = run.metrics_text || run.metrics || {};
  const keys = Object.keys(m);
  if (!keys.length) return html`<p class="quiet">The run wrote no metrics.</p>`;
  const table = (list) => html`<table class="metrics"><tbody>${list.map((k) => html`<tr><td><code>${k}</code></td><td class="n">${m[k]}</td></tr>`)}</tbody></table>`;
  const named = namedByPlan(data, keys);
  const first = named.length ? named : keys.slice(0, 8);
  if (keys.length <= 8 || first.length === keys.length) return table(keys);
  const rest = keys.filter((k) => first.indexOf(k) < 0);
  return html`${table(first)}<details class="metrics-all" data-keep="metrics:${run.experiment}#${run.number}"><summary>All ${keys.length} metrics <span class="quiet">${named.length ? rest.length + ' more that the plan does not name' : rest.length + ' more'}</span></summary>${table(rest)}</details>`;
}
function artifactsHTML(store, eid, run) {
  const urls = run.artifact_urls || {};
  const names = Object.keys(urls);
  if (!names.length) return html`<p class="quiet">The run wrote no artifacts.</p>`;
  return join(names.map((name) => (IMAGE.test(name)
    ? html`<figure class="fig"><a href="${urls[name]}" target="_blank" rel="noopener"><img src="${urls[name]}" alt="${name} of run ${run.number}" loading="lazy"></a><figcaption><code>${name}</code></figcaption></figure>`
    : html`<a class="file" href="${urls[name]}" target="_blank" rel="noopener">${icon('file')}<span>${name}</span></a>`)));
}
function findingsOf(store, run) {
  if (!run.result_page) return [];
  const a = store.adj.get(run.result_page);
  return a ? a.inn.filter((e) => e.type === 'based_on').map((e) => store.byPath.get(e.s)).filter(Boolean) : [];
}

/** What it means that an agent carried a run out, said once on the run. */
const AGENT_RUN = 'An agent carried out this run. It worked in a shell in the run\'s own folder, starting from the approved code; the log is what it did and said, and the metrics and artifacts are what its work left. Nobody measured an exit status: the agent may have run the code more than once.';
/** What the agent that carried a run out made of the approved code (`run.code_as_run`):
    the files it changed, added and removed, each one a link to the file as it left it,
    and for the person the choice to make that code the experiment's. */
function codeAsRunHTML(store, eid, run) {
  const c = run.code_as_run;
  if (!byAgent(run)) return '';
  if (!c) return run.status === 'running' ? '' : html`<p class="run-code" data-code="unknown">What the agent made of the approved code was not kept for this run.</p>`;
  const kept = (name) => html`<a class="linkish" href="${api.fileUrl(store.pid, 'experiments/' + eid + '/runs/' + run.number + '/code-as-run/' + name)}" target="_blank" rel="noopener"><code>${name}</code></a>`;
  const part = (verb, names, linked) => ((names || []).length ? html`<span class="run-code-part">${verb} ${names.map((n, i) => html`${i ? ', ' : ''}${linked ? kept(n) : html`<code>${n}</code>`}`)}</span>` : '');
  const changed = (c.changed || []).length + (c.added || []).length + (c.removed || []).length;
  if (!changed) return html`<p class="run-code" data-code="approved">The agent left the approved code as it was <span class="quiet">(digest ${String(c.digest || '').slice(0, 12)}…)</span>.</p>`;
  const busy = app.ui.adopting === eid + '/' + run.number;
  /* What the sentence may say depends on how the run ended and on whether there was approved
     code at all: a run that left no results has none that "come from that code", and a plan
     that came without code has no approved code to compare with. */
  const hadCode = c.approved != null ? c.approved > 0 : (c.changed || []).length + (c.removed || []).length > 0;
  const succeeded = run.status === 'succeeded';
  const digest = html`<span class="quiet">(digest ${String(c.digest || '').slice(0, 12)}…)</span>`;
  const did = html`${part('changed', c.changed, true)}${(c.changed || []).length && ((c.added || []).length || (c.removed || []).length) ? '; ' : ''}${part('added', c.added, true)}${(c.added || []).length && (c.removed || []).length ? '; ' : ''}${part('removed', c.removed, false)}`;
  const means = succeeded
    ? (hadCode ? html`The results of this run come from that code ${digest}, not from the approved code shown under Code, which is unchanged: run again, the experiment would start from the approved code and be repaired again.`
      : html`The results of this run come from that code ${digest}. The plan came without code, so the experiment has none of its own yet: run again, an agent would write it anew.`)
    : html`That is the code as it stood when the run ${run.status === 'cancelled' ? 'was cancelled' : 'ended'} ${digest}; this run left no results that come from it.${hadCode ? ' The approved code shown under Code is unchanged.' : ' The plan came without code, so the experiment has none of its own yet.'}`;
  return html`<div class="note note-warn run-code" data-code="changed" data-had-code="${tf(hadCode)}">${icon('warn')}<div><b>${hadCode ? 'The agent changed the code it was given.' : 'The agent wrote the code for this run.'}</b>
    <p>It ${did}. ${means}</p>
    ${(c.changed || []).length + (c.added || []).length ? html`<button class="btn btn-s" data-act="adopt-code" data-eid="${eid}" data-run="${run.number}"${busy ? raw(' disabled') : ''}>${busy ? 'Taking it over…' : 'Make this the experiment\'s code'}</button> <span class="quiet">As your own edit: the earlier version is kept, and the experiment must be approved again to run.</span>` : ''}</div></div>`;
}
/** A run whose code could call the Anthropic API: nothing here counted those calls. */
const modelAccessHTML = (run) => (run.model_access ? html`<p class="run-models" data-model-access>This run was approved with network access while model access for jobs was on: its code could call the Anthropic API with the worker's federation settings. Those calls are billed to that account and are counted nowhere here.</p>` : '');
function runBlock(store, page, data, run, open) {
  const eid = page.id, failed = run.status === 'failed' || run.status === 'cancelled';
  const head = html`<button class="run-h" data-act="show-run" data-eid="${eid}" data-run="${run.number}" aria-expanded="${tf(open)}">${icon(open ? 'down' : 'right')}<b>Run ${run.number}</b>
    <span class="st-word t-${TONE[run.status] || 'open'}">${glyph('result', 'accepted', run.status, 12)}${run.status}</span>
    <span class="run-m">${backendName(run.backend, run.plan && run.plan.gpu)}${byAgent(run) ? html` <span class="run-by" data-by="agent">by an agent</span>` : ''}</span><span class="run-m">${dateTime(run.started)}</span>
    <span class="run-m n">${run.status === 'running' ? sinceClock(run.started) : duration(run.seconds)}</span><span class="run-m n">${run.cost_usd != null ? money(run.cost_usd) : run.backend === 'local' ? 'no cost' : ''}</span>
    <span class="run-m n">${run.exit_code == null || byAgent(run) ? '' : 'exit ' + run.exit_code}</span><span class="run-m">${(run.repairs || []).length ? plural(run.repairs.length, 'repair') : ''}</span></button>`;
  if (!open) return html`<div class="run">${head}</div>`;
  const findings = findingsOf(store, run);
  const result = run.result_page && store.byPath.get(run.result_page);
  return html`<div class="run open">${head}<div class="run-b">
    ${byAgent(run) ? html`<p class="run-agent" data-by="agent">${AGENT_RUN}</p>` : ''}
    ${codeAsRunHTML(store, eid, run)}${modelAccessHTML(run)}
    ${failed ? failureHTML(store, page, data, run) : ''}
    ${run.status === 'succeeded' ? repairsHTML(run, true) : ''}
    ${run.status === 'succeeded' ? html`<div class="res-grid"><div><h3 class="sec-h">Metrics <span class="quiet">${byAgent(run) ? 'metrics.json, as the agent\'s work left it' : 'metrics.json, as the run wrote it'}</span></h3>${metricsHTML(run, data)}</div>
      <div><h3 class="sec-h">Artifacts</h3>${artifactsHTML(store, eid, run)}
        ${findings.length ? html`<h3 class="sec-h res-h">Findings from this run</h3>${findings.map((f) => html`<a class="lin" href="${link({ view: 'wiki', sel: f.path, sub: '' })}" data-sel="${f.path}">${pageGlyph(f, 16)}<span class="lin-name"><b>${f.id}</b> ${f.title}</span></a>`)}` : ''}</div></div>` : ''}
    ${failed && Object.keys(run.metrics || {}).length ? html`<h3 class="sec-h">Metrics <span class="quiet">what the run had written when it ended</span></h3>${metricsHTML(run, data)}` : ''}
    ${(run.notes || []).length ? html`<ul class="run-notes">${run.notes.map((n) => html`<li>${n}</li>`)}</ul>` : ''}
    ${logHTML(store, eid, run)}
    <p class="run-rec">${result ? html`Result record <a class="chip" href="${link({ view: 'wiki', sel: result.path, sub: '' })}" data-sel="${result.path}">${pageGlyph(result, 12)}${label(result)}</a> was written by code from this run${(result.meta || {}).edited_by_person ? html`, and <b class="rec-edited">edited by hand since</b>: it no longer reads as code wrote it` : ''}.` : run.status === 'running' ? 'The result record is written by code when the run ends.' : ''}
      ${run.code_digest ? html`<span class="quiet">${byAgent(run) ? 'Digest of the approved code' : 'Code digest'} ${String(run.code_digest).slice(0, 12)}…</span>` : ''}</p></div></div>`;
}

function liveHTML(store, page, data, run, does) {
  const eid = page.id, plan = data.plan || {};
  const est = plan.est_minutes ? plan.est_minutes * 60 : null, limit = (run.plan && run.plan.timeout_seconds) || plan.timeout_seconds || null;
  const elapsed = (Date.now() - (parseStamp(run.started) || Date.now())) / 1000;
  const total = est || limit || 0;
  return html`<section class="run-live"><div class="run-top"><h2><span class="live-dot" aria-hidden="true"></span>${byAgent(run) ? html`An agent is carrying out run ${run.number} on ${backendName(run.backend, plan.gpu)}` : html`Run ${run.number} is running on ${backendName(run.backend, plan.gpu)}`}</h2>
    ${cancelButton(eid, does, { cls: 'btn-neg-quiet', stop: true, label: 'Cancel run', tip: 'The job is ended and the run recorded as cancelled' })}</div>
    ${total ? html`<progress class="meter-p" max="1" value="${Math.min(1, elapsed / total)}" data-since="${run.started}" data-total="${total}" aria-label="Elapsed, of the ${est ? 'estimate' : 'time limit'}"></progress>` : ''}
    <dl class="run-facts"><div><dt>Elapsed</dt><dd>${sinceClock(run.started)}${est ? html` <span class="quiet">of about ${minutes(plan.est_minutes)}</span>` : ''}</dd></div>
      <div><dt>Limit</dt><dd>${limit ? minutes(limit / 60) : 'none'} <span class="quiet">${byAgent(run) ? 'for all of the agent\'s work, then it is stopped' : 'then it is stopped'}</span></dd></div>
      <div><dt>Cost</dt><dd>${byAgent(run) ? html`no cost <span class="quiet">for the machine; the agent's model counts as model spend</span>` : run.backend === 'local' ? 'no cost' : plan.est_usd != null ? 'about ' + money(plan.est_usd) : 'not estimated'}</dd></div>
      <div><dt>Started</dt><dd>${dateTime(run.started)}</dd></div></dl>
    ${logHTML(store, eid, run)}</section>`;
}

/** Who wrote the file that is shown, and when. Code that arrived after the plan was accepted says so. */
function codeBy(page, file, approval) {
  if (!file.author) return html`<p class="code-by"><span><code>${file.name}</code> has no recorded author: it was put into the project by hand.</span></p>`;
  const decided = parseStamp((page.meta || {}).decided_at), written = parseStamp(file.written) || 0;
  const late = page.review === 'accepted' && file.author !== 'human' && decided != null && written > decided;
  /* an approval made since the file was written covers it: the person has been asked to read it */
  const covered = !!(approval && approval.matches && approval.at && (parseStamp(approval.at) || 0) >= written);
  return html`<p class="code-by"><span><code>${file.name}</code> was written by ${who(file.author)} ${ago(file.written)}${late ? ', after you accepted the plan' : ''}.</span>${late && !covered ? html`<span class="code-late">${icon('warn')}Read it before you approve a run</span>` : ''}</p>`;
}
function codeSection(store, page, data) {
  const eid = page.id, files = (data.code && data.code.files) || [], plan = data.plan || {};
  const problems = (data.code && data.code.problems) || [];
  const name = files.some((f) => f.name === app.ui.codeTab[eid]) ? app.ui.codeTab[eid] : (files.find((f) => f.name === (plan.entrypoint || 'main.py')) || files[0] || {}).name;
  const file = files.find((f) => f.name === name);
  const editKey = eid + '/' + name, editing = app.ui.codeEdit[editKey];
  const locked = page.state === 'running' || page.state === 'queued';
  return html`<section class="exp-code"><h2 class="sec-t">Code <span class="quiet">${plan.mode === 'repo' ? 'runs inside a copy of your repository; the working tree is never modified' : agentWould(store, 'local') ? 'parsed; the designer may have tried it in its own sandboxed folder, and it does not run for the record before you approve' : 'parsed, never executed before you approve'}</span></h2>
    ${problems.length ? html`<div class="note note-warn">${icon('warn')}<div><b>The check of the code found:</b><ul class="reasons">${problems.map((p) => html`<li>${p}</li>`)}</ul></div></div>` : ''}
    ${files.length ? html`<div class="tabs" role="tablist" aria-label="Code files">${files.map((f) => html`<button role="tab" class="tab${f.name === name ? ' on' : ''}" aria-selected="${tf(f.name === name)}" data-act="code-tab" data-eid="${eid}" data-name="${f.name}">${f.name}<span>${plural(String(f.text).replace(/\n$/, '').split('\n').length, 'line')}</span></button>`)}
        ${editing == null ? html`<button class="btn btn-quiet btn-s tabs-r" data-act="edit-code" data-eid="${eid}"${locked ? raw(' disabled data-tip="The code cannot be changed while it runs"') : ''}>${icon('pencil')}Edit code</button>` : ''}</div>
      ${editing != null ? html`<form class="code-edit" data-form="code" data-eid="${eid}" data-name="${name}">
          <label class="sr" for="code-text">${name}</label><textarea id="code-text" name="text" data-input="code" data-key="${editKey}" spellcheck="false" wrap="off">${editing.text}</textarea>
          ${editing.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span><b>Not saved.</b> ${editing.error} Your text is still here.</span></p>` : ''}
          <div class="code-bar"><span class="quiet">${editing.restored ? html`<b>Your unsaved edit of ${name} was restored.</b> ` : ''}The earlier version is kept. A changed file is parsed, not run; an approved run must be approved again.</span><button class="btn btn-quiet" type="button" data-act="code-cancel" data-key="${editKey}">Discard</button><button class="btn btn-primary" type="submit"${editing.saving ? raw(' disabled') : ''}>Save ${name}</button></div></form>`
        : file ? html`${codeBy(page, file, data.approval)}${codeHTML(file.text)}` : ''}`
      : html`<p class="quiet">${plan.mode === 'repo' ? 'No patch: the command runs on a clean copy of the repository.' : 'No code has been written for this experiment yet.' + (agentWould(store, 'local') ? ' The agent that carries it out on this machine writes it from the plan.' : '')}</p>`}
  </section>`;
}

function detailHTML(store, page, route) {
  const eid = page.id, entry = store.experiment(eid), data = entry && entry.data, row = store.expById.get(eid) || {};
  const tests = testsOf(store, page);
  const meta = page.meta || {};
  const head = html`<header class="page-head"><div class="page-kicker"><span class="kicker-kind">${pageGlyph(page, 18)}<span>Experiment <b>${eid}</b></span></span>
      <span class="page-tools"><a class="ibtn" href="${link({ view: 'map', sel: page.path, sub: '' })}" data-tip="Show on the map (1)" aria-label="Show on the map">${icon('map')}</a><a class="ibtn" href="${link({ view: 'wiki', sel: page.path, sub: '' })}" data-tip="Open in the wiki (2)" aria-label="Open in the wiki">${icon('open')}</a></span></div>
    <h1 class="page-title">${page.title}</h1>${page.summary && page.summary !== page.title ? html`<p class="page-summary">${page.summary}</p>` : ''}
    <div class="page-status">${reviewTag(page)}${stateTag(page)}<span class="page-by"><span>${page.author === 'human' ? 'written' : 'designed'} by ${who(page.author)}${page.round ? ', round ' + page.round : ''}</span>${meta.decided_by && page.review === 'accepted' ? html`<span>accepted by you ${ago(meta.decided_at)}</span>` : ''}${meta.backend ? html`<span>${backendName(meta.backend, meta.gpu)}</span>` : ''}</span>${unacceptLink(page)}</div>
    ${tests.length ? html`<div class="ex-testsline"><span class="quiet">Tests</span>${tests.map((h) => html`<a class="chip" href="${link({ view: 'wiki', sel: h.path, sub: '' })}" data-sel="${h.path}" data-tip="${pageTip(h)}">${pageGlyph(h, 12)}<b>${h.id}</b> ${h.title}</a>`)}</div>` : ''}</header>`;
  if (!data) {
    return html`${head}${entry && entry.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The experiment could not be loaded: ${entry.error.message} <button class="linkish" data-act="reload-experiment" data-eid="${eid}">Try again</button></span></div>`
      : html`<div class="skeleton" aria-busy="true" aria-label="Loading the experiment"><i></i><i></i><i class="short"></i></div>`}`;
  }
  const runs = (data.runs || []).slice().sort((a, b) => (b.number || 0) - (a.number || 0));
  const latest = runs[0];
  const worker = store.project.worker || {};
  const busy = !!worker.state && worker.state !== 'idle';
  /* why a run cannot start right now, where one is offered */
  const held = worker.verified === false ? 'The project\'s files name a worker this server cannot verify: nothing can be started until it has ended or is forgotten (see the overview).' : 'The agents are at work; it runs when they reach it, or when they have stopped.';
  const starting = !!app.ui.starting[eid];
  const interrupted = page.state === 'running' && (data.interrupted || row.interrupted);
  /* What a press of the button that cancels would do as things stand: the server's word,
     from the list that was read together with the page's state (the experiment's own view
     may be a moment older, and about the state before). An approval can only be taken back
     from an experiment that reads approved: anything else is not said at all. */
  const told = row.cancel_does !== undefined ? row.cancel_does : data.page && data.page.state === page.state ? data.cancel_does : undefined;
  const does = told === 'cancel' || (told === 'withdraw' && page.state === 'approved') ? told : undefined;
  let top = '';
  if (page.review === 'rejected') top = reviewBar(store, page, { part: 'page' });
  else if (interrupted && app.ui.reapprove !== eid) {
    /* the page says "running" and nothing runs it: the server's own sentence, Run to start it
       again (what is left of the unfinished run is settled first), or Cancel */
    top = html`<section class="callout callout-changed callout-interrupted" data-interrupted="${eid}"><h2>${icon('warn')}The run was interrupted</h2>
      <p>${plain(data.interrupted_note) || 'The worker that ran ' + eid + ' is gone and nothing is running it now. Run it again to go on, or cancel it.'}</p>
      <div class="row"><button class="btn btn-primary" data-act="run-now" data-eid="${eid}"${busy || starting ? raw(' disabled') : ''}>${icon('play')}${starting ? 'Settling the unfinished run…' : 'Run it again'}</button>
        ${cancelButton(eid, does, { label: 'Cancel it', tip: 'The unfinished run is settled and the experiment recorded as cancelled; it can be approved again' })}</div></section>`;
  }
  else if (page.review === 'proposed' || page.state === 'planned' || !page.state || app.ui.reapprove === eid) top = approveHTML(store, page, data);
  else if ((page.state === 'approved' || page.state === 'queued') && data.approval && data.approval.digest && data.approval.matches === false) {
    /* approved, and then the plan or the code changed: a run would be refused, so the page
       says so before the person presses anything, with what changed and by whom */
    const changes = (data.approval.changed || []).map((c) => whyNot(String(c)));
    top = html`<section class="callout callout-changed"><h2>${icon('warn')}Changed since you approved it</h2>
      <p>You approved this experiment${data.approval.at ? ' ' + ago(data.approval.at) : ''}. It is no longer what you approved, so it will not run as it is. Look at what changed, then approve it again.</p>
      ${changes.length ? html`<ul class="changed-list">${changes.map((c) => html`<li>${c}</li>`)}</ul>` : ''}
      <div class="row"><button class="btn btn-primary" data-act="reapprove" data-eid="${eid}">Look and approve again…</button>
        ${cancelButton(eid, does, {})}</div></section>`;
  } else if (lapsed(row) || lapsed(Object.assign({}, page, { will_not_run: data.will_not_run }))) {
    /* it reads approved and would not run for another reason the server names (the run it was
       approved for is over, or nothing records what was approved): said before a press */
    top = html`<section class="callout callout-changed" data-will-not-run="${eid}"><h2>${icon('warn')}It will not run as it stands</h2>
      <p>${cap(whyNot(row.will_not_run || data.will_not_run))}. Approve it again when you want it run.</p>
      <div class="row"><button class="btn btn-primary" data-act="reapprove" data-eid="${eid}">Look and approve again…</button>
        ${cancelButton(eid, does, {})}</div></section>`;
  } else if (page.state === 'approved' || page.state === 'queued') {
    top = html`<section class="callout"><h2>${page.state === 'queued' ? 'Queued' : 'Approved'} to run on ${backendName(meta.backend || (data.plan || {}).backend, meta.gpu)}${meta.timeout_seconds ? ', for at most ' + minutes(meta.timeout_seconds / 60) : ''}</h2>
      <p>${busy ? held : 'It runs first, and is analysed, when you continue the project; or run it now.'}</p>
      <div class="row"><button class="btn btn-primary" data-act="run-now" data-eid="${eid}"${busy || starting ? raw(' disabled') : ''}>${icon('play')}${starting ? 'Starting…' : 'Run now'}</button><button class="btn" data-act="reapprove" data-eid="${eid}">Change where it runs…</button>
        ${cancelButton(eid, does, {})}</div></section>`;
  } else if (page.state === 'running' && latest && latest.status === 'running') top = liveHTML(store, page, data, latest, does);
  else if (page.review === 'accepted' && !runs.length && (page.state === 'cancelled' || page.state === 'failed' || page.state === 'succeeded') && app.ui.reapprove !== eid) {
    /* it ended without a run on record (cancelled while it waited in the queue): not a dead end */
    top = html`<section class="callout"><h2>${page.state === 'cancelled' ? 'Cancelled before it ran' : cap(page.state) + ', with no run on record'}</h2>
      <p>${meta.state_reason ? plain(meta.state_reason) + ' ' : ''}Nothing ran. Approve it again when you want it run.</p>
      <div class="row"><button class="btn btn-primary" data-act="reapprove" data-eid="${eid}">Approve it again…</button></div></section>`;
  }
  else if (page.state === 'running') top = html`<section class="callout"><h2><span class="live-dot" aria-hidden="true"></span>Starting…</h2><p>The run is being set up.</p></section>`;
  const openRun = app.ui.runOpen[eid] === undefined ? (latest && latest.status !== 'running' ? latest.number : null) : app.ui.runOpen[eid];
  const again = page.review === 'accepted' && (page.state === 'succeeded' || page.state === 'cancelled' || page.state === 'failed') && app.ui.reapprove !== eid;
  const listed = runs.filter((r) => !(r.status === 'running' && page.state === 'running' && !interrupted));
  return html`${head}${planProblemsHTML(page)}${unacceptForm(page)}${top}
    ${listed.length ? html`<section class="exp-runs"><h2 class="sec-t">${listed.length < runs.length ? 'Earlier runs' : 'Runs'} <span class="quiet">${listed.length}</span>${again ? html`<button class="btn btn-s tabs-r" data-act="reapprove" data-eid="${eid}">Run again…</button>` : ''}</h2>
      ${listed.map((r) => runBlock(store, page, data, r, r.number === openRun))}</section>` : ''}
    <div class="exp-cols"><section class="exp-plan"><h2 class="sec-t">Plan</h2><div class="prose prose-s" data-plan="1" data-prose="${page.path}">${raw(data.html)}</div></section>${codeSection(store, page, data)}</div>
    ${runs.length ? '' : html`<section class="exp-runs"><h2 class="sec-t">Runs</h2><p class="quiet">It has not run yet.</p></section>`}`;
}

export const experimentsView = {
  render(root, route) {
    const store = app.store;
    const exps = store.experiments.filter((x) => x.review !== 'rejected' || app.ui.showRejected);
    const selected = store.byPath.get(route.sel);
    const cur = selected && selected.kind === 'experiment' && selected.id ? selected : null;
    const related = (x) => selected && !cur && testsOf(store, x).some((h) => h.path === selected.path);
    if (!exps.length) {
      setHTML(root, html`<div class="blank"><h1>No experiments yet</h1><p>Experiments are designed after you accept a hypothesis. Each one waits here for your approval before anything runs.</p></div>`);
      return;
    }
    if (!cur) {
      const rows = GROUPS.map((g) => {
        const list = exps.filter(g[1]);
        if (!list.length) return '';
        return html`<tr class="ex-g"><th colspan="7">${g[0]}<span>${list.length}</span></th></tr>${list.map((x) => html`
          <tr class="ex-row r-${x.review}${related(x) ? ' rel' : ''}" data-go="${link({ view: 'experiments', sel: x.path, sub: '' })}">
            <td class="ex-id">${pageGlyph(x, 18)}<b>${x.id}</b></td>
            <td class="ex-name"><a href="${link({ view: 'experiments', sel: x.path, sub: '' })}"><span class="pn-t">${x.title}</span></a><span class="ex-sub">${subline(store, x)}</span></td>
            <td class="ex-tests">${testsOf(store, x).map((h) => html`<a class="chip" href="${link({ view: 'experiments', sel: h.path, sub: '' })}" data-sel="${h.path}" data-tip="${pageTip(h)}">${pageGlyph(h, 12)}${h.id}</a>`)}</td>
            <td class="ex-where">${whereOf(x)}</td>
            <td class="ex-st">${stateCell(x)}</td>
            <td class="n">${timeCell(x)}</td>
            <td class="n">${costCell(x)}</td></tr>`)}`;
      });
      setHTML(root, html`<div class="ledger"><header class="view-head ex-head"><div><h1>Experiments</h1>
          <p>${selected && selected.kind === 'hypothesis' ? html`Marked rows test <a href="${link({ view: 'wiki', sel: selected.path })}" data-sel="${selected.path}">${selected.id}</a>, the page you have selected. ` : ''}An experiment runs only after you approve it. Accepting a plan alone leaves it planned.</p></div>${budgetLine(store)}</header>
        <table class="ex-table"><thead><tr><th>ID</th><th>Experiment</th><th>Tests</th><th>Backend</th><th>State</th><th class="n">Time</th><th class="n">Cost</th></tr></thead><tbody>${rows}</tbody></table></div>`);
      afterRender(root);
      return;
    }
    /* one experiment open: a rail of all experiments, and the detail */
    const rail = GROUPS.map((g) => {
      const list = exps.filter(g[1]);
      if (!list.length) return '';
      return html`<div class="idx-group"><h3>${g[0]}<span>${list.length}</span></h3>${list.map((x) => html`
        <a class="idx idx-2 r-${x.review}${x.path === cur.path ? ' cur' : ''}" href="${link({ view: 'experiments', sel: x.path, sub: '' })}"${x.path === cur.path ? raw(' aria-current="page"') : ''}>${pageGlyph(x, 16)}
          <span class="idx-name">${pageName(x)}<span class="idx-sub">${standingWords(x)}</span></span></a>`)}</div>`;
    });
    setHTML(root, html`<div class="exp"><nav class="exp-rail" aria-label="Experiments"><a class="back" href="${link({ view: 'experiments', sel: '', sub: '' })}">All experiments</a>${rail}<p class="rail-budget">${budgetLine(store)}</p></nav>
      <div class="exp-main"><div class="exp-page">${detailHTML(store, cur, route)}</div></div></div>`);
    afterRender(root);
  },
};
