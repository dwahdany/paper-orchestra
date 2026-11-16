/* Experiments: the table of all of them, and one experiment with its approval, plan, code,
   runs, live log, metrics and artifacts. Nothing here runs anything by itself: every run
   starts from a button the person presses. */

import { app, link } from './core.js';
import { api } from './api.js';
import { $$, ago, clock, dateTime, duration, html, join, minutes, money, parseStamp, plural, raw, setHTML, tf } from './util.js';
import { TONE, agentName, glyph, icon, pageGlyph } from './vocab.js';
import { afterRender, approvalChoice, backendName, label, pageName, pageTip, reviewBar, reviewTag, stateTag, who } from './page.js';

const GROUPS = [
  ['Needs your approval', (x) => x.review !== 'rejected' && (x.review === 'proposed' || x.state === 'planned' || !x.state)],
  ['Approved, queued and running', (x) => x.review === 'accepted' && (x.state === 'approved' || x.state === 'queued' || x.state === 'running')],
  ['Failed', (x) => x.review === 'accepted' && x.state === 'failed'],
  ['Finished', (x) => x.review === 'accepted' && (x.state === 'succeeded' || x.state === 'cancelled')],
  ['Rejected', (x) => x.review === 'rejected'],
];
const IMAGE = /\.(png|jpe?g|gif|webp|svg)$/i;

const whereOf = (x) => { const m = x.meta || {}; return m.backend ? backendName(m.backend, m.gpu) : m.mode === 'gpu' ? 'not chosen · needs a GPU' : m.mode === 'repo' ? 'not chosen · your repository' : 'not chosen'; };
const metricLine = (run) => { const m = (run && (run.metrics_text || run.metrics)) || {}; return Object.keys(m).slice(0, 2).map((k) => k + ' ' + m[k]).join(' · '); };
const sinceClock = (stamp) => html`<span data-since="${stamp}">${clock((Date.now() - (parseStamp(stamp) || Date.now())) / 1000)}</span>`;

function subline(store, x) {
  const run = x.last_run || {};
  if (x.review === 'rejected') return html`rejected${(x.meta || {}).decision_reason ? ': ' + x.meta.decision_reason : ''}`;
  if (x.review === 'proposed') return html`waiting for you · proposed ${ago(x.updated)}`;
  if (x.state === 'planned' || !x.state) return html`accepted · runs only after you approve it`;
  if (x.state === 'approved') return html`approved · runs with the next run phase, or now`;
  if (x.state === 'queued') return html`queued · starts when a worker takes it`;
  if (x.state === 'running') return html`<span class="ex-tail" data-tail="${x.id}">${store.tails[x.id] || 'run ' + (run.number || '') + ' started ' + ago(run.started)}</span>`;
  if (x.state === 'succeeded') return html`${metricLine(run) || 'succeeded'}`;
  if (x.state === 'failed') return html`<span class="t-neg">${(x.meta && x.meta.state_reason) || run.error || 'failed'}</span>`;
  if (x.state === 'cancelled') return html`cancelled ${ago(run.finished)}`;
  return '';
}
function stateCell(x) {
  if (x.review === 'rejected') return html`<span class="tag tag-rejected">rejected</span>`;
  if (x.review === 'proposed') return html`<span class="tag tag-proposed">proposed</span>`;
  const tone = TONE[x.state] || 'open';
  return html`<span class="st-word t-${tone}">${glyph('experiment', 'accepted', x.state, 14)}${x.state || 'planned'}</span>`;
}
function timeCell(x) {
  const run = x.last_run || {}, m = x.meta || {};
  if (x.state === 'running' && run.started) return sinceClock(run.started);
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
    if (piece.reset || entry.next == null) entry.text = text; else entry.text += text;
    if (entry.text.length > MAX_LOG) entry.text = entry.text.slice(-MAX_LOG);
    entry.cut = entry.next == null && piece.offset > 0 ? piece.offset : entry.cut || 0;
    entry.next = piece.next;
    const first = !entry.loaded;
    entry.loaded = true;
    if (first && !entry.live) app.render('log', true);
    entry.error = null;
    $$('[data-log="' + CSS.escape(eid + '/' + number) + '"]').forEach((el) => {
      const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      if (piece.reset || el.dataset.empty) { el.textContent = entry.text; delete el.dataset.empty; } else if (text) el.append(text);
      if (stick) el.scrollTop = el.scrollHeight;
    });
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
    <a class="linkish log-open" href="${api.fileUrl(store.pid, 'experiments/' + eid + '/runs/' + run.number + '/log.txt')}" target="_blank" rel="noopener">Open the whole log</a>
    <button class="btn btn-quiet btn-s" data-act="toggle-log" aria-expanded="${tf(open)}">${open ? 'Collapse' : 'Expand'} <kbd>L</kbd></button></div>
    <pre class="log-body" tabindex="0" aria-label="Log of run ${run.number}" data-log="${eid}/${run.number}"${entry.loaded ? '' : raw(' data-empty="1"')} data-follow>${entry.loaded ? entry.text || (live ? '' : 'The run wrote nothing to its log.') : entry.error ? 'The log could not be read: ' + entry.error.message : 'Reading the log…'}</pre></div>`;
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
  const capMin = Math.max(1, Math.floor((limits.experiment_seconds || 900) / 60));
  const planMin = Math.max(1, Math.round((plan.timeout_seconds || 900) / 60));
  const limitMin = chosen.minutes != null && chosen.minutes !== '' ? chosen.minutes : Math.min(planMin, capMin);
  const network = chosen.network != null ? chosen.network : !!plan.network;
  const gpu = chosen.gpu != null ? chosen.gpu : plan.gpu || '';
  const pick = options.find((o) => o.backend === chosen.backend);
  const proposed = page.review === 'proposed';
  const row = (o) => {
    const on = o.backend === chosen.backend;
    return html`<label class="backend${on ? ' on' : ''}${o.can_run ? '' : ' off'}"><input type="radio" name="backend-${eid}" value="${o.backend}" data-input="backend" data-eid="${eid}"${on ? raw(' checked') : ''}${o.can_run ? '' : raw(' disabled')}>
      <span class="be-name">${backendName(o.backend, o.backend === 'modal' ? gpu : '')}${plan.backend === o.backend && o.can_run ? html`<span class="flag">suggested</span>` : ''}</span>
      <span class="be-desc">${o.can_run ? (o.notes || []).join(' ') || (o.backend === 'local' ? 'A subprocess on this machine.' : 'A sandbox made for this job.') : 'Cannot run this plan.'}</span>
      <span class="be-num">${o.est_minutes != null ? 'about ' + minutes(o.est_minutes) : 'not estimated'}</span><span class="be-num">${o.est_usd ? 'about ' + money(o.est_usd) : o.backend === 'local' ? 'no cost' : o.est_usd === 0 ? '$0.00' : 'not estimated'}</span>
      ${o.can_run ? '' : html`<span class="be-flag">${icon('warn')}<span>${(o.reasons || []).join(' ')}</span></span>`}</label>`;
  };
  const sum = pick ? backendName(pick.backend, pick.backend === 'modal' ? gpu : '') + ', for at most ' + minutes(+limitMin || planMin) + (pick.max_usd ? ' and ' + money(pick.max_usd * ((+limitMin || planMin) * 60) / Math.max(1, pick.timeout_seconds || planMin * 60)) : '') : '';
  return html`<section class="approve" id="approve" data-eid="${eid}"><h2>Where should it run?</h2>
    ${proposed ? html`<p class="approve-say"><b>Proposed</b> by the ${agentName(page.author)}${page.round ? ' in round ' + page.round : ''}, ${ago(page.updated)}. It is not part of the wiki until you accept it, and it will not run until you approve it.</p>` : ''}
    ${data.plan_error ? html`<p class="note note-warn">${icon('warn')}<span>The plan cannot be read as it is: ${data.plan_error}</span></p>` : ''}
    <div class="backends" role="radiogroup" aria-label="Backend">${options.map(row)}</div>
    ${data.options_error ? html`<p class="note note-warn">${icon('warn')}<span>${data.options_error}</span></p>` : ''}
    <div class="approve-fields">
      <label><span>Time limit</span><input type="number" id="ap-min-${eid}" min="1" max="${capMin}" step="1" value="${limitMin}" data-input="ap-minutes" data-eid="${eid}" inputmode="numeric"><span class="quiet">minutes, then it is stopped (at most ${capMin} in this project)</span></label>
      ${chosen.backend === 'modal' || plan.mode === 'gpu' ? html`<label><span>GPU</span><input id="ap-gpu-${eid}" value="${gpu}" placeholder="none (CPU)" data-input="ap-gpu" data-eid="${eid}" size="8" autocomplete="off"><span class="quiet">for example T4, L4, A10G</span></label>` : ''}
      <label class="tick"><input type="checkbox" id="ap-net-${eid}" data-input="ap-network" data-eid="${eid}"${network ? raw(' checked') : ''}> Allow network access<span class="quiet">${chosen.backend === 'local' ? 'the local backend cannot block it either way' : 'off unless you tick this'}</span></label>
    </div>
    <div class="approve-opts"><span><b>Mode</b> ${plan.mode || 'script'}</span>${(plan.packages || []).length ? html`<span><b>Packages</b> ${plan.packages.join(', ')}</span>` : ''}${plan.repo ? html`<span><b>Repository</b> ${plan.repo}</span>` : ''}
      <span><b>Left for experiments</b> ${money(Math.max(0, (limits.experiment_usd || 0) - (((store.project.state || {}).spent || {}).experiment_usd || 0)))}</span></div>
    ${app.ui.rejecting === page.path ? html`<form class="reject-form" data-form="reject" data-ref="${page.path}"><label class="sr" for="why-x-${eid}">Reason</label><input id="why-x-${eid}" name="reason" data-input="reason" data-ref="${page.path}" value="${app.ui.reasons[page.path] || ''}" placeholder="Why? The agents read this reason." autocomplete="off"><button class="btn btn-neg" type="submit">Reject</button><button class="btn btn-quiet" type="button" data-act="reject-cancel">Cancel</button></form>`
      : html`<div class="approve-go">${runnable.length ? html`<p>Approving lets the runner execute <code>${(plan.command || ['python', plan.entrypoint || 'main.py']).join(' ')}</code> once on ${sum}. Nothing else is approved.</p>` : html`<p><b>No backend can run this plan as it is.</b> You can still accept the plan and change it, or the code, before approving a run.</p>`}
        <div class="review-actions">${proposed
          ? html`<button class="btn btn-primary" data-act="accept-run" data-ref="${page.path}"${runnable.length ? '' : raw(' disabled')}>${icon('play')}Accept and run</button>
              <button class="btn" data-act="accept" data-ref="${page.path}">Accept, keep planned <kbd>A</kbd></button>
              <button class="btn" data-act="edit" data-ref="${page.path}">Edit <kbd>E</kbd></button>
              <button class="btn btn-neg-quiet" data-act="reject-ask" data-ref="${page.path}">Reject… <kbd>R</kbd></button>`
          : html`<button class="btn btn-primary" data-act="approve" data-eid="${eid}" data-start="1"${runnable.length ? '' : raw(' disabled')}>${icon('play')}Approve and run now</button>
              <button class="btn" data-act="approve" data-eid="${eid}"${runnable.length ? '' : raw(' disabled')}>Approve, run with the next run phase</button>
              <button class="btn" data-act="edit" data-ref="${page.path}">Edit the plan <kbd>E</kbd></button>
              ${app.ui.reapprove === eid ? html`<button class="btn btn-quiet" data-act="reapprove" data-eid="">Cancel</button>` : ''}`}</div></div>`}
  </section>`;
}

export function failureKind(run) {
  const k = run.error_kind;
  if (run.status === 'cancelled') return 'cancelled';
  if (run.timed_out || k === 'timeout') return 'timeout';
  if (k === 'budget') return 'budget';
  if (['backend', 'setup', 'worker', 'internal'].indexOf(k) >= 0) return 'backend';
  if (['exit', 'metrics', 'code', 'patch', 'command'].indexOf(k) >= 0) return 'program';
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
    say = html`<p>This is a limit, not a bug: the run was stopped after ${usedSeconds < 90 ? duration(usedSeconds) : minutes(usedMin)}. The runner did not attempt a repair because no code is wrong.</p>`;
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
    title = 'Cancelled';
    say = html`<p>You cancelled this run. Nothing was repaired or retried.</p>`;
    actions = html`${again}${elsewhere}`;
  } else {
    const repairs = run.repairs || [];
    title = 'Failed with a programming error';
    say = html`<p>${repairs.length ? html`The runner repaired the code ${repairs.length === 1 ? 'once' : repairs.length + ' times'} before this run (${repairs.length} of ${limits.repair_rounds != null ? limits.repair_rounds : 3} repairs used), and it failed again.` : 'The code ended with an error. The runner has not repaired it.'}</p>`;
    actions = html`${edit}${again}${elsewhere}`;
  }
  return html`<div class="failure k-${kind}"><h3 class="fail-t">${icon('warn')}${title}</h3>
    <pre class="code code-err" tabindex="0" data-follow>${kind === 'program' && tail ? tail : run.error || 'No error message was recorded.'}</pre>
    ${say}
    <div class="choices">${actions}</div>
    ${(run.repairs || []).length ? html`<div class="repair"><h4>What the runner changed before this run</h4>${run.repairs.map((r) => html`<details class="repair-i"${(run.repairs.length === 1 ? raw(' open') : '')}><summary>Repair ${r.round}${r.after_run ? ' after run ' + r.after_run : ''}${(r.files || []).length ? ' · ' + r.files.join(', ') : ''}</summary><p>${r.reason}</p>${r.diff ? diffText(r.diff) : ''}</details>`)}</div>` : ''}</div>`;
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

function runBlock(store, page, data, run, open) {
  const eid = page.id, failed = run.status === 'failed' || run.status === 'cancelled';
  const head = html`<button class="run-h" data-act="show-run" data-eid="${eid}" data-run="${run.number}" aria-expanded="${tf(open)}">${icon(open ? 'down' : 'right')}<b>Run ${run.number}</b>
    <span class="st-word t-${TONE[run.status] || 'open'}">${glyph('result', 'accepted', run.status, 12)}${run.status}</span>
    <span class="run-m">${backendName(run.backend, run.plan && run.plan.gpu)}</span><span class="run-m">${dateTime(run.started)}</span>
    <span class="run-m n">${run.status === 'running' ? sinceClock(run.started) : duration(run.seconds)}</span><span class="run-m n">${run.cost_usd != null ? money(run.cost_usd) : run.backend === 'local' ? 'no cost' : ''}</span>
    <span class="run-m n">${run.exit_code == null ? '' : 'exit ' + run.exit_code}</span><span class="run-m">${(run.repairs || []).length ? plural(run.repairs.length, 'repair') : ''}</span></button>`;
  if (!open) return html`<div class="run">${head}</div>`;
  const findings = findingsOf(store, run);
  const result = run.result_page && store.byPath.get(run.result_page);
  return html`<div class="run open">${head}<div class="run-b">
    ${failed ? failureHTML(store, page, data, run) : ''}
    ${run.status === 'succeeded' ? html`<div class="res-grid"><div><h3 class="sec-h">Metrics <span class="quiet">metrics.json, as the run wrote it</span></h3>${metricsHTML(run, data)}</div>
      <div><h3 class="sec-h">Artifacts</h3>${artifactsHTML(store, eid, run)}
        ${findings.length ? html`<h3 class="sec-h res-h">Findings from this run</h3>${findings.map((f) => html`<a class="lin" href="${link({ view: 'wiki', sel: f.path, sub: '' })}" data-sel="${f.path}">${pageGlyph(f, 16)}<span class="lin-name"><b>${f.id}</b> ${f.title}</span></a>`)}` : ''}</div></div>` : ''}
    ${failed && Object.keys(run.metrics || {}).length ? html`<h3 class="sec-h">Metrics <span class="quiet">what the run had written when it ended</span></h3>${metricsHTML(run, data)}` : ''}
    ${(run.notes || []).length ? html`<ul class="run-notes">${run.notes.map((n) => html`<li>${n}</li>`)}</ul>` : ''}
    ${logHTML(store, eid, run)}
    <p class="run-rec">${result ? html`Result record <a class="chip" href="${link({ view: 'wiki', sel: result.path, sub: '' })}" data-sel="${result.path}">${pageGlyph(result, 12)}${label(result)}</a> was written by code from this run.` : run.status === 'running' ? 'The result record is written by code when the run ends.' : ''}
      ${run.code_digest ? html`<span class="quiet">Code digest ${String(run.code_digest).slice(0, 12)}…</span>` : ''}</p></div></div>`;
}

function liveHTML(store, page, data, run) {
  const eid = page.id, plan = data.plan || {};
  const est = plan.est_minutes ? plan.est_minutes * 60 : null, limit = (run.plan && run.plan.timeout_seconds) || plan.timeout_seconds || null;
  const elapsed = (Date.now() - (parseStamp(run.started) || Date.now())) / 1000;
  const total = est || limit || 0;
  return html`<section class="run-live"><div class="run-top"><h2><span class="live-dot" aria-hidden="true"></span>Run ${run.number} is running on ${backendName(run.backend, plan.gpu)}</h2>
    <button class="btn btn-neg-quiet" data-act="cancel-run" data-eid="${eid}">${icon('stop')}Cancel run</button></div>
    ${total ? html`<progress class="meter-p" max="1" value="${Math.min(1, elapsed / total)}" data-since="${run.started}" data-total="${total}" aria-label="Elapsed, of the ${est ? 'estimate' : 'time limit'}"></progress>` : ''}
    <dl class="run-facts"><div><dt>Elapsed</dt><dd>${sinceClock(run.started)}${est ? html` <span class="quiet">of about ${minutes(plan.est_minutes)}</span>` : ''}</dd></div>
      <div><dt>Limit</dt><dd>${limit ? minutes(limit / 60) : 'none'} <span class="quiet">then it is stopped</span></dd></div>
      <div><dt>Cost</dt><dd>${run.backend === 'local' ? 'no cost' : plan.est_usd != null ? 'about ' + money(plan.est_usd) : 'not estimated'}</dd></div>
      <div><dt>Started</dt><dd>${dateTime(run.started)}</dd></div></dl>
    ${logHTML(store, eid, run)}</section>`;
}

/** Who wrote the file that is shown, and when. Code that arrived after the plan was accepted says so. */
function codeBy(page, file) {
  if (!file.author) return html`<p class="code-by"><code>${file.name}</code> has no recorded author: it was put into the project by hand.</p>`;
  const decided = parseStamp((page.meta || {}).decided_at);
  const late = page.review === 'accepted' && file.author !== 'human' && decided != null && (parseStamp(file.written) || 0) > decided;
  return html`<p class="code-by"><code>${file.name}</code> was written by ${who(file.author)} ${ago(file.written)}${late ? html`<span class="code-late">${icon('warn')}after you accepted the plan: read it before you approve a run</span>` : ''}</p>`;
}
function codeSection(store, page, data) {
  const eid = page.id, files = (data.code && data.code.files) || [], plan = data.plan || {};
  const problems = (data.code && data.code.problems) || [];
  const name = files.some((f) => f.name === app.ui.codeTab[eid]) ? app.ui.codeTab[eid] : (files.find((f) => f.name === (plan.entrypoint || 'main.py')) || files[0] || {}).name;
  const file = files.find((f) => f.name === name);
  const editKey = eid + '/' + name, editing = app.ui.codeEdit[editKey];
  const locked = page.state === 'running' || page.state === 'queued';
  return html`<section class="exp-code"><h2 class="sec-t">Code <span class="quiet">${plan.mode === 'repo' ? 'runs inside a copy of your repository; the working tree is never modified' : 'parsed, never executed before you approve'}</span></h2>
    ${problems.length ? html`<div class="note note-warn">${icon('warn')}<div><b>The check of the code found:</b><ul class="reasons">${problems.map((p) => html`<li>${p}</li>`)}</ul></div></div>` : ''}
    ${files.length ? html`<div class="tabs" role="tablist" aria-label="Code files">${files.map((f) => html`<button role="tab" class="tab${f.name === name ? ' on' : ''}" aria-selected="${tf(f.name === name)}" data-act="code-tab" data-eid="${eid}" data-name="${f.name}">${f.name}<span>${plural(String(f.text).replace(/\n$/, '').split('\n').length, 'line')}</span></button>`)}
        ${editing == null ? html`<button class="btn btn-quiet btn-s tabs-r" data-act="edit-code" data-eid="${eid}"${locked ? raw(' disabled data-tip="The code cannot be changed while it runs"') : ''}>${icon('pencil')}Edit code</button>` : ''}</div>
      ${editing != null ? html`<form class="code-edit" data-form="code" data-eid="${eid}" data-name="${name}">
          <label class="sr" for="code-text">${name}</label><textarea id="code-text" name="text" data-input="code" data-key="${editKey}" spellcheck="false" wrap="off">${editing.text}</textarea>
          ${editing.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span><b>Not saved.</b> ${editing.error} Your text is still here.</span></p>` : ''}
          <div class="code-bar"><span class="quiet">The earlier version is kept. A changed file is parsed, not run; an approved run must be approved again.</span><button class="btn btn-quiet" type="button" data-act="code-cancel" data-key="${editKey}">Discard</button><button class="btn btn-primary" type="submit"${editing.saving ? raw(' disabled') : ''}>Save ${name}</button></div></form>`
        : file ? html`${codeBy(page, file)}${codeHTML(file.text)}` : ''}`
      : html`<p class="quiet">${plan.mode === 'repo' ? 'No patch: the command runs on a clean copy of the repository.' : 'No code has been written for this experiment yet.'}</p>`}
  </section>`;
}

function detailHTML(store, page, route) {
  const eid = page.id, entry = store.experiment(eid), data = entry && entry.data, row = store.expById.get(eid) || {};
  const tests = testsOf(store, page);
  const meta = page.meta || {};
  const head = html`<header class="page-head"><div class="page-kicker"><span class="kicker-kind">${pageGlyph(page, 18)}<span>Experiment <b>${eid}</b></span></span>
      <span class="page-tools"><a class="ibtn" href="${link({ view: 'map', sel: page.path, sub: '' })}" data-tip="Show on the map (1)" aria-label="Show on the map">${icon('map')}</a><a class="ibtn" href="${link({ view: 'wiki', sel: page.path, sub: '' })}" data-tip="Open in the wiki (2)" aria-label="Open in the wiki">${icon('open')}</a></span></div>
    <h1 class="page-title">${page.title}</h1>${page.summary && page.summary !== page.title ? html`<p class="page-summary">${page.summary}</p>` : ''}
    <div class="page-status">${reviewTag(page)}${stateTag(page)}<span class="page-by"><span>designed by the ${agentName(page.author)}${page.round ? ', round ' + page.round : ''}</span>${meta.decided_by && page.review === 'accepted' ? html`<span>accepted by you ${ago(meta.decided_at)}</span>` : ''}${meta.backend ? html`<span>${backendName(meta.backend, meta.gpu)}</span>` : ''}</span></div>
    ${tests.length ? html`<div class="ex-testsline"><span class="quiet">Tests</span>${tests.map((h) => html`<a class="chip" href="${link({ view: 'wiki', sel: h.path, sub: '' })}" data-sel="${h.path}" data-tip="${pageTip(h)}">${pageGlyph(h, 12)}<b>${h.id}</b> ${h.title}</a>`)}</div>` : ''}</header>`;
  if (!data) {
    return html`${head}${entry && entry.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The experiment could not be loaded: ${entry.error.message} <button class="linkish" data-act="reload-experiment" data-eid="${eid}">Try again</button></span></div>`
      : html`<div class="skeleton" aria-busy="true" aria-label="Loading the experiment"><i></i><i></i><i class="short"></i></div>`}`;
  }
  const runs = (data.runs || []).slice().sort((a, b) => (b.number || 0) - (a.number || 0));
  const latest = runs[0];
  const busy = store.project.worker && store.project.worker.state !== 'idle';
  let top = '';
  if (page.review === 'rejected') top = reviewBar(store, page, { part: 'page' });
  else if (page.review === 'proposed' || page.state === 'planned' || !page.state || app.ui.reapprove === eid) top = approveHTML(store, page, data);
  else if (page.state === 'approved' || page.state === 'queued') {
    top = html`<section class="callout"><h2>${page.state === 'queued' ? 'Queued' : 'Approved'} to run on ${backendName(meta.backend || (data.plan || {}).backend, meta.gpu)}${meta.timeout_seconds ? ', for at most ' + minutes(meta.timeout_seconds / 60) : ''}</h2>
      <p>${busy ? 'The agents are at work; it runs when they reach it, or when they have stopped.' : 'It runs with the next run phase, or now.'}</p>
      <div class="row"><button class="btn btn-primary" data-act="run-now" data-eid="${eid}"${busy ? raw(' disabled') : ''}>${icon('play')}Run now</button><button class="btn" data-act="reapprove" data-eid="${eid}">Change where it runs…</button></div></section>`;
  } else if (page.state === 'running' && latest && latest.status === 'running') top = liveHTML(store, page, data, latest);
  else if (page.state === 'running') top = html`<section class="callout"><h2><span class="live-dot" aria-hidden="true"></span>Starting…</h2><p>The run is being set up.</p></section>`;
  const openRun = app.ui.runOpen[eid] === undefined ? (latest && latest.status !== 'running' ? latest.number : null) : app.ui.runOpen[eid];
  const again = page.review === 'accepted' && (page.state === 'succeeded' || page.state === 'cancelled' || page.state === 'failed') && app.ui.reapprove !== eid;
  return html`${head}${top}
    ${runs.length ? html`<section class="exp-runs"><h2 class="sec-t">Runs <span class="quiet">${runs.length}</span>${again ? html`<button class="btn btn-s tabs-r" data-act="reapprove" data-eid="${eid}">Run again…</button>` : ''}</h2>
      ${runs.filter((r) => !(r.status === 'running' && page.state === 'running')).map((r) => runBlock(store, page, data, r, r.number === openRun))}</section>` : ''}
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
          <span class="idx-name">${pageName(x)}<span class="idx-sub">${x.review === 'proposed' ? 'proposed · ' + agentName(x.author) : x.review === 'rejected' ? 'rejected' : x.state || 'planned'}</span></span></a>`)}</div>`;
    });
    setHTML(root, html`<div class="exp"><nav class="exp-rail" aria-label="Experiments"><a class="back" href="${link({ view: 'experiments', sel: '', sub: '' })}">All experiments</a>${rail}<p class="rail-budget">${budgetLine(store)}</p></nav>
      <div class="exp-main"><div class="exp-page">${detailHTML(store, cur, route)}</div></div></div>`);
    afterRender(root);
  },
};
