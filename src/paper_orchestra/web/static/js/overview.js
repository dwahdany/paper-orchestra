/* The overview: where the project stands, in one page. The checkpoint sentence, the strip
   map, the rounds, what waits, the argument so far, activity, instructions, spend. */

import { app, link } from './core.js';
import { ago, clock, dayOf, duration, hm, html, money, parseStamp, plural, raw, setHTML } from './util.js';
import { CHECKPOINT_AFTER, KIND, PHASES, STAGES, TONE, agentName, glyph, icon, pageGlyph, phaseName } from './vocab.js';
import { afterRender, label, pageName, pageTip, statusParts } from './page.js';
import { continueLabel, crashText, engineView, foldRuns, runSentence, waitingSentence, whoOf } from './shell.js';
import { summaryOf, whatOf } from './inbox.js';

/** The strip map: every stage as a station, every page as a mark. */
export function stripMap(store) {
  return html`<div class="smap">${STAGES.map((k) => {
    const list = store.pages.filter((pg) => pg.kind === k);
    const wait = list.filter((pg) => pg.review === 'proposed').length + list.filter((pg) => pg.review !== 'proposed' && store.pendingFor(pg.path).length).length;
    const shown = list.length > 36 ? list.slice(0, 35) : list;
    return html`<div class="smap-st${list.length ? '' : ' none'}"><div class="smap-dot"></div>
      <div class="smap-name">${k === 'idea' ? 'Idea' : KIND[k].many}<span>${k === 'idea' ? '' : list.length}</span></div>
      <div class="smap-marks">${shown.map((pg) => html`<a href="${link({ view: 'map', sel: pg.path, sub: '' })}" data-tip="${pageTip(pg)}" aria-label="${pg.title}">${pageGlyph(pg, 16)}</a>`)}${list.length > shown.length ? html`<a class="smap-more" href="${link({ view: 'wiki', sel: shown[0].path, sub: '' })}">+${list.length - shown.length}</a>` : ''}</div>
      ${wait ? html`<div class="smap-wait">${wait} waiting</div>` : ''}</div>`;
  })}</div>`;
}
/** The small one for the library: a dot per stage and the number of pages. */
export function stripMapSmall(counts, waiting) {
  return html`<div class="smap smap-s">${STAGES.map((k) => { const n = (counts || {})[k] || 0; const w = (waiting || {})[k] || 0; return html`<div class="smap-st${n ? '' : ' none'}${w ? ' wait' : ''}"><div class="smap-dot"></div><div class="smap-n">${k === 'idea' || !n ? '' : n}</div></div>`; })}</div>`;
}

function roundsHTML(store, e) {
  const phases = ((store.project.state || {}).phases || []);
  if (!phases.length && !e.next) return '';
  const rounds = {};
  phases.forEach((ph) => (rounds[ph.round] = rounds[ph.round] || []).push(ph));
  const current = e.round || Math.max(1, ...Object.keys(rounds).map(Number));
  const done = (rounds[current] || []).map((ph) => ph.phase);
  /* what is still to come in the open round, in the engine's order */
  const coming = [];
  if (e.next && e.mode !== 'running') {
    let i = PHASES.indexOf(e.next);
    const last = (rounds[current] || []).slice(-1)[0];
    const retry = last && last.phase === e.next && last.status !== 'completed';
    if (i >= 0) for (let k = retry ? i + 1 : i; k < PHASES.length; k++) { if (PHASES[k] === 'intake' || (done.indexOf(PHASES[k]) >= 0 && k !== i)) continue; coming.push(PHASES[k]); }
  }
  const here = e.mode !== 'running' && e.mode !== 'idle';
  return html`<div class="trks">${Object.keys(rounds).sort((a, b) => a - b).map((r) => {
    const list = rounds[r], open = +r === current;
    return html`<div class="trk"><span class="trk-r">Round ${r}</span>${list.map((ph, i) => {
      const isLast = open && i === list.length - 1;
      const made = (ph.produced || []).length;
      const tip = (ph.status === 'completed' ? '' : ph.status + ' · ') + (made ? plural(made, 'page') + ' written · ' : '') + (ph.seconds != null ? duration(ph.seconds) : '') + (ph.attempts > 1 ? ' · ' + ph.attempts + ' attempts' : '') + (ph.error ? ' · ' + ph.error : '');
      return html`<span class="trk-p s-${ph.status}" data-tip="${tip || ph.status}"><i></i>${phaseName(ph.phase)}${ph.status === 'failed' ? ' (failed)' : ph.status === 'interrupted' ? ' (interrupted)' : ph.status === 'running' ? ' (running)' : ''}</span>
        ${ph.status === 'completed' && CHECKPOINT_AFTER[ph.phase] ? html`<span class="trk-c${isLast && here && e.mode === 'checkpoint' ? ' open' : ''}" data-tip="Checkpoint">${isLast && here && e.mode === 'checkpoint' ? 'you are here' : ''}</span>` : isLast && here ? html`<span class="trk-c open">you are here</span>` : ''}`;
    })}${open ? coming.map((ph) => html`<span class="trk-p s-next"><i></i>${phaseName(ph)}</span>`) : ''}</div>`;
  })}${phases.length ? '' : html`<div class="trk"><span class="trk-r">Round 1</span>${coming.map((ph) => html`<span class="trk-p s-next"><i></i>${phaseName(ph)}</span>`)}</div>`}</div>`;
}

/** Each hypothesis with its experiments, its findings and where it stands. */
function argumentHTML(store) {
  const hyps = store.pages.filter((pg) => pg.kind === 'hypothesis' && pg.review !== 'rejected');
  if (!hyps.length) return html`<p class="quiet">No hypothesis yet. They are proposed after the first round of research.</p>`;
  return html`<table class="arg"><thead><tr><th colspan="2">Hypothesis</th><th>Evidence</th><th>Stands as</th></tr></thead><tbody>${hyps.map((h) => {
    const inn = (store.adj.get(h.path) || { inn: [] }).inn;
    const exps = inn.filter((e) => e.type === 'tests').map((e) => store.byPath.get(e.s)).filter((x) => x && x.review !== 'rejected');
    const finds = inn.filter((e) => e.type === 'supports' || e.type === 'refutes').map((e) => ({ page: store.byPath.get(e.s), type: e.type })).filter((f) => f.page && f.page.review !== 'rejected');
    const change = store.pendingFor(h.path)[0];
    return html`<tr class="r-${h.review}"><td class="arg-id"><a href="${link({ view: 'wiki', sel: h.path, sub: '' })}" data-sel="${h.path}">${pageGlyph(h, 16)}<b>${h.id}</b></a></td>
      <td class="arg-t"><a href="${link({ view: 'wiki', sel: h.path, sub: '' })}" data-sel="${h.path}" data-tip="${h.summary || h.title}">${h.title}</a></td>
      <td class="arg-ev">${exps.map((x) => html`<a class="chip" href="${link({ view: 'experiments', sel: x.path, sub: '' })}" data-tip="${pageTip(x)} · ${statusParts(store, x).join(' · ')}">${pageGlyph(x, 12)}${x.id}</a>`)}
        ${finds.map((f) => html`<a class="chip chip-${f.type}" href="${link({ view: 'wiki', sel: f.page.path, sub: '' })}" data-sel="${f.page.path}" data-tip="${pageTip(f.page)}">${pageGlyph(f.page, 12)}${f.page.id} ${f.type}</a>`)}
        ${exps.length || finds.length ? '' : html`<span class="quiet">nothing yet</span>`}</td>
      <td class="arg-st">${h.review === 'proposed' ? html`<span class="st-word t-prop">proposed</span>` : html`<span class="st-word t-${TONE[h.state] || 'open'}">${glyph('hypothesis', 'accepted', h.state, 14)}${h.state || 'open'}</span>`}${change && change.fields && change.fields.state ? html`<span class="arg-chg">→ ${change.fields.state} suggested</span>` : ''}</td></tr>`;
  })}</tbody></table>`;
}

function feedHTML(store) {
  const events = foldRuns(store.activity.filter((ev) => ev.type !== 'run.log')).slice(-app.ui.feedLimit).reverse();
  if (!events.length) return html`<p class="quiet">Nothing has happened yet.</p>`;
  let day = '';
  const rows = events.map((ev) => {
    const d = dayOf(ev.ts);
    const head = d !== day ? html`<li class="ev-day">${d}</li>` : '';
    day = d;
    if (ev.run) {
      const path = (store.expById.get(ev.eid) || {}).path;
      return html`${head}<li class="ev ev-run ev-${ev.status}"><time>${hm(ev.ts)}</time><span class="ev-who">runner</span>
        <span class="ev-run-l">${path ? html`<a href="${link({ view: 'experiments', sel: path, sub: '' })}">${runSentence(ev)}</a>` : runSentence(ev)}
          <details class="ev-steps" data-keep="run:${ev.key}"><summary>${plural(ev.parts.length, 'step')}</summary><ol>${ev.parts.map((p) => html`<li><time>${hm(p.ts)}</time><span>${p.summary}</span></li>`)}</ol></details></span></li>`;
    }
    const you = whoOf(ev) === 'you';
    const ref = (ev.refs || []).find((r) => store.byPath.has(r));
    return html`${head}<li class="ev ev-${String(ev.type).replace(/[^a-z]/g, '-')}${you ? ' ev-you' : whoOf(ev) === 'engine' ? ' ev-engine' : ''}"><time>${hm(ev.ts)}</time><span class="ev-who">${whoOf(ev)}</span>
      ${ref ? html`<a href="${link({ view: 'wiki', sel: ref, sub: '' })}" data-sel="${ref}">${ev.summary}</a>` : html`<span>${ev.summary}</span>`}</li>`;
  });
  const more = store.activity.length > app.ui.feedLimit || store.moreActivity;
  return html`<ol class="feed">${rows}</ol>${more ? html`<button class="btn btn-quiet btn-s feed-more" data-act="feed-more">Show earlier activity</button>` : ''}`;
}

function requestsHTML(store, e) {
  const sel = app.route.sel && store.byPath.get(app.route.sel);
  const rq = app.ui.request;
  const next = e.mode === 'running' ? null : e.next;
  const list = store.requests.slice().reverse();
  return html`<form class="rq-form" data-form="request"><label class="sr" for="rq-text">Instruction</label>
    <textarea id="rq-text" name="text" data-input="request" rows="2" placeholder="Tell the agents what to do next, in your own words. For example: compare against RMSprop as well.">${rq.text}</textarea>
    <div class="rq-bar"><span class="quiet">${sel ? html`<label class="tick"><input type="checkbox" id="rq-about" data-input="request-about"${rq.about ? raw(' checked') : ''}> about ${label(sel)}</label> · ` : ''}Given to the next phase that runs${next ? ' (' + phaseName(next) + ')' : ''}. You can withdraw it until then.</span>
      <button class="btn" type="submit"${rq.sending ? raw(' disabled') : ''}>Add instruction</button></div>
    ${rq.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span><b>Not added.</b> ${rq.error} Your text is still in the box.</span></p>` : ''}</form>
    <ul class="rqs">${list.map((r) => { const target = r.target && store.byPath.get(r.target); return html`<li class="rq rq-${r.state}"><span class="rq-t">${r.text}</span>
      <span class="rq-m">${r.state === 'open' ? 'open · ' + ago(r.created) : 'done · read in round ' + ((r.handled_in || {}).round || '?') + ', ' + phaseName((r.handled_in || {}).phase)}${target ? html` · about <a href="${link({ view: 'wiki', sel: target.path, sub: '' })}" data-sel="${target.path}">${label(target)}</a>` : ''}</span>
      ${r.state === 'open' ? html`<button class="ibtn" data-act="rq-del" data-ref="${r.id}" data-tip="Withdraw" aria-label="Withdraw this instruction">${icon('x')}</button>` : ''}</li>`; })}</ul>`;
}

function limitsForm(limits) {
  const ui = app.ui.limits || (app.ui.limits = { model_usd: String(limits.model_usd), experiment_usd: String(limits.experiment_usd), minutes: String(Math.round((limits.experiment_seconds || 0) / 6) / 10), error: null, saving: false });
  const field = (name, label, unit) => html`<label><span>${label}</span><input id="lim-${name}" data-input="limits" data-name="${name}" value="${ui[name]}" inputmode="decimal" autocomplete="off" size="7"><span class="quiet">${unit}</span></label>`;
  return html`<form class="limits-form" data-form="limits">${field('model_usd', 'Models', 'dollars in all')}${field('experiment_usd', 'Experiments', 'dollars in all')}${field('minutes', 'One run', 'minutes at most')}
    ${ui.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span>${ui.error}</span></p>` : ''}
    <button class="btn btn-s" type="submit"${ui.saving ? raw(' disabled') : ''}>Save the limits</button></form>`;
}

function spendHTML(store) {
  const p = store.project, limits = (p.settings || {}).limits || {}, spent = (p.state || {}).spent || {};
  const frac = (a, b) => (b > 0 ? Math.max(0, Math.min(1, a / b)) : 0);
  const phases = ((p.state || {}).phases || []).filter((ph) => ph.seconds != null);
  const agentSeconds = phases.reduce((a, ph) => a + (ph.seconds || 0), 0);
  return html`<div class="spend-big">
    <div><b>${money(spent.model_usd || 0)}</b><span>of ${money(limits.model_usd)} for models</span><progress class="meter-p${frac(spent.model_usd || 0, limits.model_usd) > 0.85 ? ' hot' : ''}" max="1" value="${frac(spent.model_usd || 0, limits.model_usd)}" aria-label="Model spend, of the limit"></progress></div>
    <div><b>${money(spent.experiment_usd || 0)}</b><span>of ${money(limits.experiment_usd)} for experiments</span><progress class="meter-p" max="1" value="${frac(spent.experiment_usd || 0, limits.experiment_usd)}" aria-label="Experiment spend, of the limit"></progress></div>
    <div><b>${duration(agentSeconds) || '0 s'}</b><span>of agent time · ${duration(spent.experiment_seconds || 0) || '0 s'} of runs</span></div></div>
    ${phases.length ? html`<details class="spend-more"><summary>Time by phase</summary><table class="metrics"><tbody>${phases.map((ph) => html`<tr><td>Round ${ph.round} · ${phaseName(ph.phase)}${ph.status !== 'completed' ? ' (' + ph.status + ')' : ''}</td><td class="n">${duration(ph.seconds)}</td><td class="n">${plural((ph.produced || []).length, 'page')}</td></tr>`)}</tbody></table></details>` : ''}
    <details class="spend-more"${app.ui.limitsOpen ? raw(' open') : ''}><summary data-act="limits-open">Change the limits</summary>${limitsForm(limits)}</details>
    <p class="quiet spend-note">The engine stops at a checkpoint when a limit is reached, and says so.</p>`;
}

export const overviewView = {
  render(root) {
    const store = app.store, p = store.project, e = engineView(store), items = store.pending();
    const running = store.experiments.filter((x) => x.state === 'running' || x.state === 'queued');
    const broken = store.experiments.filter((x) => x.state === 'failed' && x.review !== 'rejected');
    const waitingApproval = store.experiments.filter((x) => x.review === 'accepted' && (x.state === 'planned' || !x.state));
    const lastAct = store.activity.slice().reverse().find((ev) => ev.agent);
    const headline = e.mode === 'running' ? 'Round ' + (e.round || 1) + ': the ' + (lastAct ? lastAct.agent : 'engine') + ' is at work on ' + phaseName(e.phase) + '.'
      : e.mode === 'experiment' ? 'An experiment is running.' : e.mode === 'writing' ? 'The paper is being written.'
      : e.mode === 'failed' ? cap1(phaseName(e.next || e.phase)) + ' failed.' : e.died || e.crashed ? 'The agents stopped unexpectedly.'
      : e.mode === 'stopped' ? 'You stopped the agents during ' + phaseName(e.next || e.phase) + '.' : e.fresh ? 'Research has not started.'
      : e.budget ? 'Round ' + (e.round || 1) + ' stopped at a budget.' : 'Round ' + (e.round || 1) + ' is paused at a checkpoint.';
    const sub = e.mode === 'running' || e.mode === 'experiment' || e.mode === 'writing' ? 'You can keep reading and deciding while it works.'
      : e.mode === 'failed' ? e.error : e.died ? e.error : e.crashed ? crashText(store, e.exit)
      : e.fresh ? 'Nothing has been spent. The idea' + ((p.attachments || []).length ? ' and ' + plural(p.attachments.length, 'file') : '') + ' are in place.'
      : e.budget ? e.budget : waitingSentence(store);
    const idle = e.mode !== 'running' && e.mode !== 'experiment' && e.mode !== 'writing';
    const actions = !idle ? html`<a class="btn btn-l" href="${link({ view: 'map', sel: '', sub: '' })}">Watch on the map</a><button class="btn btn-l" data-act="stop"${e.stopping ? raw(' disabled') : ''}>${icon('stop')}${e.stopping ? 'Stopping…' : 'Stop'}</button>`
      : html`${items.length ? html`<a class="btn btn-l" href="${link({ view: 'inbox', sel: '', sub: '' })}">Review ${items.length}</a>` : ''}
        <button class="btn btn-l btn-primary" data-act="continue">${e.fresh ? 'Start research' : e.mode === 'failed' ? 'Retry ' + phaseName(e.next || e.phase) : e.mode === 'stopped' ? 'Resume ' + phaseName(e.next || e.phase) : continueLabel(e.next, store)} ${icon('right')}</button>
        ${e.mode === 'failed' || e.died || e.crashed ? html`<a class="btn btn-l" href="#/setup">Open Setup</a>` : ''}`;
    const waiting = items.length ? items.map((it) => { const sum = summaryOf(it); return html`<div class="ov-item" data-ref="${it.ref}"><a class="ov-main" href="${link({ view: 'inbox', sel: it.page.path, sub: '' })}">${glyph(it.kind, it.type === 'page' ? 'proposed' : 'accepted', it.page.state, 18)}
        <span class="in-name">${pageName(it.page)}</span>${sum ? html`<span class="in-sum">${sum}</span>` : ''}<span class="in-what"><span class="in-kind">${whatOf(it)}</span>${agentName(it.author)} · ${ago(it.when)}</span></a>
        <span class="in-quick on"><button class="ibtn ibtn-yes" data-act="accept" data-ref="${it.ref}" data-tip="Accept" aria-label="Accept">${icon('check')}</button><button class="ibtn ibtn-no" data-act="reject-ask" data-ref="${it.ref}" data-key="${it.page.path}" data-view="inbox" data-tip="Reject, with a reason" aria-label="Reject">${icon('x')}</button></span></div>`; })
      : html`<p class="quiet ov-none">Nothing is waiting for you.</p>`;
    const watch = html`${running.map((x) => { const run = x.last_run && x.last_run.status === 'running' ? x.last_run : null; const est = (x.meta || {}).est_minutes; return html`<a class="ov-run" href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${pageGlyph(x, 18)}<span class="in-name">${pageName(x)}</span>
        <span class="in-what">${run ? html`run ${run.number} · <span data-since="${run.started}">${clock((Date.now() - (parseStamp(run.started) || Date.now())) / 1000)}</span>${est ? ' of about ' + est + ' min' : ''}` : 'queued'}</span>
        ${run && est ? html`<progress class="meter-p" max="1" value="0" data-since="${run.started}" data-total="${est * 60}" aria-label="Elapsed, of the estimate"></progress>` : ''}<code class="ov-tail" data-tail="${x.id}">${store.tails[x.id] || ''}</code></a>`; })}
      ${broken.map((x) => html`<a class="ov-run ov-broken" href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${pageGlyph(x, 18)}<span class="in-name">${pageName(x)}</span><span class="in-what">failed${x.runs > 1 ? ' ' + x.runs + ' times' : ''} · ${(x.meta || {}).state_reason || (x.last_run || {}).error || 'needs a decision from you'}</span></a>`)}
      ${waitingApproval.map((x) => html`<a class="ov-run" href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${pageGlyph(x, 18)}<span class="in-name">${pageName(x)}</span><span class="in-what">accepted · runs only after you approve it</span></a>`)}`;
    const anyWatch = running.length + broken.length + waitingApproval.length;
    const problems = (p.problems || []);
    setHTML(root, html`<div class="overview"><div class="ov-wrap">
      <header class="ov-head"><div class="ov-state">${idle ? '' : html`<span class="live-dot"></span>`}${p.title} · ${e.round ? 'round ' + e.round : 'new'}</div><h1>${headline}</h1><p class="ov-sub">${sub}</p>
        ${e.mode === 'checkpoint' && p.checkpoint && p.checkpoint.summary ? html`<p class="ov-note">When it paused, ${ago(p.checkpoint.at)}: ${p.checkpoint.summary}</p>` : ''}
        <div class="row">${actions}</div></header>
      <section class="ov-map" aria-label="Where things stand"><a class="ov-maplink" href="${link({ view: 'map', sel: '', sub: '' })}">Open the map <kbd>1</kbd></a>${stripMap(store)}${roundsHTML(store, e)}</section>
      <div class="ov-cols"><div class="ov-col">
        <section class="ov-sec"><h2>Waiting for you<span>${items.length}</span></h2>${waiting}</section>
        ${anyWatch ? html`<section class="ov-sec"><h2>Experiments that need watching</h2>${watch}</section>` : ''}
        <section class="ov-sec ov-arg"><h2>The argument so far<a class="ov-more" href="${link({ view: 'map', sel: '', sub: '' })}">see it as the map</a></h2>${argumentHTML(store)}</section>
        <section class="ov-sec"><h2>Instructions for the agents</h2>${requestsHTML(store, e)}</section>
      </div><div class="ov-col">
        <section class="ov-sec"><h2>Activity</h2>${feedHTML(store)}</section>
        <section class="ov-sec"><h2>Spend</h2>${spendHTML(store)}</section>
        ${problems.length ? html`<section class="ov-sec"><h2>Problems in the project's files<span>${problems.length}</span></h2><ul class="problems">${problems.slice(0, 12).map((x) => html`<li>${typeof x === 'string' ? x : x.message || JSON.stringify(x)}</li>`)}</ul></section>` : ''}
        <section class="ov-sec"><h2>Write it up <span>optional</span></h2><p class="ov-paper">When the wiki says what you want to say, it can be written up as a paper with one of the bundled templates. <a class="linkish" href="${link({ view: 'paper', sel: '', sub: '' })}">Open the paper step</a></p></section>
      </div></div></div></div>`);
    afterRender(root);
  },
};
const cap1 = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '');
