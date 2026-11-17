/* The overview: where the project stands, in one page. The checkpoint sentence, the strip
   map, the rounds, what waits, the argument so far, activity, instructions, spend. */

import { app, defaults, link } from './core.js';
import { MONTH_NAMES, ago, checkpointDid, clock, cutoffParts, cutoffWords, dayOf, duration, hm, html, listOf, money, parseStamp, plain, plural, raw, setHTML, whyNot } from './util.js';
import { CHECKPOINT_AFTER, KIND, PHASES, ROLE, STAGES, TONE, agentName, glyph, icon, lapsed, pageGlyph, phaseName, roleName, shownState } from './vocab.js';
import { afterRender, label, pageName, pageTip, statusParts } from './page.js';
import { DEMO_COST, UNVERIFIED, continueControl, continueNotes, crashText, engineView, foldRuns, forgetButton, nextStep, problemsHTML, researchReasons, runSentence, waitingSentence, whoOf } from './shell.js';
import { acceptTip, checkMark, summaryOf, whatOf } from './inbox.js';
import { decisionLink, openDecisions } from './decisions.js';

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
/** The small one for the library: a dot per stage and the number of pages. What a number
    counts is said when it is pointed at: "14 sources", "6 questions, 1 of them waiting". */
export function stripMapSmall(counts, waiting) {
  return html`<div class="smap smap-s">${STAGES.map((k) => {
    const n = (counts || {})[k] || 0, w = (waiting || {})[k] || 0;
    const says = k === 'idea' ? 'The idea' : (n ? plural(n, KIND[k].one.toLowerCase(), KIND[k].many.toLowerCase()) : 'No ' + KIND[k].many.toLowerCase() + ' yet') + (w ? ', ' + w + ' of them waiting for you' : '');
    return html`<div class="smap-st${n ? '' : ' none'}${w ? ' wait' : ''}" data-tip="${says}"><div class="smap-dot"></div><div class="smap-n">${k === 'idea' || !n ? '' : n}</div></div>`;
  })}</div>`;
}

/** The rounds as lines of phases: what ran, where the project stands, what comes. A phase
    that ran twice in a row (a run with nothing approved, then the run of what was) is one
    mark that says so, and a phase that comes round again in the open round says "again":
    neither reads as a stutter. */
export function roundsLine(store, e) {
  e = e || engineView(store);
  const phases = ((store.project.state || {}).phases || []);
  if (!phases.length && !e.next) return '';
  const rounds = {};
  phases.forEach((ph) => (rounds[ph.round] = rounds[ph.round] || []).push(ph));
  const current = e.round || Math.max(1, ...Object.keys(rounds).map(Number));
  const done = (rounds[current] || []).map((ph) => ph.phase);
  /* what is still to come in the open round, in the engine's order */
  const coming = [];
  let retry = false;
  if (e.next && e.mode !== 'running') {
    let i = PHASES.indexOf(e.next);
    const last = (rounds[current] || []).slice(-1)[0];
    retry = !!last && last.phase === e.next && last.status !== 'completed';
    if (i >= 0) for (let k = retry ? i + 1 : i; k < PHASES.length; k++) { if (PHASES[k] === 'intake' || (done.indexOf(PHASES[k]) >= 0 && k !== i)) continue; coming.push(PHASES[k]); }
  }
  /* what is approved runs and is analysed before the phase that is next: drawn in that order */
  const step = e.working || e.unverified ? null : nextStep(store, e);
  /* (a phase that is resumed is drawn as the record it has; it comes again after what runs first) */
  if (step && step.back) coming.unshift(...step.steps, ...(retry && step.back === e.next ? [step.back] : []));
  const here = e.mode !== 'running' && e.mode !== 'idle';
  /* records of one phase that follow one another, all but the last completed, are one mark */
  const marks = (list) => list.reduce((out, ph) => {
    const before = out[out.length - 1];
    if (before && before.phase === ph.phase && before.status === 'completed') out[out.length - 1] = Object.assign({}, ph, { times: before.times + 1, produced: (before.produced || []).concat(ph.produced || []), seconds: before.seconds == null && ph.seconds == null ? null : (before.seconds || 0) + (ph.seconds || 0), requests: (before.requests || []).concat(ph.requests || []) });
    else out.push(Object.assign({ times: 1 }, ph));
    return out;
  }, []);
  /* a phase that ran to its end in the open round, or is to come earlier in it, and comes round
     once more (not one that is resumed: its mark says "interrupted", and it has not ended) */
  const ended = (rounds[current] || []).filter((ph) => ph.status === 'completed').map((ph) => ph.phase);
  const next = (ph, i) => html`<span class="trk-p s-next"><i></i>${phaseName(ph)}${ended.indexOf(ph) >= 0 || coming.indexOf(ph) < i ? ' again' : ''}</span>`;
  return html`<div class="trks">${Object.keys(rounds).sort((a, b) => a - b).map((r) => {
    const list = marks(rounds[r]), open = +r === current;
    return html`<div class="trk"><span class="trk-r">Round ${r}</span>${list.map((ph, i) => {
      const isLast = open && i === list.length - 1;
      const made = (ph.produced || []).length;
      const told = (ph.requests || []).length;
      const tip = (ph.status === 'completed' ? '' : ph.status + ' · ') + (ph.times > 1 ? 'ran ' + ph.times + ' times · ' : '') + (made ? plural(made, 'page') + ' written · ' : '') + (ph.seconds != null ? duration(ph.seconds) : '') + (ph.attempts > 1 ? ' · ' + ph.attempts + ' attempts' : '') + (told ? ' · given ' + plural(told, 'instruction') : '') + (ph.error ? ' · ' + plain(ph.error) : '');
      return html`<span class="trk-p s-${ph.status}" data-tip="${tip || ph.status}"><i></i>${phaseName(ph.phase)}${ph.times === 2 ? ' (twice)' : ph.times > 2 ? ' (' + ph.times + ' times)' : ''}${ph.status === 'failed' ? ' (failed)' : ph.status === 'interrupted' ? ' (interrupted)' : ph.status === 'running' ? ' (running)' : ''}</span>
        ${ph.status === 'completed' && CHECKPOINT_AFTER[ph.phase] ? html`<span class="trk-c${isLast && here && e.mode === 'checkpoint' ? ' open' : ''}" data-tip="Checkpoint">${isLast && here && e.mode === 'checkpoint' ? 'you are here' : ''}</span>` : isLast && here ? html`<span class="trk-c open">you are here</span>` : ''}`;
    })}${open ? coming.map(next) : ''}</div>`;
  })}${phases.length ? '' : html`<div class="trk"><span class="trk-r">Round 1</span>${coming.map(next)}</div>`}</div>`;
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
          <details class="ev-steps" data-keep="run:${ev.key}"><summary>${plural(ev.parts.length, 'step')}</summary><ol>${ev.parts.map((p) => html`<li><time>${hm(p.ts)}</time><span>${plain(p.summary)}</span></li>`)}</ol></details></span></li>`;
    }
    /* a phase that starts says which of the person's instructions it was given */
    const told = ev.type === 'phase.started' ? ((ev.data || {}).requests || []).map((id) => store.requests.find((r) => r.id === id)).filter(Boolean) : [];
    const word = ev.type === 'phase.started' && typeof (ev.data || {}).instruction === 'string' ? ev.data.instruction : '';
    const given = told.length || word ? html` <span class="ev-given quiet" data-tip="${told.map((r) => '“' + r.text + '”').concat(word ? ['“' + word + '” (given with this run)'] : []).join(' · ')}">Given ${plural(told.length + (word ? 1 : 0), 'instruction')} of yours.</span>` : '';
    /* a checkpoint is the engine pausing, whichever agent worked last; it is the line to find again */
    const pause = ev.type === 'checkpoint';
    const who = pause ? 'engine' : whoOf(ev), you = who === 'you';
    const ref = pause ? null : (ev.refs || []).find((r) => store.byPath.has(r));
    return html`${head}<li class="ev ev-${String(ev.type).replace(/[^a-z]/g, '-')}${you ? ' ev-you' : who === 'engine' && !pause ? ' ev-engine' : ''}"><time>${hm(ev.ts)}</time><span class="ev-who">${who}</span>
      ${/^decision\./.test(ev.type) ? html`<a href="${decisionLink((ev.data || {}).decision)}">${plain(ev.summary)}</a>` /* (a decision is answered, and found again, in the inbox) */
        : ref ? html`<a href="${link({ view: 'wiki', sel: ref, sub: '' })}" data-sel="${ref}">${plain(ev.summary)}</a>` : html`<span>${plain(ev.summary)}${given}</span>`}</li>`;
  });
  const more = store.activity.length > app.ui.feedLimit || store.moreActivity;
  return html`<ol class="feed">${rows}</ol>${more ? html`<button class="btn btn-quiet btn-s feed-more" data-act="feed-more">Show earlier activity</button>` : ''}`;
}

/** Whom an instruction can be for: the server's list (`request_for` of /api/defaults). Until
    it is known the one choice is everyone, which is what the server takes an instruction
    without an addressee to mean. */
function addressees() {
  const d = defaults();
  const list = d && Array.isArray(d.request_for) ? d.request_for.filter((r) => typeof r === 'string' && r) : [];
  return list.length ? list : ['all'];
}
/** When an instruction for `role` reaches an agent: the engine's rule, said before it is sent. */
function whenGiven(role, e) {
  const phase = (ROLE[role] || {}).phase;
  if (!phase) return 'Every phase that runs is given it, until the analysis that ends this round.';
  if (phase === 'research') return e.next === 'research' && !e.working ? 'Given to the scout in the research phase, which is next.' : 'Kept for the scout until a research phase runs. Research does not come round by itself: choose “Search the literature again” beside Continue.';
  return 'Given to ' + roleName(role) + ' in the next ' + (phase === 'analyze' ? 'analysis' : phaseName(phase) + ' phase') + '.';
}
/** The phases that were given a request so far, by round: "research and hypotheses (round 2)". */
function givenTo(request) {
  const had = Array.isArray(request.handled_in) ? request.handled_in : request.handled_in && typeof request.handled_in === 'object' ? [request.handled_in] : [];
  const rounds = [];
  had.forEach((h) => {
    if (!h || !h.phase) return;
    let group = rounds.find((g) => g.round === h.round);
    if (!group) { group = { round: h.round, phases: [] }; rounds.push(group); }
    if (group.phases.indexOf(h.phase) < 0) group.phases.push(h.phase);
  });
  return rounds.map((g) => listOf(g.phases.map(phaseName)) + (g.round ? ' (round ' + g.round + ')' : '')).join(', ');
}

function requestsHTML(store, e) {
  const sel = app.route.sel && store.byPath.get(app.route.sel);
  const rq = app.ui.request;
  let roles = addressees();
  /* a choice kept from before a reload stands until the server's list is known */
  if (roles.indexOf(rq.for) < 0) { if (defaults()) rq.for = roles[0]; else roles = roles.concat([rq.for]); }
  const list = store.requests.slice().reverse();
  return html`<form class="rq-form" data-form="request"><label class="sr" for="rq-text">Instruction</label>
    <textarea id="rq-text" name="text" data-input="request" rows="2" placeholder="Tell the agents what to do next, in your own words. For example: compare against RMSprop as well.">${rq.text}</textarea>
    <div class="rq-bar"><label class="rq-for"><span>For</span><select id="rq-for" name="for" data-input="request-for">${roles.map((r) => html`<option value="${r}"${r === rq.for ? raw(' selected') : ''}>${(ROLE[r] || { pick: 'The ' + r }).pick}</option>`)}</select></label>
      ${sel ? html`<label class="tick"><input type="checkbox" id="rq-about" data-input="request-about"${rq.about ? raw(' checked') : ''}> about ${label(sel)}</label>` : ''}</div>
    <div class="rq-send"><p class="rq-when quiet">${whenGiven(rq.for, e)} You can withdraw it until it is done.</p>
      <button class="btn" type="submit"${rq.sending ? raw(' disabled') : ''}>Add instruction</button></div>
    ${rq.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span><b>Not added.</b> ${rq.error} Your text is still in the box.</span></p>` : ''}</form>
    <ul class="rqs">${list.map((r) => {
      const target = r.target && store.byPath.get(r.target), had = givenTo(r), open = r.state === 'open';
      return html`<li class="rq rq-${open ? 'open' : 'done'}" data-rq="${r.id}"><span class="rq-t">${r.text}</span>
        <span class="rq-m"><span class="rq-for-who">${open ? 'open' : 'done'} · for ${roleName(r.for || 'all')}</span> · ${ago(r.created)}${target ? html` · about <a href="${link({ view: 'wiki', sel: target.path, sub: '' })}" data-sel="${target.path}">${label(target)}</a>` : ''}
          <span class="rq-had">${had ? (open ? 'given so far to ' : 'given to ') + had : open ? 'no phase has been given it yet' : ''}</span></span>
        ${open ? html`<button class="btn btn-quiet btn-s rq-x" data-act="rq-del" data-ref="${r.id}" data-tip="Take the instruction back: no further phase is given it">Withdraw</button>` : ''}</li>`;
    })}</ul>`;
}

function limitsForm(limits) {
  const ui = app.ui.limits || (app.ui.limits = { model_usd: String(limits.model_usd), experiment_usd: String(limits.experiment_usd), minutes: String(Math.round((limits.experiment_seconds || 0) / 6) / 10), error: null, saving: false });
  const field = (name, label, unit) => html`<label><span>${label}</span><input id="lim-${name}" data-input="limits" data-name="${name}" value="${ui[name]}" inputmode="decimal" autocomplete="off" size="7"><span class="quiet">${unit}</span></label>`;
  return html`<form class="limits-form" data-form="limits">${field('model_usd', 'Models', 'dollars in all')}${field('experiment_usd', 'Experiments', 'dollars in all')}${field('minutes', 'One run', 'minutes at most')}
    ${ui.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span>${ui.error}</span></p>` : ''}
    <button class="btn btn-s" type="submit"${ui.saving ? raw(' disabled') : ''}>Save the limits</button></form>`;
}

/** The research cutoff of the project: said, and changed, as a year and a month. A value the
    server could not read is shown as it is written; every save of the settings is refused
    until it is corrected, so this is where it is put right. */
function cutoffHTML(store) {
  const value = (store.project.settings || {}).research_cutoff;
  const parts = cutoffParts(value), ui = app.ui.cutoffEdit;
  const says = !parts ? 'No cutoff: literature of any date is in scope.'
    : parts.raw != null ? 'The cutoff cannot be read as it is written (“' + parts.raw + '”). Research stops, and the settings cannot be saved, until it is corrected here.'
      : 'Only literature published before ' + cutoffWords(parts) + ' is in scope.';
  const month = ui ? +ui.month || 0 : 0;
  return html`<details class="spend-more cutoff${parts && parts.raw != null ? ' cutoff-bad' : ''}"${ui ? raw(' open') : ''}><summary data-act="cutoff-open">Research cutoff <span class="quiet">${!parts ? 'none' : parts.raw != null ? 'cannot be read' : 'before ' + cutoffWords(parts)}</span></summary>
    <p class="cutoff-says">${parts && parts.raw != null ? icon('warn') : ''}${says}</p>
    ${ui ? html`<form class="cutoff-form" data-form="cutoff"><span class="quiet">Only literature before</span>
      <label class="sr" for="cut-month">Month of the cutoff</label><select id="cut-month" data-input="cutoff" data-name="month"><option value="0"${month ? '' : raw(' selected')}>any month</option>${MONTH_NAMES.map((name, i) => html`<option value="${i + 1}"${month === i + 1 ? raw(' selected') : ''}>${name}</option>`)}</select>
      <label class="sr" for="cut-year">Year of the cutoff</label><input id="cut-year" data-input="cutoff" data-name="year" value="${ui.year}" inputmode="numeric" maxlength="4" size="5" placeholder="year" autocomplete="off">
      <button class="btn btn-s" type="submit"${ui.saving ? raw(' disabled') : ''}>Save the cutoff</button>
      ${parts ? html`<button class="btn btn-quiet btn-s" type="button" data-act="cutoff-none"${ui.saving ? raw(' disabled') : ''}>No cutoff</button>` : ''}
      ${ui.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span>${ui.error}</span></p>` : ''}</form>` : ''}</details>`;
}

function spendHTML(store) {
  const p = store.project, limits = (p.settings || {}).limits || {}, spent = (p.state || {}).spent || {};
  const frac = (a, b) => (b > 0 ? Math.max(0, Math.min(1, a / b)) : 0);
  const phases = ((p.state || {}).phases || []).filter((ph) => ph.seconds != null);
  const agentSeconds = phases.reduce((a, ph) => a + (ph.seconds || 0), 0);
  return html`${p.demonstration ? html`<p class="quiet spend-demo" data-demonstration>${DEMO_COST}</p>` : ''}<div class="spend-big">
    <div><b>${money(spent.model_usd || 0)}</b><span>of ${money(limits.model_usd)} for models${p.demonstration ? ', in the story' : ''}</span><progress class="meter-p${frac(spent.model_usd || 0, limits.model_usd) > 0.85 ? ' hot' : ''}" max="1" value="${frac(spent.model_usd || 0, limits.model_usd)}" aria-label="Model spend, of the limit"></progress></div>
    <div><b>${money(spent.experiment_usd || 0)}</b><span>of ${money(limits.experiment_usd)} for experiments</span><progress class="meter-p" max="1" value="${frac(spent.experiment_usd || 0, limits.experiment_usd)}" aria-label="Experiment spend, of the limit"></progress></div>
    <div><b>${duration(agentSeconds) || '0 s'}</b><span>of agent time · ${duration(spent.experiment_seconds || 0) || '0 s'} of runs</span></div></div>
    ${phases.length ? html`<details class="spend-more"><summary>Time by phase</summary><table class="metrics"><tbody>${phases.map((ph) => html`<tr><td>Round ${ph.round} · ${phaseName(ph.phase)}${ph.status !== 'completed' ? ' (' + ph.status + ')' : ''}</td><td class="n">${duration(ph.seconds)}</td><td class="n">${plural((ph.produced || []).length, 'page')}</td></tr>`)}</tbody></table></details>` : ''}
    <details class="spend-more"${app.ui.limitsOpen ? raw(' open') : ''}><summary data-act="limits-open">Change the limits</summary>${limitsForm(limits)}</details>
    ${cutoffHTML(store)}
    <p class="quiet spend-note">The engine stops at a checkpoint when a limit is reached, and says so.</p>`;
}

export const overviewView = {
  render(root) {
    const store = app.store, p = store.project, e = engineView(store), items = store.pending(), asked = openDecisions(store);
    const running = store.experiments.filter((x) => (x.state === 'running' && !x.interrupted) || x.state === 'queued');
    /* the page says "running" and nothing is running it: its worker is gone */
    const interrupted = store.experiments.filter((x) => x.state === 'running' && x.interrupted && x.review !== 'rejected');
    const broken = store.experiments.filter((x) => x.state === 'failed' && x.review !== 'rejected');
    const waitingApproval = store.experiments.filter((x) => x.review === 'accepted' && (x.state === 'planned' || !x.state));
    /* approved, and no longer what was approved: it will not run until it is approved again */
    const changed = store.experiments.filter((x) => lapsed(x));
    const lastAct = store.activity.slice().reverse().find((ev) => ev.agent);
    const headline = e.unverified ? 'This server cannot verify the project\'s worker.'
      : e.unknown && !e.crashed ? 'Where this project stands is not known.'
      : e.mode === 'running' ? 'Round ' + (e.round || 1) + ': the ' + (lastAct ? lastAct.agent : 'engine') + ' is at work on ' + phaseName(e.phase) + '.'
      : e.mode === 'experiment' ? 'An experiment is running.' : e.mode === 'writing' ? 'The paper is being written.'
      : e.mode === 'failed' ? cap1(phaseName(e.next || e.phase)) + ' failed.' : e.died || e.crashed ? 'The agents stopped unexpectedly.'
      : e.mode === 'stopped' ? 'You stopped the agents during ' + phaseName(e.next || e.phase) + '.' : e.fresh ? 'Research has not started.'
      : e.budget ? 'Round ' + (e.round || 1) + ' stopped at a budget.' : 'Round ' + (e.round || 1) + ' is paused at a checkpoint.';
    const sub = e.unverified ? plain(e.note) || UNVERIFIED
      : e.unknown && !e.crashed ? plain(e.stateNote) || 'Nothing on file says which round and phase it is in. Choose the phase to go on with.'
      : e.working ? 'You can keep reading and deciding while it works.'
      : e.mode === 'failed' ? plain(e.error) : e.died ? plain(e.error) : e.crashed ? crashText(store, e.exit)
      : e.fresh ? 'Nothing has been spent. The idea' + ((p.attachments || []).length ? ' and ' + plural(p.attachments.length, 'file') : '') + ' are in place.'
      : e.budget ? e.budget : waitingSentence(store);
    const idle = !e.working && !e.unverified;
    const actions = e.unverified ? html`${forgetButton(true)}<span class="quiet ov-unv">It is not this server's to stop, so there is no Stop. Nothing can be started here until it has ended or is forgotten.</span>`
      : !idle ? html`<a class="btn btn-l" href="${link({ view: 'map', sel: '', sub: '' })}">Watch on the map</a><button class="btn btn-l" data-act="stop"${e.stopping ? raw(' disabled') : ''}>${icon('stop')}${e.stopping ? 'Stopping…' : 'Stop'}</button>`
      : html`${items.length ? html`<a class="btn btn-l" href="${link({ view: 'inbox', sel: '', sub: '' })}">Review ${items.length}</a>` : ''}
        ${continueControl(store, e, 'overview', true)}
        ${e.mode === 'failed' || e.died || e.crashed ? html`<a class="btn btn-l" href="#/setup">Open Setup</a>` : ''}`;
    const waiting = items.length ? items.map((it) => { const sum = summaryOf(it); return html`<div class="ov-item" data-ref="${it.ref}"><a class="ov-main" href="${link({ view: 'inbox', sel: it.page.path, sub: '' })}">${glyph(it.kind, it.type === 'page' ? 'proposed' : 'accepted', shownState(it.page), 18)}
        <span class="in-name">${pageName(it.page)}</span>${sum ? html`<span class="in-sum">${sum}</span>` : ''}<span class="in-what"><span class="in-kind">${whatOf(it)}</span>${agentName(it.author)} · ${ago(it.when)}</span>${checkMark(it)}</a>
        <span class="in-quick on"><button class="ibtn ibtn-yes" data-act="accept" data-ref="${it.ref}" data-tip="${acceptTip(it)}" aria-label="${acceptTip(it)}">${icon('check')}</button><button class="ibtn ibtn-no" data-act="reject-ask" data-ref="${it.ref}" data-key="${it.page.path}" data-view="inbox" data-tip="Reject, with a reason" aria-label="Reject">${icon('x')}</button></span></div>`; })
      : html`<p class="quiet ov-none">${asked.length ? 'No proposal is waiting for you.' : 'Nothing is waiting for you.'}</p>`;
    /* what the agents asked the person to choose: each leads to the inbox, where it is answered */
    const decide = asked.map((d) => html`<a class="ov-run ov-decide" href="${decisionLink(d.id)}" data-decision="${d.id}">${icon('dot')}<span class="in-name"><span class="pn-t">${d.question}</span></span><span class="in-what"><span class="in-kind">Decision</span>${(d.raised_by || {}).role || 'agent'} · ${ago(d.created)} · answer it in the inbox</span></a>`);
    const watch = html`${running.map((x) => { const run = x.last_run && x.last_run.status === 'running' ? x.last_run : null; const est = (x.meta || {}).est_minutes; return html`<a class="ov-run" href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${pageGlyph(x, 18)}<span class="in-name">${pageName(x)}</span>
        <span class="in-what">${run ? html`run ${run.number} · <span data-since="${run.started}">${clock((Date.now() - (parseStamp(run.started) || Date.now())) / 1000)}</span>${est ? ' of about ' + est + ' min' : ''}` : 'queued'}</span>
        ${run && est ? html`<progress class="meter-p" max="1" value="0" data-since="${run.started}" data-total="${est * 60}" aria-label="Elapsed, of the estimate"></progress>` : ''}<code class="ov-tail" data-tail="${x.id}">${store.tails[x.id] || ''}</code></a>`; })}
      ${broken.map((x) => html`<a class="ov-run ov-broken" href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${pageGlyph(x, 18)}<span class="in-name">${pageName(x)}</span><span class="in-what">failed${x.runs > 1 ? ' ' + x.runs + ' times' : ''} · ${(x.meta || {}).state_reason || (x.last_run || {}).error || 'needs a decision from you'}</span></a>`)}
      ${interrupted.map((x) => html`<a class="ov-run ov-broken" href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${pageGlyph(x, 18)}<span class="in-name">${pageName(x)}</span><span class="in-what">interrupted · the worker that ran it is gone · run it again, or cancel it</span></a>`)}
      ${changed.map((x) => html`<a class="ov-run ov-broken" data-will-not-run="${x.id}" href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${pageGlyph(x, 18)}<span class="in-name">${pageName(x)}</span><span class="in-what">needs approval again · ${whyNot(x.will_not_run)}</span></a>`)}
      ${waitingApproval.map((x) => html`<a class="ov-run" href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${pageGlyph(x, 18)}<span class="in-name">${pageName(x)}</span><span class="in-what">accepted · runs only after you approve it</span></a>`)}`;
    const anyWatch = running.length + broken.length + interrupted.length + waitingApproval.length + changed.length;
    /* What the checkpoint said, as it said it then. (What the analyst recommended is among it;
       whether that still stands is the checkpoint's field, shown beside the button.) The
       count of instructions that wait for the scout is said live beside the button instead. */
    const why = idle ? researchReasons(store, e) : [];
    let did = e.mode === 'checkpoint' && !e.budget && p.checkpoint ? checkpointDid(p.checkpoint) : '';
    if (why.some((w) => w.kind === 'requests')) did = did.replace(/\d+ requests? for the scout waits? for the research phase[^.]*\.\s*/, '');
    /* A pause for want of a model: the engine's sentence about it stands beside the button for
       as long as it is true (the forecast carries it), and is said once. In the account of what
       the phases did it would be said twice, and go on being said once a model can be used. */
    const need = idle ? nextStep(store, e).need : null;
    if (need) did = did.replace(need.message, '');
    did = did.replace(/\bThe [a-z ]{3,40} phase needs a model, and [^]*?show what is configured\)\.\s*/, '').trim();
    const beside = idle ? continueNotes(store, e, !!did) : '';
    setHTML(root, html`<div class="overview"><div class="ov-wrap">
      <header class="ov-head"><div class="ov-state">${e.working ? html`<span class="live-dot"></span>` : ''}${p.title}${e.unknown ? '' : ' · ' + (e.round ? 'round ' + e.round : 'new')}</div><h1>${headline}</h1><p class="ov-sub">${sub}</p>
        ${did ? html`<p class="ov-note">${cap1(phaseName(p.checkpoint.phase)) || 'The last phase'} finished ${ago(p.checkpoint.at)}: ${did}</p>` : ''}
        <div class="row ov-actions">${actions}</div>${String(beside) ? html`<div class="ov-next">${beside}</div>` : ''}</header>
      ${problemsHTML(store)}
      <section class="ov-map" aria-label="Where things stand"><a class="ov-maplink" href="${link({ view: 'map', sel: '', sub: '' })}">Open the map <kbd>1</kbd></a>${stripMap(store)}${roundsLine(store, e)}</section>
      <div class="ov-cols"><div class="ov-col">
        <section class="ov-sec"><h2>Waiting for you<span>${items.length + asked.length}</span></h2>${decide}${waiting}</section>
        ${anyWatch ? html`<section class="ov-sec"><h2>Experiments that need watching</h2>${watch}</section>` : ''}
        <section class="ov-sec ov-arg"><h2>The argument so far<a class="ov-more" href="${link({ view: 'map', sel: '', sub: '' })}">see it as the map</a></h2>${argumentHTML(store)}</section>
        <section class="ov-sec"><h2>Instructions for the agents</h2>${requestsHTML(store, e)}</section>
      </div><div class="ov-col">
        <section class="ov-sec"><h2>Activity</h2>${feedHTML(store)}</section>
        <section class="ov-sec"><h2>Spend</h2>${spendHTML(store)}</section>
        <section class="ov-sec"><h2>Write it up <span>optional</span></h2><p class="ov-paper">When the wiki says what you want to say, it can be written up as a paper with one of the bundled templates. <a class="linkish" href="${link({ view: 'paper', sel: '', sub: '' })}">Open the paper step</a></p></section>
      </div></div></div></div>`);
    afterRender(root);
  },
};
const cap1 = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '');
