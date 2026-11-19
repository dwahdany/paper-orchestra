/* Autopilot: the one switch with its caps, and "While you were away". What the person's
   standing rule decided is the server's account (`autopilot` of the project view: the mode,
   the digest of the latest unattended stretch, what waits all the same); nothing here
   decides or counts for itself. */

import { app, link } from './core.js';
import { ago, duration, html, money, plain, plural, raw } from './util.js';
import { KIND, icon } from './vocab.js';

/** `settings.autonomy` of the project, as the server shows it (always there). */
export const autonomyOf = (store) => (((store || {}).project || {}).settings || {}).autonomy || null;
export const autopilotOn = (store) => { const a = autonomyOf(store); return !!a && a.mode === 'autopilot'; };
/** `autopilot` of the project view: `{mode, refused, since, digest, held}`. */
export const autopilotView = (store) => ((store || {}).project || {}).autopilot || {};
/** The experiments autopilot does not approve, each `{id, reason}`: they wait for the person. */
export const heldRuns = (store) => (autopilotOn(store) ? autopilotView(store).held || [] : []);
/** Why autopilot leaves an experiment to the person, or nothing. */
export const heldReason = (store, id) => { const h = heldRuns(store).find((x) => x.id === id); return h ? h.reason : ''; };

/** `12 hours`, `1 hour`, `30 minutes`. */
const hoursWords = (h) => (h >= 1 ? plural(+(+h).toFixed(2), 'hour') : plural(Math.round(h * 60), 'minute'));
/** What one stretch is held to: "at most 3 rounds, $200.00 on models or 12 hours". */
export const capsWords = (a) => 'at most ' + plural(a.rounds, 'round') + ', ' + money(a.model_usd) + ' on models or ' + hoursWords(a.hours) + ', whichever comes first';
const networkWords = (a) => (a.network ? 'Runs it approves get the network access their plans ask for.' : 'Runs it approves get no network access.');

/** The sentence under the headline while autopilot is on. */
export function autopilotSub(store, e) {
  const d = autopilotView(store).digest;
  if (!autopilotOn(store)) return '';
  if (e.working) return 'Autopilot is on: the agents go on by themselves until a cap is reached. Stop ends the stretch.';
  if (d && d.ended) return 'Autopilot stopped ' + ago(d.ended) + ': ' + plain(d.why || 'its worker ended without saying why') + '. Continue starts a new stretch.';
  if (d && !d.ended) return 'The last stretch of autopilot ended without saying why: its worker is gone. Continue starts a new stretch.';
  return 'Autopilot is on: Continue starts a stretch that goes on by itself until a cap is reached.';
}

/** "While you were away": the server's digest of the latest stretch, line by line. */
export function digestHTML(store, e) {
  const d = autopilotView(store).digest;
  if (!d) return '';
  const live = !d.ended && e.working, caps = d.caps || {};
  const kinds = Object.keys(d.accepted_by_kind || {}).sort().map((k) => plural(d.accepted_by_kind[k], (KIND[k] || { one: k }).one.toLowerCase(), (KIND[k] || { many: k + 's' }).many.toLowerCase()));
  const runs = (d.runs || []).map((r) => { const x = store.expById.get(r.id); const said = r.id + ' ' + (r.outcome || (live ? 'is running' : 'has not run')); return x ? html`<a href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${said}</a>` : said; });
  const held = d.held || [];
  /* one reason is said once, for all the runs that share it */
  const groups = [];
  held.forEach((h) => { const g = groups.find((x) => x.reason === h.reason); if (g) g.ids.push(h.id); else groups.push({ reason: h.reason, ids: [h.id] }); });
  return html`<div class="auto-digest" data-digest="${live ? 'running' : 'ended'}"><h3>${live ? 'This stretch so far' : 'While you were away'}</h3>
    <p class="auto-why">${live ? 'Started ' + ago(d.started) + '.' : d.ended ? html`Stopped ${ago(d.ended)}: ${plain(d.said || d.why)}` : 'The stretch ended without saying why: its worker is gone.'}</p>
    <ul class="auto-list">
      <li data-auto="accepted"><b>${plural((d.accepted || []).length, 'page')} accepted</b>${kinds.length ? html` <span class="quiet">${kinds.join(', ')}</span>` : ''}</li>
      <li data-auto="edits"><b>${plural((d.edits || []).length, 'edit')} applied</b></li>
      <li data-auto="decisions"><b>${plural((d.decisions || []).length, 'decision')}</b> went on at the agents' default${(d.decisions || []).length ? html` · <a href="${link({ view: 'inbox', sel: '', sub: '' })}">answer in the inbox</a>` : ''}</li>
      <li data-auto="runs"><b>${plural((d.runs || []).length, 'run')} approved</b>${runs.length ? html`: ${runs.map((r, i) => html`${i ? ', ' : ''}${r}`)}` : ''}</li>
      <li data-auto="spent"><b>${money(d.spent_usd)}</b> of ${money(caps.model_usd)} on the agents' models, ${duration(d.seconds) || '0 s'} of ${hoursWords(caps.hours)}, ${d.rounds} of ${plural(caps.rounds, 'round')}</li>
      <li class="quiet" data-auto="uncounted">${plain(d.uncounted)}; ${d.runs_with_network ? plural(d.runs_with_network, 'run') + ' had network access and could make such calls' : 'no run of this stretch had network access'}.</li></ul>
    ${held.length ? html`<p class="note note-warn auto-held">${icon('warn')}<span><b>Waiting for you:</b> ${groups.map((g, i) => html`${i ? '; ' : ''}${g.ids.map((id, k) => { const x = store.expById.get(id); return html`${k ? ', ' : ''}${x ? html`<a href="${link({ view: 'experiments', sel: x.path, sub: '' })}">${id}</a>` : id}`; })}: ${plain(g.reason)}`)}.</span></p>` : ''}</div>`;
}

/** The switch, the caps and the digest, as one section of the overview. */
export function autopilotHTML(store, e, waiting) {
  const a = autonomyOf(store);
  if (!a) return '';
  const view = autopilotView(store), ui = app.ui.autopilotEdit, on = a.mode === 'autopilot';
  const field = (name, text, unit) => html`<label><span>${text}</span><input id="auto-${name}" data-input="autopilot" data-name="${name}" value="${ui[name]}" inputmode="decimal" autocomplete="off" size="7"><span class="quiet">${unit}</span></label>`;
  return html`<section class="ov-sec ov-auto" data-autopilot="${on ? 'on' : 'off'}"><h2>Autopilot <span>${on ? 'on' : 'off'}</span></h2>
    <p class="auto-now">${on ? html`<b>On.</b> Pages are accepted, edits applied, runs approved and checkpoints passed without you. One stretch is held to ${capsWords(a)}. ${networkWords(a)}`
      : html`<b>Off.</b> Every proposal, edit, run and checkpoint waits for you.`}</p>
    ${view.refused && !on ? html`<p class="note note-warn auto-refused">${icon('warn')}<span>${plain(view.refused)}</span></p>` : ''}
    ${digestHTML(store, e)}
    <details class="spend-more auto-more"${ui ? raw(' open') : ''}><summary data-act="autopilot-open">${on ? 'Change the caps, or switch it off' : 'Switch it on'}</summary>
    ${ui ? html`<form class="limits-form auto-form" data-form="autopilot">
      <label class="tick auto-on"><input type="checkbox" id="auto-on" data-input="autopilot-tick" data-name="on"${ui.on ? raw(' checked') : ''}> <span>Go on without me, up to these caps</span></label>
      ${field('rounds', 'Rounds', 'at most, in one stretch')}${field('model_usd', 'Models', 'dollars at most')}${field('hours', 'Time', 'hours at most')}
      <label class="tick"><input type="checkbox" id="auto-network" data-input="autopilot-tick" data-name="network"${ui.network ? raw(' checked') : ''}> <span>Runs it approves may reach the network <span class="quiet">(where their plan asks for it; their code can then call model APIs that nothing here counts)</span></span></label>
      ${ui.on && waiting ? html`<label class="tick auto-settle"><input type="checkbox" id="auto-settle" data-input="autopilot-tick" data-name="settle"${ui.settle ? raw(' checked') : ''}> <span>Also settle what waits now <span class="quiet">(${waiting} to decide: accepted, applied or left at the default, each recorded as autopilot's; without this they stay yours)</span></span></label>` : ''}
      <p class="quiet auto-when">The caps count from your last start. Only you can switch this on or raise a cap: no agent reaches it. ${e.working ? 'A change holds from the next phase on.' : ''}</p>
      ${ui.error ? html`<p class="note note-neg" role="alert">${icon('warn')}<span><b>Not changed.</b> ${ui.error}</span></p>` : ''}
      <button class="btn btn-s" type="submit"${ui.saving ? raw(' disabled') : ''}>Save</button></form>` : ''}</details></section>`;
}
