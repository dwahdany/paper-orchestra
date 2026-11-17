/* Decisions: what an agent asked the person to choose, and what they answered.

   An agent raises a decision when the answer is the researcher's choice and no evidence
   settles it. The open ones are the first thing in the inbox: the question, why it matters,
   the options as buttons a finger can press, the one the agent would take, what the agents do
   meanwhile, and a field for the person's own words. Nothing waits for an answer: the agents
   go on with the default, so a decision is counted with what waits and never blocks Continue.
   What was answered is listed below the proposals, folded, and can be answered again.

   The records are the server's (`GET .../decisions`); options are numbered from 1. */

import { app, link } from './core.js';
import { api, Unreachable } from './api.js';
import { ago, html, plural, raw, tf } from './util.js';
import { icon } from './vocab.js';

/** The decisions of a store, whatever it has loaded (a store without the list has none). */
export const allDecisions = (store) => (store && Array.isArray(store.decisions) ? store.decisions : []);
export const openDecisions = (store) => allDecisions(store).filter((d) => d.state === 'open');
/** Everything that waits for the person: the proposals and the open decisions. */
export const waitingCount = (store) => store.pending().length + openDecisions(store).length;
/** "1 decision waits for your answer.", or nothing. */
export const decisionsSentence = (n) => (n ? plural(n, 'decision') + (n === 1 ? ' waits' : ' wait') + ' for your answer.' : '');

/** What the person answered, in a few words; empty while the decision is open. */
export function answerWords(d) {
  const a = d && d.state !== 'open' ? d.answer : null;
  if (!a) return '';
  if (a.dismissed) return 'left to the agents: their default holds';
  const chosen = a.option ? (d.options || [])[a.option - 1] : null;
  const parts = [];
  if (chosen) parts.push('option ' + a.option + ', “' + chosen.label + '”');
  if (a.text) parts.push(chosen ? '“' + a.text + '”' : a.text);
  return parts.join('; ');
}

/** What is being answered and not yet sent, per decision: the option tapped and the words typed. */
const draftOf = (id) => {
  const all = (app.ui.decide = app.ui.decide || {});
  return (all[id] = all[id] || { option: null, text: '', sending: false, error: null });
};
const find = (id) => allDecisions(app.store).find((d) => d.id === id) || null;

function about(store, d) {
  const page = d.page && store.byPath ? store.byPath.get(d.page) : null;
  if (!page) return '';
  return html` · about <a href="${link({ view: 'inbox', sel: page.path, sub: '' })}">${page.id ? page.id + ' ' : ''}${page.title}</a>`;
}

/** The body of one decision: what was asked, the options, and the ways to answer. */
function card(store, d) {
  const draft = draftOf(d.id), settled = d.state !== 'open', role = (d.raised_by || {}).role || 'an agent';
  const now = settled && d.answer && !d.answer.dismissed ? d.answer.option : null;
  const picked = draft.option ? (d.options || [])[draft.option - 1] : null;
  const options = (d.options || []).map((o, i) => {
    const n = i + 1, on = draft.option === n;
    return html`<button type="button" class="dc-opt${on ? ' on' : ''}" data-act="decide-pick" data-ref="${d.id}" data-i="${n}" aria-pressed="${tf(on)}">
      <span class="dc-n" aria-hidden="true">${n}</span><span class="dc-l">${o.label}${d.recommended === n ? html` <i class="dc-rec">the ${role} would take this</i>` : ''}${now === n ? html` <i class="dc-cur">your answer</i>` : ''}</span>
      ${o.meaning ? html`<span class="dc-m">${o.meaning}</span>` : ''}</button>`;
  });
  return html`${d.context_html ? html`<div class="dc-ctx prose">${raw(d.context_html)}</div>` : ''}
    <div class="dc-opts" role="group" aria-label="Options">${options}</div>
    ${d.recommended && d.reason ? html`<p class="dc-note"><b>Why option ${d.recommended}:</b> ${d.reason}</p>` : ''}
    ${d.default ? html`<p class="dc-note"><b>${settled ? 'The default was' : 'Until you answer, the agents go on with'}:</b> ${d.default}</p>` : ''}
    <label class="sr" for="dc-say-${d.id}">Your answer in your own words</label>
    <textarea id="dc-say-${d.id}" class="dc-say" data-input="decide-text" data-ref="${d.id}" rows="2" placeholder="Or say it in your own words${(d.options || []).length ? ' (also together with an option)' : ''}">${draft.text}</textarea>
    ${draft.error ? html`<p class="dc-err" role="alert">${draft.error}</p>` : ''}
    <div class="dc-go"><button type="button" class="btn btn-primary" data-act="decide-send" data-ref="${d.id}"${draft.sending ? raw(' disabled') : ''}><span>${settled ? 'Change the answer' : 'Answer'}${picked ? html`: <span class="dc-pick">${picked.label}</span>` : ''}</span></button>
      ${d.state === 'withdrawn' ? '' : html`<button type="button" class="btn btn-quiet" data-act="decide-dismiss" data-ref="${d.id}" data-tip="The agents go on with their default, and are told that the choice was left to them"${draft.sending ? raw(' disabled') : ''}>Not for me to decide</button>`}</div>`;
}

/** The open decisions, for the top of the inbox. */
export function decisionsPanel(store) {
  const open = openDecisions(store);
  if (!open.length) return '';
  return html`<section class="in-group dc-panel" aria-label="Decisions the agents ask of you"><h3>Decisions for you<span>${open.length}</span></h3>
    <p class="dc-lead">Choices that are yours and that no evidence settles. The agents do not wait: they go on with the default until you answer.</p>
    ${open.map((d) => html`<article class="dc" data-decision="${d.id}"><h4 class="dc-q">${d.question}</h4>
      <p class="dc-by">asked by the ${(d.raised_by || {}).role || 'agents'}${(d.raised_by || {}).round ? ', round ' + d.raised_by.round : ''} · ${ago(d.created)}${about(store, d)}</p>${card(store, d)}</article>`)}</section>`;
}

/** What was answered or left to the agents, the latest first, each folded. */
export function decidedDecisions(store) {
  const done = allDecisions(store).filter((d) => d.state !== 'open')
    .sort((a, b) => String((b.answer || {}).at).localeCompare(String((a.answer || {}).at)));
  if (!done.length) return '';
  return html`<section class="in-group dc-done" aria-label="Decisions you answered"><h3>Your decisions<span>${done.length}</span></h3>
    ${done.map((d) => html`<details class="dc-old" data-keep="dc-${d.id}" data-decision="${d.id}"><summary><span class="in-dec">${icon(d.state === 'withdrawn' ? 'x' : 'check')}</span>
      <span class="dc-oq">${d.question}</span><span class="dc-oa">${answerWords(d)}</span><span class="in-when">${ago((d.answer || {}).at)}${(d.answers || []).length > 1 ? ' · changed' : ''}${about(store, d)}</span></summary>
      <div class="dc dc-in">${card(store, d)}</div></details>`)}</section>`;
}

/** The decisions that belong to a page, as one line each: for the page's own view. */
export function pageDecisions(store, path) {
  const mine = allDecisions(store).filter((d) => d.page === path);
  if (!mine.length) return '';
  return html`<div class="dc-on-page">${mine.map((d) => html`<p class="dc-ref" data-decision-of="${d.id}">${icon(d.state === 'open' ? 'dot' : d.state === 'withdrawn' ? 'x' : 'check')}<span><b>${d.state === 'open' ? 'Waits for your decision' : d.state === 'withdrawn' ? 'Left to the agents' : 'You decided'}:</b> ${d.question}${d.state === 'open' ? '' : html` <q>${answerWords(d)}</q>`}
    <a href="${link({ view: 'inbox', sel: '', sub: '' })}">${d.state === 'open' ? 'Answer it in the inbox' : 'Change it in the inbox'}</a></span></p>`)}</div>`;
}

const message = (error) => (error instanceof Unreachable ? 'The server is not answering.' : (error && error.message) || 'Something went wrong.');

async function send(id, dismiss, say) {
  const store = app.store, d = find(id), draft = draftOf(id);
  if (!store || !d || draft.sending) return;
  const text = (draft.text || '').trim();
  if (!dismiss && !draft.option && !text) {
    draft.error = 'Choose an option, or say it in your own words.';
    app.render('decide');
    return;
  }
  draft.sending = true; draft.error = null;
  app.render('decide');
  try {
    const answer = dismiss ? await api.dismissDecision(store.pid, id) : await api.answerDecision(store.pid, id, draft.option, text);
    delete app.ui.decide[id];
    await store.refreshNow();
    const words = answerWords((answer || {}).decision || {});
    say(dismiss ? 'Left to the agents: they go on with their default.' : 'Answered: ' + words + '. Every later session of the agents is shown it.');
  } catch (error) {
    draft.sending = false;
    draft.error = 'Not sent: ' + message(error);
    /* (another window may have answered it meanwhile: the list says how it stands) */
    store.refresh();
  }
  app.render('decide');
}

/** The actions of the panel, for the table of `actions.js`; `say` is its toast. */
export const decisionActs = (say) => ({
  'decide-pick': (el) => { const draft = draftOf(el.dataset.ref), n = +el.dataset.i; draft.option = draft.option === n ? null : n; draft.error = null; app.render('decide'); },
  'decide-send': (el) => send(el.dataset.ref, false, say),
  'decide-dismiss': (el) => send(el.dataset.ref, true, say),
});
/** The person typed into the field of a decision. */
export const decisionTyped = (field) => { draftOf(field.dataset.ref).text = field.value; };
