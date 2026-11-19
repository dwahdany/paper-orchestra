/* Decisions: what an agent asked the person to choose, and what they answered.

   An agent raises a decision when the answer is the researcher's choice and no evidence
   settles it. The open ones are the first thing in the inbox: the question, why it matters,
   the options as buttons a finger can press, the one the agent would take, what the agents do
   meanwhile, and a field for the person's own words. Nothing waits for an answer: the agents
   go on with the default, so a decision is counted with what waits and never blocks Continue.
   What was answered is listed below the proposals, folded, and can be answered again: the
   card then opens with the answer that holds (its option pressed, its words in the field),
   because a new answer replaces the earlier one as a whole. What is shown is what is sent.

   Pressing an option chooses it and sends nothing: words may go with it. So that nobody
   leaves with an answer that was only chosen, the card says "Not sent yet" beside the button
   that sends, and keeps the two at the foot of the list while the card is in it (on a phone
   the button is otherwise below the screen). An answer in the person's own words may be
   long: the field counts towards the server's limit (`say_chars` of a decision) before
   anything is sent, and where the answer is repeated (the toast, the folded row, the page it
   is about) it is cut short; the field of the open card has all of it.

   An address of the inbox can name a decision (`#/p/<pid>/inbox/d-1a2b3c4d`): whoever
   says "answer it in the inbox" leads to that decision, opened and in sight (`main.js`).

   The records are the server's (`GET .../decisions`); options are numbered from 1. */

import { app, link } from './core.js';
import { api, Unreachable } from './api.js';
import { ago, html, plural, raw, tf } from './util.js';
import { icon } from './vocab.js';
import { heldRuns } from './autopilot.js';

/** The decisions of a store, whatever it has loaded (a store without the list has none). */
export const allDecisions = (store) => (store && Array.isArray(store.decisions) ? store.decisions : []);
export const openDecisions = (store) => allDecisions(store).filter((d) => d.state === 'open');
/** Those that went on at the agents' default where autopilot is on: not waiting, still answerable. */
export const defaultedDecisions = (store) => allDecisions(store).filter((d) => d.state === 'defaulted');
/** Everything that waits for the person: the proposals, the open decisions, and under
    autopilot the runs it did not approve. */
export const waitingCount = (store) => store.pending().length + openDecisions(store).length + heldRuns(store).length;
/** "1 decision waits for your answer.", or nothing. */
export const decisionsSentence = (n) => (n ? plural(n, 'decision') + (n === 1 ? ' waits' : ' wait') + ' for your answer.' : '');

/** `text` on one line, cut after `limit` characters (whole ones) with an ellipsis. */
const clipped = (text, limit) => { const all = [...String(text || '').replace(/\s+/g, ' ').trim()]; return limit && all.length > limit ? all.slice(0, limit).join('').trimEnd() + '…' : all.join(''); };
/** How much of the person's own words is repeated where an answer is only named. */
const NAMED = 160;
/** What the person answered, in a few words; empty while the decision is open. Their own
    words are given whole, or cut after `limit` characters where the answer is only named. */
export function answerWords(d, limit = 0) {
  const a = d && d.state !== 'open' && d.state !== 'defaulted' ? d.answer : null;
  if (!a) return '';
  if (a.dismissed) return 'left to the agents: their default holds';
  const chosen = a.option ? (d.options || [])[a.option - 1] : null;
  const parts = [];
  if (chosen) parts.push('option ' + a.option + ', “' + chosen.label + '”');
  const text = limit ? clipped(a.text, limit) : a.text;
  if (text) parts.push(chosen ? '“' + text + '”' : text);
  return parts.join('; ');
}

const find = (id) => allDecisions(app.store).find((d) => d.id === id) || null;
/** The answer of a decision that holds an option or words (not one left to the agents). */
const held = (d) => (d && d.state === 'answered' && d.answer && !d.answer.dismissed ? d.answer : null);
/** What is being answered and not yet sent, per decision: the option tapped and the words typed.
    It starts as the answer that holds: a new answer replaces the earlier one as a whole, so
    adding words to an option must not drop the option unseen, nor a new option the words.
    Until the person touches it, it follows that answer (another window may change it). */
function draftOf(id, d = find(id)) {
  const all = (app.ui.decide = app.ui.decide || {}), a = held(d), from = a ? a.at || '' : '';
  const was = all[id];
  if (was && (was.touched || was.sending || was.from === from)) return was;
  const option = a && a.option >= 1 && a.option <= ((d || {}).options || []).length ? a.option : null;
  return (all[id] = { option, text: a ? a.text || '' : '', from, touched: false, sending: false, error: null });
}
/** Whether the card holds an answer that was not sent: an option or words the person gave
    and that are not the answer as it stands. */
function unsent(d, draft) {
  const text = (draft.text || '').trim(), now = held(d);
  if (!draft.touched || (!draft.option && !text)) return false;
  return !now || (now.option || null) !== (draft.option || null) || (now.text || '').trim() !== text;
}
/** How long the words in the field are against what the server takes (`say_chars`, counted
    as the server counts: without the blanks around them, in whole characters). Said from
    four fifths of the limit on; `over` when the server would refuse them. */
function length(d, draft) {
  const limit = +(d || {}).say_chars || 0, n = [...(draft.text || '').trim()].length;
  if (!limit || n < limit * 0.8) return { text: '', over: false };
  if (n <= limit) return { text: n.toLocaleString('en-US') + ' of ' + limit.toLocaleString('en-US') + ' characters', over: false };
  return { text: n.toLocaleString('en-US') + ' characters: ' + plural(n - limit, 'character') + ' more than the ' + limit.toLocaleString('en-US') + ' an answer can have. Shorten it, then send it.', over: true };
}
/** The address of the inbox with one decision in front. */
export const decisionLink = (id) => link({ view: 'inbox', sel: '', sub: '', ref: id || '' });
/** The decision an address of the inbox names, or nothing. */
export const decisionNamed = (route) => (route && route.view === 'inbox' && !route.sel && /^d-[0-9a-f]+$/.test(route.ref || '') ? route.ref : '');

function about(store, d) {
  const page = d.page && store.byPath ? store.byPath.get(d.page) : null;
  if (!page) return '';
  return html` · about <a href="${link({ view: 'inbox', sel: page.path, sub: '' })}">${page.id ? page.id + ' ' : ''}${page.title}</a>`;
}

/** The body of one decision: what was asked, the options, and the ways to answer. */
function card(store, d) {
  const draft = draftOf(d.id, d), settled = d.state !== 'open', role = (d.raised_by || {}).role || 'an agent';
  const now = settled && d.answer && !d.answer.dismissed ? d.answer.option : null;
  const picked = draft.option ? (d.options || [])[draft.option - 1] : null;
  const count = length(d, draft), waits = unsent(d, draft) && !draft.sending;
  /* (the field opens tall enough to read a long answer in: its row has only the beginning, and a phone has no handle to pull) */
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
    ${held(d) ? html`<p class="dc-note dc-whole">A new answer replaces this one as a whole: the agents are shown the option that is pressed and the words in the field, and nothing else of the earlier answer.</p>` : ''}
    <label class="sr" for="dc-say-${d.id}">Your answer in your own words</label>
    <textarea id="dc-say-${d.id}" class="dc-say" data-input="decide-text" data-ref="${d.id}" rows="${Math.min(8, Math.max(2, Math.ceil((draft.text || '').length / 45)))}" placeholder="Or say it in your own words${(d.options || []).length ? ' (also together with an option)' : ''}" aria-describedby="dc-count-${d.id}">${draft.text}</textarea>
    <p class="dc-count${count.over ? ' over' : ''}" id="dc-count-${d.id}" aria-live="polite"${count.text ? '' : raw(' hidden')}>${count.text}</p>
    ${draft.error ? html`<p class="dc-err" role="alert">${draft.error}</p>` : ''}
    <div class="dc-go${waits ? ' dc-unsent' : ''}"><p class="dc-wait" role="status"><b>Not sent yet.</b> ${settled ? 'The earlier answer holds until you send this one.' : 'Press “Answer” to send it.'}</p><button type="button" class="btn btn-primary" data-act="decide-send" data-ref="${d.id}"${draft.sending ? raw(' disabled') : ''}><span>${settled ? 'Change the answer' : 'Answer'}${picked ? html`: <span class="dc-pick">${picked.label}</span>` : ''}</span></button>
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
  const done = allDecisions(store).filter((d) => d.state === 'answered' || d.state === 'withdrawn')
    .sort((a, b) => String((b.answer || {}).at).localeCompare(String((a.answer || {}).at)));
  if (!done.length) return '';
  return html`<section class="in-group dc-done" aria-label="Decisions you answered"><h3>Your decisions<span>${done.length}</span></h3>
    ${done.map((d) => html`<details class="dc-old" data-keep="dc-${d.id}" data-decision="${d.id}"><summary><span class="in-dec">${icon(d.state === 'withdrawn' ? 'x' : 'check')}</span>
      <span class="dc-oq">${d.question}</span><span class="dc-oa">${answerWords(d)}</span><span class="in-when">${ago((d.answer || {}).at)}${(d.answers || []).length > 1 ? ' · changed' : ''}${about(store, d)}</span></summary>
      <div class="dc dc-in">${card(store, d)}</div></details>`)}</section>`;
}

/** What autopilot left at the agents' default: listed, each still to be answered. Nobody
    said anything to these, and nothing here says the person did. */
export function defaultedPanel(store) {
  const list = defaultedDecisions(store).slice().sort((a, b) => String(b.created).localeCompare(String(a.created)));
  if (!list.length) return '';
  return html`<section class="in-group dc-done dc-defaulted" aria-label="Decisions that went on at the agents' default"><h3>Went on at the default<span>${list.length}</span></h3>
    <p class="dc-lead">Autopilot did not wait for you: the agents went on with the default each of these states. You can still answer; your answer holds from then on.</p>
    ${list.map((d) => html`<details class="dc-old" data-keep="dc-${d.id}" data-decision="${d.id}"><summary><span class="in-dec">${icon('dot')}</span>
      <span class="dc-oq">${d.question}</span><span class="dc-oa">default: ${d.default || 'what the agent that asked would have done'}</span><span class="in-when">autopilot · ${ago(d.defaulted_at || d.created)}${about(store, d)}</span></summary>
      <div class="dc dc-in">${card(store, d)}</div></details>`)}</section>`;
}

/** The decisions that belong to a page, as one line each: for the page's own view. */
export function pageDecisions(store, path) {
  const mine = allDecisions(store).filter((d) => d.page === path);
  if (!mine.length) return '';
  return html`<div class="dc-on-page">${mine.map((d) => html`<p class="dc-ref" data-decision-of="${d.id}">${icon(d.state === 'open' || d.state === 'defaulted' ? 'dot' : d.state === 'withdrawn' ? 'x' : 'check')}<span><b>${d.state === 'open' ? 'Waits for your decision' : d.state === 'withdrawn' ? 'Left to the agents' : d.state === 'defaulted' ? 'Went on at the default (autopilot)' : 'You decided'}:</b> ${d.question}${d.state === 'open' ? '' : html` <q>${d.state === 'defaulted' ? d.default || '' : answerWords(d, NAMED)}</q>`}
    <a href="${decisionLink(d.id)}">${d.state === 'open' || d.state === 'defaulted' ? 'Answer it in the inbox' : 'Change it in the inbox'}</a></span></p>`)}</div>`;
}

const message = (error) => (error instanceof Unreachable ? 'The server is not answering.' : (error && error.message) || 'Something went wrong.');

async function send(id, dismiss, say) {
  const store = app.store, d = find(id), draft = draftOf(id, d);
  if (!store || !d || draft.sending) return;
  const text = (draft.text || '').trim(), now = held(d);
  /* (the answer as it stands, sent again, would be one more answer that says nothing new) */
  const again = !!now && (now.option || null) === (draft.option || null) && (now.text || '').trim() === text;
  if (!dismiss && (again || (!draft.option && !text))) {
    draft.error = again ? 'This is the answer you gave. Press another option or change the words, then send it.' : 'Choose an option, or say it in your own words.';
    app.render('decide');
    return;
  }
  /* (the server would refuse words over its limit: said here, with the words kept, before anything is sent) */
  const long = dismiss ? { over: false } : length(d, draft);
  if (long.over) { draft.error = 'Not sent: ' + long.text; app.render('decide'); return; }
  draft.sending = true; draft.error = null;
  app.render('decide');
  try {
    const answer = dismiss ? await api.dismissDecision(store.pid, id) : await api.answerDecision(store.pid, id, draft.option, text);
    delete app.ui.decide[id];
    await store.refreshNow();
    /* (a long answer is named by its beginning: its card under "Your decisions" has all of it) */
    const words = answerWords((answer || {}).decision || {}, NAMED);
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
  'decide-pick': (el) => { const draft = draftOf(el.dataset.ref), n = +el.dataset.i; draft.option = draft.option === n ? null : n; draft.touched = true; draft.error = null; app.render('decide'); },
  'decide-send': (el) => send(el.dataset.ref, false, say),
  'decide-dismiss': (el) => send(el.dataset.ref, true, say),
});
/** The person typed into the field of a decision. The card is not drawn again (the field
    would lose its place): what depends on the words is set where it stands. */
export const decisionTyped = (field) => {
  const id = field.dataset.ref, d = find(id), draft = draftOf(id, d);
  draft.text = field.value; draft.touched = true;
  const card = field.closest ? field.closest('.dc') : null, go = card && card.querySelector('.dc-go'), counter = card && card.querySelector('.dc-count');
  if (go) go.classList.toggle('dc-unsent', unsent(d, draft));
  if (counter) { const count = length(d, draft); counter.textContent = count.text; counter.hidden = !count.text; counter.classList.toggle('over', count.over); }
};
