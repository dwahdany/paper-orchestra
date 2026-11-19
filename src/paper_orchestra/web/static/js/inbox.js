/* The inbox: everything that waits for a decision, with the proposal beside it. */

import { app, link, listAlone } from './core.js';
import { ago, html, plain, plural, raw, setHTML, tf } from './util.js';
import { KIND, STAGES, agentName, glyph, icon, kindName, shownState } from './vocab.js';
import { afterRender, pageName, panelHTML, proposalWhat } from './page.js';
import { continueControl, engineView } from './shell.js';
import { decidedDecisions, decisionsPanel, defaultedPanel, openDecisions } from './decisions.js';
import { heldRuns } from './autopilot.js';

export const whatOf = (item) => (item.type === 'page' ? 'New ' + kindName(item.kind).toLowerCase() : proposalWhat(item.prop, item.page));
export const summaryOf = (item) => (item.type === 'edit' ? item.prop.summary : item.page.summary && item.page.summary !== item.page.title ? item.page.summary : '');

/** What the listing knows of the number check of a suggested edit to a finding. The rows of
    GET .../proposals carry it for such an edit (`number_report`: the check of the finding as
    it would read with the edit applied), so that the edit can be judged where it is listed:
    `null` for anything else; `{checked: false}` where the listing has no report (it spends a
    bounded time on them, or the check failed: the edit's own view shows it); otherwise
    `{checked: true, missing}`, the numbers that were found on no record. */
export function editCheck(item) {
  if (!item || item.type !== 'edit' || !item.prop || item.prop.number_report === undefined) return null;
  const report = item.prop.number_report;
  if (!report || typeof report !== 'object') return { checked: false, missing: [] };
  return { checked: true, missing: (Array.isArray(report.unverified) ? report.unverified : []).map(String) };
}
/** The mark a row carries for it: quiet when every number was found, said when not. */
export function checkMark(item) {
  const check = editCheck(item);
  if (!check || (check.checked && !check.missing.length)) return '';
  if (!check.checked) return html`<span class="in-flag" data-numbers="unchecked" data-tip="The numbers this edit would bring were not checked in this list. Open it: the check stands beside the text.">${icon('warn')}numbers not checked here</span>`;
  return html`<span class="in-flag" data-numbers="unverified" data-tip="With this edit the finding would name ${check.missing.slice(0, 6).join(', ')}${check.missing.length > 6 ? ' and more' : ''}: found in no run record, source or file of yours">${icon('warn')}${plural(check.missing.length, 'number')} not found</span>`;
}
/** What the quick accept of a row says it does. */
export function acceptTip(item, key) {
  const check = editCheck(item), k = key ? ' (A)' : '';
  if (check && !check.checked) return 'Open it first: its numbers were not checked in this list';
  if (check && check.missing.length) return 'Accept' + k + ': it brings ' + plural(check.missing.length, 'number') + ' found in no record';
  return 'Accept' + k;
}

/** What the person decided, the latest first: the pages an agent proposed, and the edits an
    agent suggested to a page (the store asks the server for those when this list is shown).
    `{page, edit, review, at, reason, summary}`: the page that was decided or that the edit
    was about, whether it is an edit, the decision, when, the reason of a rejection, and for
    an edit what it suggested. */
export function decidedItems(store) {
  const pages = store.pages.filter((pg) => pg.meta && pg.meta.decided_by === 'human' && pg.meta.decided_at && pg.author !== 'human')
    .map((pg) => ({ page: pg, edit: false, review: pg.review === 'rejected' ? 'rejected' : 'accepted', at: pg.meta.decided_at, reason: pg.review === 'rejected' ? pg.meta.decision_reason || '' : '', summary: '' }));
  const edits = (store.decidedEdits || []).map((p) => ({ page: store.byPath.get(p.target), edit: true, id: p.id, review: p.review, at: p.decided_at, reason: p.review === 'rejected' ? p.decision_reason || '' : '', summary: p.summary || '' })).filter((it) => it.page);
  /* (decisions of one moment: an edit after the page it was about, as they were made) */
  return pages.concat(edits).sort((a, b) => String(b.at).localeCompare(String(a.at)) || (b.edit ? 1 : 0) - (a.edit ? 1 : 0));
}
/** How many of them the list shows before "earlier" has to be asked for. */
const DECIDED_SHOWN = 8;

/** The item the inbox is on: the one about the selected page, or the first. */
export function currentItem(store, route) {
  const items = store.pending();
  return items.find((it) => it.page.path === route.sel) || (route.sel && store.byPath.has(route.sel) ? null : items[0]) || null;
}

export const inboxView = {
  render(root, route) {
    const store = app.store, items = store.pending(), cur = currentItem(store, route), alone = listAlone(route);
    /* what autopilot does not approve: it waits for the person although autopilot is on */
    const heldList = heldRuns(store).map((h) => ({ h, x: store.expById.get(h.id) })).filter((it) => it.x);
    const asked = openDecisions(store).length, waiting = items.length + asked + heldList.length;   // (a decision waits like a proposal, and holds nothing up)
    const checked = app.ui.checked;
    Object.keys(checked).forEach((ref) => { if (!items.find((it) => it.ref === ref)) delete checked[ref]; });
    const nChecked = Object.keys(checked).length;
    const kinds = STAGES.concat(Array.from(new Set(items.map((it) => it.kind))).filter((k) => STAGES.indexOf(k) < 0));
    const list = kinds.map((k) => {
      const group = items.filter((it) => it.kind === k);
      if (!group.length) return '';
      return html`<div class="in-group"><h3>${(KIND[k] || { many: k }).many}<span>${group.length}</span></h3>${group.map((it) => {
        const on = !alone && cur && it.ref === cur.ref;      // (the list alone, on a phone, has no proposal in front)
        const sum = summaryOf(it);
        return html`<div class="in-row${on ? ' cur' : ''}" data-ref="${it.ref}">
          <label class="in-check"><input type="checkbox" data-check="${it.ref}"${checked[it.ref] ? raw(' checked') : ''} aria-label="Select ${it.page.id || it.page.title}"></label>
          <a class="in-main" href="${link({ view: 'inbox', sel: it.page.path, sub: '' })}"${on ? raw(' aria-current="true"') : ''}>
            <span class="in-name">${glyph(it.kind, it.type === 'page' ? 'proposed' : 'accepted', shownState(it.page), 18)}${pageName(it.page)}</span>
            ${sum ? html`<span class="in-sum">${sum}</span>` : ''}
            <span class="in-what"><span class="in-kind">${whatOf(it)}</span>${agentName(it.author)}${it.round ? ', round ' + it.round : ''} · ${ago(it.when)}</span>${checkMark(it)}</a>
          <span class="in-quick"><button class="ibtn ibtn-yes" data-act="accept" data-ref="${it.ref}" data-tip="${acceptTip(it, true)}" aria-label="${acceptTip(it)}">${icon('check')}</button>
            <button class="ibtn ibtn-no" data-act="reject-ask" data-ref="${it.ref}" data-key="${it.page.path}" data-tip="Reject, with a reason (R)" aria-label="Reject">${icon('x')}</button></span></div>`;
      })}</div>`;
    });
    /* what was decided: pages, and suggested edits, each with the decision and the reason of a rejection */
    store.loadDecided(false);
    const decidedAll = decidedItems(store);
    const decided = decidedAll.slice(0, app.ui.decidedAll ? 60 : DECIDED_SHOWN);
    const done = decided.length ? html`<div class="in-group in-done"><h3>Decided<span>${decidedAll.length}</span></h3>${decided.map((it) => html`
      <a class="in-row in-old${it.edit ? ' in-edit' : ''}${route.sel === it.page.path ? ' cur' : ''}" href="${link({ view: 'inbox', sel: it.page.path, sub: '' })}"${it.edit ? html` data-decided="${it.id}"` : ''}><span class="in-dec">${icon(it.review === 'rejected' ? 'x' : 'check')}</span>
        <span class="in-name">${pageName(it.page)}</span><span class="in-when">${it.edit ? 'edit ' : ''}${it.review} ${ago(it.at)}</span>
        ${it.edit && it.summary ? html`<span class="in-why in-sugg">${it.summary}</span>` : ''}${it.reason ? html`<span class="in-why"><q>${it.reason}</q></span>` : ''}</a>`)}
      ${decidedAll.length > decided.length && !app.ui.decidedAll ? html`<button class="linkish in-more" data-act="decided-more">Show earlier decisions</button>` : ''}</div>` : '';
    const held = heldList.length ? html`<div class="in-group in-held"><h3>${heldList.some(({ h }) => h.kind === 'refused') ? 'Runs autopilot did not approve' : 'Runs autopilot left for you'}<span>${heldList.length}</span></h3>${heldList.map(({ h, x }) => html`
      <a class="in-row in-old" data-held="${h.id}" href="${link({ view: 'experiments', sel: x.path, sub: '' })}"><span class="in-dec">${icon('warn')}</span><span class="in-name">${pageName(x)}</span><span class="in-when">${h.kind === 'refused' ? 'cannot be approved' : 'not approved'}</span><span class="in-why">${plain(h.reason)}</span></a>`)}</div>` : '';
    const shown = cur ? cur.page : route.sel && store.byPath.has(route.sel) ? store.byPath.get(route.sel) : null;
    const e = engineView(store);
    const idle = !e.working && !e.unverified;
    setHTML(root, html`<div class="inbox${route.sel ? ' in-open' : ''}">
      <section class="in-list" aria-label="Waiting for a decision"><header class="in-head"><h1>${waiting ? waiting + ' waiting for you' : 'Nothing is waiting'}</h1>
        ${items.length ? html`<div class="in-bulk"><label class="in-all"><input type="checkbox" data-check="all"${nChecked && nChecked === items.length ? raw(' checked') : ''}> Select all</label>
          <button class="btn btn-s" data-act="bulk-accept"${nChecked ? '' : raw(' disabled')}>Accept ${nChecked || ''} selected <kbd>⇧A</kbd></button>
          <button class="btn btn-s btn-neg-quiet" data-act="bulk-reject"${nChecked ? '' : raw(' disabled')} aria-expanded="${tf(app.ui.rejecting === 'bulk')}">Reject selected…</button></div>
          ${app.ui.rejecting === 'bulk' ? html`<form class="reject-form" data-form="bulk-reject"><label class="sr" for="why-bulk">Reason</label><input id="why-bulk" name="reason" data-input="reason" data-ref="bulk" value="${app.ui.reasons.bulk || ''}" placeholder="Why? One reason for all ${nChecked}; the agents read it." autocomplete="off"><button class="btn btn-neg" type="submit">Reject ${nChecked}</button><button class="btn btn-quiet" type="button" data-act="reject-cancel">Cancel</button></form>` : ''}` : ''}</header>
        <div class="in-scroll">${decisionsPanel(store)}${held}${items.length ? list : html`<div class="in-clear"><p>Every proposal has a decision.${heldList.length ? html` <span class="in-held-still">${plural(heldList.length, 'run')} above still ${heldList.length === 1 ? 'waits' : 'wait'} for you: autopilot did not approve ${heldList.length === 1 ? 'it' : 'them'}.</span>` : idle && e.mode === 'checkpoint' ? ' The checkpoint can go on' + (asked ? ' without your answer: the agents use their default.' : '.') : ''}</p>${idle && !e.fresh ? continueControl(store, e, 'inbox') : ''}</div>`}${defaultedPanel(store)}${decidedDecisions(store)}${done}</div>
        <footer class="in-keys"><span><kbd>J</kbd><kbd>K</kbd> move</span><span><kbd>X</kbd> select</span><span><kbd>A</kbd> accept</span><span data-tip="Saving an edit does not decide the proposal: “Save and accept” does"><kbd>E</kbd> edit</span><span><kbd>R</kbd> reject</span><span data-tip="A decision is sent after five seconds; until then Z takes it back"><kbd>Z</kbd> undo, within 5 s</span></footer></section>
      <section class="in-detail" aria-label="The proposal"><a class="back in-back" href="${link({ view: 'inbox', sel: '', sub: '' })}">Inbox${waiting ? html` <span class="quiet">· ${waiting} waiting</span>` : ''}</a>${shown ? panelHTML(store, shown, { mode: 'inbox', sub: route.sub }) : html`<div class="in-none"><p>${items.length ? 'Select a proposal to read it.' : 'Proposals appear here when the agents have something for you to decide.'}</p></div>`}</section></div>`);
    afterRender(root);
  },
};
