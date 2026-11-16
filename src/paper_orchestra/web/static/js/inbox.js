/* The inbox: everything that waits for a decision, with the proposal beside it. */

import { app, link } from './core.js';
import { ago, html, raw, setHTML, tf } from './util.js';
import { KIND, STAGES, agentName, glyph, icon, kindName } from './vocab.js';
import { afterRender, pageName, panelHTML, proposalWhat } from './page.js';
import { continueLabel, engineView } from './shell.js';

export const whatOf = (item) => (item.type === 'page' ? 'New ' + kindName(item.kind).toLowerCase() : proposalWhat(item.prop, item.page));
export const summaryOf = (item) => (item.type === 'edit' ? item.prop.summary : item.page.summary && item.page.summary !== item.page.title ? item.page.summary : '');

/** The item the inbox is on: the one about the selected page, or the first. */
export function currentItem(store, route) {
  const items = store.pending();
  return items.find((it) => it.page.path === route.sel) || (route.sel && store.byPath.has(route.sel) ? null : items[0]) || null;
}

export const inboxView = {
  render(root, route) {
    const store = app.store, items = store.pending(), cur = currentItem(store, route);
    const checked = app.ui.checked;
    Object.keys(checked).forEach((ref) => { if (!items.find((it) => it.ref === ref)) delete checked[ref]; });
    const nChecked = Object.keys(checked).length;
    const kinds = STAGES.concat(Array.from(new Set(items.map((it) => it.kind))).filter((k) => STAGES.indexOf(k) < 0));
    const list = kinds.map((k) => {
      const group = items.filter((it) => it.kind === k);
      if (!group.length) return '';
      return html`<div class="in-group"><h3>${(KIND[k] || { many: k }).many}<span>${group.length}</span></h3>${group.map((it) => {
        const on = cur && it.ref === cur.ref;
        const sum = summaryOf(it);
        return html`<div class="in-row${on ? ' cur' : ''}" data-ref="${it.ref}">
          <label class="in-check"><input type="checkbox" data-check="${it.ref}"${checked[it.ref] ? raw(' checked') : ''} aria-label="Select ${it.page.id || it.page.title}"></label>
          <a class="in-main" href="${link({ view: 'inbox', sel: it.page.path, sub: '' })}"${on ? raw(' aria-current="true"') : ''}>
            <span class="in-name">${glyph(it.kind, it.type === 'page' ? 'proposed' : 'accepted', it.page.state, 18)}${pageName(it.page)}</span>
            ${sum ? html`<span class="in-sum">${sum}</span>` : ''}
            <span class="in-what"><span class="in-kind">${whatOf(it)}</span>${agentName(it.author)}${it.round ? ', round ' + it.round : ''} · ${ago(it.when)}</span></a>
          <span class="in-quick"><button class="ibtn ibtn-yes" data-act="accept" data-ref="${it.ref}" data-tip="Accept (A)" aria-label="Accept">${icon('check')}</button>
            <button class="ibtn ibtn-no" data-act="reject-ask" data-ref="${it.ref}" data-key="${it.page.path}" data-tip="Reject, with a reason (R)" aria-label="Reject">${icon('x')}</button></span></div>`;
      })}</div>`;
    });
    const decidedAll = store.pages.filter((pg) => pg.meta && pg.meta.decided_by === 'human' && pg.meta.decided_at && pg.author !== 'human');
    const decided = decidedAll.slice().sort((a, b) => String(b.meta.decided_at).localeCompare(String(a.meta.decided_at))).slice(0, 8);
    const done = decided.length ? html`<div class="in-group in-done"><h3>Decided<span>${decidedAll.length}</span></h3>${decided.map((pg) => html`
      <a class="in-row in-old${route.sel === pg.path ? ' cur' : ''}" href="${link({ view: 'inbox', sel: pg.path, sub: '' })}"><span class="in-dec">${icon(pg.review === 'rejected' ? 'x' : 'check')}</span>
        <span class="in-name">${pageName(pg)}</span><span class="in-when">${pg.review === 'rejected' ? 'rejected ' : 'accepted '}${ago(pg.meta.decided_at)}</span></a>`)}</div>` : '';
    const shown = cur ? cur.page : route.sel && store.byPath.has(route.sel) ? store.byPath.get(route.sel) : null;
    const e = engineView(store);
    const idle = e.mode !== 'running' && e.mode !== 'experiment' && e.mode !== 'writing';
    setHTML(root, html`<div class="inbox">
      <section class="in-list" aria-label="Waiting for a decision"><header class="in-head"><h1>${items.length ? items.length + ' waiting for you' : 'Nothing is waiting'}</h1>
        ${items.length ? html`<div class="in-bulk"><label class="in-all"><input type="checkbox" data-check="all"${nChecked && nChecked === items.length ? raw(' checked') : ''}> Select all</label>
          <button class="btn btn-s" data-act="bulk-accept"${nChecked ? '' : raw(' disabled')}>Accept ${nChecked || ''} selected <kbd>⇧A</kbd></button>
          <button class="btn btn-s btn-neg-quiet" data-act="bulk-reject"${nChecked ? '' : raw(' disabled')} aria-expanded="${tf(app.ui.rejecting === 'bulk')}">Reject selected…</button></div>
          ${app.ui.rejecting === 'bulk' ? html`<form class="reject-form" data-form="bulk-reject"><label class="sr" for="why-bulk">Reason</label><input id="why-bulk" name="reason" data-input="reason" data-ref="bulk" value="${app.ui.reasons.bulk || ''}" placeholder="Why? One reason for all ${nChecked}; the agents read it." autocomplete="off"><button class="btn btn-neg" type="submit">Reject ${nChecked}</button><button class="btn btn-quiet" type="button" data-act="reject-cancel">Cancel</button></form>` : ''}` : ''}</header>
        <div class="in-scroll">${items.length ? list : html`<div class="in-clear"><p>Every proposal has a decision.${idle && e.mode === 'checkpoint' ? ' The checkpoint can go on.' : ''}</p>${idle && !e.fresh ? html`<button class="btn btn-primary" data-act="continue">${continueLabel(e.next, store)} ${icon('right')}</button>` : ''}</div>`}${done}</div>
        <footer class="in-keys"><span><kbd>J</kbd><kbd>K</kbd> move</span><span><kbd>X</kbd> select</span><span><kbd>A</kbd> accept</span><span><kbd>E</kbd> edit, then accept</span><span><kbd>R</kbd> reject</span><span><kbd>Z</kbd> undo, for a few seconds</span></footer></section>
      <section class="in-detail" aria-label="The proposal">${shown ? panelHTML(store, shown, { mode: 'inbox', sub: route.sub }) : html`<div class="in-none"><p>${items.length ? 'Select a proposal to read it.' : 'Proposals appear here when the agents have something for you to decide.'}</p></div>`}</section></div>`);
    afterRender(root);
  },
};
