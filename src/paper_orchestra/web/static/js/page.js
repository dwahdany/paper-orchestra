/* The page component. One component in every view: status line, review bar (or margin
   note), body, lineage, editor, history. The body is the server's rendering; a suggested
   edit is read in place (the server's `marked_html`), the unified diff one click away. */

import { app, link } from './core.js';
import { api } from './api.js';
import { $, $$, ago, bytes, cap, dateTime, debounce, html, join, lineDiff, minutes, money, plain, plural, raw, storage, tf } from './util.js';
import { STATES, TONE, agentName, glyph, icon, kindName, pageGlyph } from './vocab.js';

/* ---------- naming a page ---------- */
export function shortTitle(page) {
  if (page.kind === 'result') { const m = /([A-Za-z]+\d+)-run(\d+)\.md$/.exec(page.path); if (m) return m[1] + ' · run ' + m[2]; }
  if (page.kind === 'source') return page.id || page.title;
  return page.title;
}
/** The shortest thing that names the page: H2, Kingma2015Adam, E1 · run 1, Idea, or its title. */
export const label = (page) => (page.kind === 'idea' ? 'Idea' : page.kind === 'source' || page.kind === 'result' ? shortTitle(page) : page.id || page.title);
const hasId = (page) => page.id && page.kind !== 'source';
/** A row that names a page: id and title. Used by every list. */
export const pageName = (page) => html`${hasId(page) ? html`<b class="pn-id">${page.id}</b>` : ''}<span class="pn-t">${page.kind === 'source' || page.kind === 'result' ? shortTitle(page) : page.title}</span>`;
export const pageTip = (page) => kindName(page.kind) + (hasId(page) ? ' ' + page.id : '') + ' · ' + page.title;

/* ---------- status: three dimensions, each with a word and a shape ---------- */
export const stateTag = (page) => (page.state ? html`<span class="tag tag-state t-${TONE[page.state] || 'none'}">${glyph(page.kind, 'accepted', page.state, 14)}${page.state}</span>` : '');
export const reviewTag = (page) => (page.review === 'proposed' ? html`<span class="tag tag-proposed">Proposed</span>` : page.review === 'rejected' ? html`<span class="tag tag-rejected">Rejected</span>` : '');
export const who = (author) => (author === 'human' ? 'you' : author === 'system' ? 'code' : 'the ' + agentName(author));
function byline(page) {
  const meta = page.meta || {};
  const parts = [(page.author === 'human' ? 'written by you' : page.author === 'system' ? 'record written by code' : 'by the ' + agentName(page.author)) + (page.round ? ', round ' + page.round : '')];
  if (page.review === 'accepted' && meta.decided_by && page.author !== 'human' && page.author !== 'system') parts.push('accepted by ' + who(meta.decided_by) + ' ' + ago(meta.decided_at));
  if (page.optimistic) parts.push('decided just now');
  parts.push('updated ' + ago(page.updated) + (meta.updated_by && meta.updated_by !== page.author ? ' by ' + who(meta.updated_by) : ''));
  return parts.map((text) => html`<span>${text}</span>`);
}

/** The words that say where a page stands; the first is the essential one. */
export function statusParts(store, page) {
  if (page.review === 'proposed') return ['proposed', agentName(page.author)];
  if (page.review === 'rejected') return ['rejected'];
  const change = store.pendingFor(page.path).length ? ['change proposed'] : [];
  if (page.kind === 'hypothesis' || page.kind === 'question') return [page.state || 'open'].concat(change);
  if (page.kind === 'experiment') {
    const row = store.expById.get(page.id) || {};
    const run = row.last_run || {};
    if (page.state === 'running') return ['running'];
    if (page.state === 'succeeded') return ['succeeded'].concat(run.seconds != null ? [durationShort(run.seconds)] : []);
    if (page.state === 'failed') return ['failed', 'needs you'];
    if (page.state === 'approved') return ['approved', 'runs next'];
    if (page.state === 'queued') return ['queued'];
    if (page.state === 'planned') return ['planned', 'not approved'];
    return [page.state || 'planned'];
  }
  if (page.kind === 'result') return [page.state || ''];
  if (page.kind === 'finding') {
    const name = (ref) => { const p = store.page(ref); return p ? label(p) : ref; };
    const rel = page.relations || {};
    const parts = [];
    if ((rel.supports || []).length) parts.push('supports ' + rel.supports.map(name).join(', '));
    if ((rel.refutes || []).length) parts.push('refutes ' + rel.refutes.map(name).join(', '));
    return parts.concat(change);
  }
  if (page.kind === 'idea') { const n = ((store.project || {}).attachments || []).length; return n ? [plural(n, 'file')] : []; }
  return change;
}
const durationShort = (s) => (s < 60 ? (s < 10 ? Math.round(s * 10) / 10 : Math.round(s)) + ' s' : s < 3600 ? Math.floor(s / 60) + ' m ' + Math.round(s % 60) + ' s' : Math.floor(s / 3600) + ' h ' + Math.round((s % 3600) / 60) + ' m');

/** The same, as one line for the map: the full wording, a short one for narrow columns, and
    the ref of what can be decided there. */
export function statusLine(store, page) {
  const parts = statusParts(store, page).filter(Boolean);
  const proposal = page.review === 'accepted' ? store.pendingFor(page.path)[0] : null;
  if (proposal) {
    const to = proposal.fields && proposal.fields.state;
    if (to) return { full: (page.state || '') + ' → ' + to, short: '→ ' + to, ref: proposal.id };
    return { full: (page.state ? page.state + ' · ' : '') + 'edit proposed', short: 'edit proposed', ref: proposal.id };
  }
  return { full: parts.join(' · '), short: parts[0] || '', ref: page.review === 'proposed' ? page.path : '' };
}

/* ---------- what an edit proposal changes, in words ---------- */
export function proposalWhat(proposal, page) {
  const fields = Object.entries(proposal.fields || {});
  const parts = fields.map(([name, value]) => name + ' ' + (page && page[name] != null ? page[name] + ' → ' : '→ ') + value);
  const added = Object.entries(proposal.relations_add || {}).filter(([, list]) => list && list.length).map(([name, list]) => 'adds ' + name.replace(/_/g, ' ') + ' ' + list.join(', '));
  if (proposal.body != null) parts.push('Edit to the text');
  return parts.concat(added).join(', ') || 'Edit';
}

/* ---------- reject, with a reason ---------- */
const rejectForm = (ref) => html`<form class="reject-form" data-form="reject" data-ref="${ref}">
  <label class="sr" for="why-${ref}">Reason</label>
  <input id="why-${ref}" name="reason" data-input="reason" data-ref="${ref}" value="${app.ui.reasons[ref] || ''}" placeholder="Why? The agents read this reason." autocomplete="off">
  <button class="btn btn-neg" type="submit">Reject</button><button class="btn btn-quiet" type="button" data-act="reject-cancel">Cancel</button></form>`;

/* ---------- experiments: where it would run ---------- */
const BACKEND_NAME = { local: 'Local CPU', modal: 'Modal' };
export const backendName = (name, gpu) => (BACKEND_NAME[name] || cap(name || '')) + (name === 'modal' ? (gpu ? ' GPU · ' + gpu : ' sandbox') : '');
export function optionLabel(option, plan) {
  const parts = [backendName(option.backend, plan && plan.gpu)];
  if (option.est_minutes != null) parts.push('about ' + minutes(option.est_minutes));
  parts.push(option.est_usd ? 'about ' + money(option.est_usd) : option.backend === 'local' ? 'no cost' : option.est_usd === 0 ? '$0.00' : 'cost not estimated');
  return parts.join(' · ');
}
/** The options chosen for an experiment's run, starting from what the plan suggests. */
export function approvalChoice(eid, data) {
  const chosen = app.ui.approve[eid] || (app.ui.approve[eid] = {});
  const options = (data && data.options) || [];
  const runnable = options.filter((o) => o.can_run);
  const plan = (data && data.plan) || {};
  if (!chosen.backend || !runnable.some((o) => o.backend === chosen.backend)) {
    const suggested = runnable.find((o) => o.backend === plan.backend) || runnable[0];
    chosen.backend = suggested ? suggested.backend : '';
  }
  return chosen;
}
/** A plan that asks for network access is never given it by default, and never refused it
    by default either: the person answers, one way or the other, before a run is approved.
    The answer is kept in `app.ui.approve[eid].network` (true or false; undefined = not yet). */
export function networkAsk(eid, plan) {
  if (!plan || !plan.network) return '';
  const answer = (app.ui.approve[eid] || {}).network;
  const answered = answer === true || answer === false;
  const nudge = !answered && app.ui.netNudge === eid;
  return html`<fieldset class="net-ask${answered ? '' : ' unanswered'}${nudge ? ' nudge' : ''}" data-net="${eid}">
    <legend><b>This plan asks for network access.</b> The run gets it only if you allow it here.</legend>
    <div class="net-opts">
      <label><input type="radio" name="net-${eid}" value="yes" data-input="ap-net-choice" data-eid="${eid}"${answer === true ? raw(' checked') : ''}> Allow network access</label>
      <label><input type="radio" name="net-${eid}" value="no" data-input="ap-net-choice" data-eid="${eid}"${answer === false ? raw(' checked') : ''}> Run it without network access</label></div>
    ${answered ? '' : html`<span class="net-need"${nudge ? raw(' role="alert"') : ''}>Choose one of the two before you approve the run.</span>`}</fieldset>`;
}
function runOn(store, page, entry) {
  const data = entry && entry.data;
  if (!data) return html`<p class="quiet runon-wait">${entry && entry.error ? 'What can run this could not be read: ' + entry.error.message : 'Checking where this can run…'}</p>`;
  const options = data.options || [], runnable = options.filter((o) => o.can_run);
  if (data.plan_error) return html`<p class="note note-warn">${icon('warn')}The plan cannot be read as it is: ${data.plan_error}</p>`;
  if (!runnable.length) {
    return html`<div class="note note-warn">${icon('warn')}<div><b>No backend can run this as it is.</b> You can still accept the plan and keep it planned.<ul class="reasons">${options.map((o) => html`<li><b>${backendName(o.backend, data.plan && data.plan.gpu)}</b> ${(o.reasons || []).join(' ')}</li>`)}</ul>
      ${data.options_error ? html`<p>${data.options_error}</p>` : ''}</div></div>`;
  }
  const chosen = approvalChoice(page.id, data);
  return html`<label class="runon"><span>Run on</span><select data-input="runon" data-eid="${page.id}" id="runon-${page.id}">
    ${runnable.map((o) => html`<option value="${o.backend}"${o.backend === chosen.backend ? raw(' selected') : ''}>${optionLabel(o, data.plan)}</option>`)}</select></label>${networkAsk(page.id, data.plan)}`;
}

/* ---------- review bar: accept / edit, then accept / reject ---------- */
export function reviewBar(store, page, opt) {
  opt = opt || {};
  const key = (k) => (opt.compact ? '' : html` <kbd>${k}</kbd>`);
  const parts = [];
  const whole = opt.part !== 'edits', edits = opt.part !== 'page';
  if (whole && page.review === 'proposed') {
    const exp = page.kind === 'experiment';
    const entry = exp && !opt.compact ? store.experiment(page.id) : null;
    const canRun = !!(entry && entry.data && (entry.data.options || []).some((o) => o.can_run) && !entry.data.plan_error);
    parts.push(html`<div class="review${opt.compact ? ' review-compact' : ''}" data-ref="${page.path}">
      ${opt.compact ? '' : html`<p class="review-say"><b>Proposed</b> by ${who(page.author)}${page.round ? ' in round ' + page.round : ''}, ${ago(page.updated)}. It is not part of the wiki until you accept it${exp ? ', and it will not run until you approve it' : ''}.</p>`}
      ${exp && !opt.compact ? runOn(store, page, entry) : ''}
      ${app.ui.rejecting === page.path ? rejectForm(page.path) : html`<div class="review-actions">
        ${exp
          ? html`${opt.compact ? '' : html`<button class="btn btn-primary" data-act="accept-run" data-ref="${page.path}"${canRun ? '' : raw(' disabled')}>${icon('play')}Accept and run</button>`}
              <button class="btn${opt.compact ? ' btn-primary' : ''}" data-act="accept" data-ref="${page.path}">${opt.compact ? html`${icon('check')}Accept, keep planned` : 'Accept, keep planned'}${key('A')}</button>`
          : html`<button class="btn btn-primary" data-act="accept" data-ref="${page.path}">${icon('check')}Accept${key('A')}</button>`}
        <button class="btn" data-act="edit" data-ref="${page.path}">${exp || opt.compact ? 'Edit' : 'Edit, then accept'}${key('E')}</button>
        <button class="btn btn-neg-quiet" data-act="reject-ask" data-ref="${page.path}">Reject…${key('R')}</button></div>`}
    </div>`);
  }
  if (edits) store.pendingFor(page.path).forEach((proposal) => parts.push(changeNote(store, page, proposal, opt)));
  if (whole && page.review === 'rejected' && !opt.compact) {
    const meta = page.meta || {};
    parts.push(html`<div class="note note-rejected"><b>Rejected</b> by ${who(meta.decided_by || 'human')}${meta.decided_at ? ', ' + dateTime(meta.decided_at) : ''}.${meta.decision_reason ? html` <q>${meta.decision_reason}</q>` : ''}
      <span class="note-sub">The page is kept. Agents see its title and this reason, nothing else. <button class="linkish" data-act="reaccept" data-ref="${page.path}">Accept it after all</button></span></div>`);
  }
  return join(parts);
}

/** A suggested edit as a note: who, what, how much, and the three answers. */
function changeNote(store, page, proposal, opt) {
  const view = (store.views.get(page.path) || {}).data;
  const full = view && (view.proposals || []).find((p) => p.id === proposal.id);
  const changes = Object.entries(proposal.fields || {}).map(([name, value]) => html`<span class="chg"><span class="chg-f">${name}</span>${page.meta && page.meta[name] != null ? html`<s>${page.meta[name]}</s>${icon('right')}` : ''}<b>${value}</b></span>`);
  const added = Object.entries(proposal.relations_add || {}).filter(([, list]) => list && list.length).map(([name, list]) => html`<span class="chg"><span class="chg-f">adds ${name.replace(/_/g, ' ')}</span><b>${list.join(', ')}</b></span>`);
  const text = proposal.body != null;
  const asDiff = !!app.ui.asDiff[proposal.id];
  const key = (k) => (opt.compact || opt.margin ? '' : html` <kbd>${k}</kbd>`);   // the margin is narrow: its three answers keep to one line
  return html`<div class="review review-change${opt.compact ? ' review-compact' : ''}${opt.margin ? ' review-margin' : ''}" data-ref="${proposal.id}">
    <p class="review-say"><b>${cap(who(proposal.author))} suggests</b>${opt.compact ? ': ' : html` <span class="quiet">· ${ago(proposal.created)}${proposal.round ? ', round ' + proposal.round : ''}</span><br>`}<span class="review-sum">${proposal.summary}</span></p>
    ${changes.length || added.length ? html`<div class="chg-row">${changes}${added}</div>` : ''}
    ${text && !opt.compact ? html`<p class="review-count">${full ? html`<span class="w-add">+${full.words_added} ${full.words_added === 1 ? 'word' : 'words'}</span> <span class="w-del">−${full.words_removed}</span>` : ''}
      <button class="linkish" data-act="toggle-diff" data-ref="${proposal.id}" aria-pressed="${tf(asDiff)}">${asDiff ? 'show in the text' : 'show as diff'}</button>
      ${asDiff ? '' : html`<button class="linkish" data-act="jump-change">go to the change</button>`}</p>` : ''}
    ${full && full.page_changed ? html`<p class="review-why">${icon('warn')} The page changed after this was suggested. Accepting applies the fields; the text is not replaced.</p>` : ''}
    ${text && asDiff && full && !opt.compact ? html`<div class="diff-wrap">${raw(full.diff_html)}</div>` : ''}
    ${app.ui.rejecting === proposal.id ? rejectForm(proposal.id) : html`<div class="review-actions">
      <button class="btn btn-primary" data-act="accept" data-ref="${proposal.id}">${icon('check')}Accept${opt.margin || opt.compact ? '' : ' change'}${key('A')}</button>
      ${text && !opt.compact ? html`<button class="btn" data-act="edit-proposal" data-ref="${proposal.id}" data-path="${page.path}">Edit</button>` : ''}
      <button class="btn btn-neg-quiet" data-act="reject-ask" data-ref="${proposal.id}">Reject…${key('R')}</button></div>`}
  </div>`;
}

/* ---------- number check on findings ---------- */
export function numberCheckHTML(store, view) {
  if (!view || view.page.kind !== 'finding') return '';
  const checks = view.number_check;
  if (checks == null) return html`<section class="numcheck"><h3 class="rail-h">Number check</h3><p class="quiet">The numbers could not be checked${view.number_check_error ? ': ' + view.number_check_error : '.'}</p></section>`;
  if (!checks.length) return html`<section class="numcheck"><h3 class="rail-h">Number check</h3><p class="quiet">No number in this finding is specific enough to need a record.</p></section>`;
  const found = checks.filter((c) => (c.found_in || []).length), missing = checks.filter((c) => !(c.found_in || []).length);
  const places = new Map();
  found.forEach((c) => c.found_in.forEach((p) => places.set(p, (places.get(p) || 0) + 1)));
  const top = Array.from(places.entries()).sort((a, b) => (b[0].startsWith('wiki/results/') ? 1 : 0) - (a[0].startsWith('wiki/results/') ? 1 : 0) || b[1] - a[1]).slice(0, 3);
  return html`<section class="numcheck"><h3 class="rail-h">Number check</h3>
    <p class="numcheck-sum">${found.length} of ${plural(checks.length, 'number')} ${found.length === 1 ? 'occurs' : 'occur'} in a record, a source or the inputs${top.length ? html`: ${join(top.map(([path]) => placeLink(store, path)), ', ')}` : ''}.</p>
    ${missing.map((c) => html`<p class="numcheck-flag">${icon('warn')}<span><b>${c.number}</b> was not found in a record, a source or the inputs. It may be derived: check it before you rely on it.</span></p>`)}
  </section>`;
}
function placeLink(store, path) {
  const page = store.page(path);
  if (page) return html`<a class="numplace" href="${link({ sel: page.path, sub: '' })}" data-sel="${page.path}">${label(page)}</a>`;
  return html`<a class="numplace" href="${api.fileUrl(store.pid, path)}" target="_blank" rel="noopener">${path.replace(/^inputs\//, '')}</a>`;
}
/** Mark every checked number in the rendered text, through text nodes: a number found in a
    record links to it; one found nowhere is flagged. Nothing is built from strings here. */
export function markNumbers(root, checks, store) {
  if (!checks || !checks.length || root.dataset.marked) return;
  root.dataset.marked = '1';
  const byNumber = new Map(checks.map((c) => [String(c.number), c]));
  const numbers = Array.from(byNumber.keys()).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!numbers.length) return;
  const pattern = new RegExp(numbers.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (node) => (node.parentElement && node.parentElement.closest('code, pre, .num, del') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT) });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    const text = node.nodeValue;
    const pieces = [];
    let last = 0, match;
    pattern.lastIndex = 0;
    while ((match = pattern.exec(text))) {
      const before = text[match.index - 1] || '', after = text.slice(match.index + match[0].length, match.index + match[0].length + 2);
      /* part of a longer number (12.5 in 112.57): not this one */
      if (/[\d]/.test(before) || (/[.,]/.test(before) && /\d/.test(text[match.index - 2] || '')) || /^\d/.test(after) || /^[.,]\d/.test(after)) continue;
      if (match.index > last) pieces.push(text.slice(last, match.index));
      pieces.push(numberNode(match[0], byNumber.get(match[0]), store, !!node.parentElement.closest('a')));
      last = match.index + match[0].length;
    }
    if (!pieces.length) return;
    if (last < text.length) pieces.push(text.slice(last));
    node.replaceWith(...pieces);
  });
}
function numberNode(text, check, store, insideLink) {
  const found = (check.found_in || []);
  if (!found.length) {
    const flag = document.createElement('span');
    flag.className = 'num num-flag';
    flag.tabIndex = 0;
    flag.dataset.tip = text + ' was not found in a record, a source or the inputs.';
    flag.append(text);
    const mark = document.createElement('sup');
    mark.textContent = '?';
    mark.setAttribute('aria-label', 'not found in a record');
    flag.append(mark);
    return flag;
  }
  const place = found.find((p) => p.startsWith('wiki/results/')) || found.find((p) => store.page(p)) || found[0];
  const page = store.page(place);
  const names = found.slice(0, 3).map((p) => { const pg = store.page(p); return pg ? label(pg) : p.replace(/^inputs\//, ''); });
  const tip = text + ' is in ' + names.join(', ') + (found.length > 3 ? ' and ' + (found.length - 3) + ' more' : '');
  const el = document.createElement(insideLink ? 'span' : 'a');
  el.className = 'num';
  el.dataset.tip = tip;
  el.textContent = text;
  if (!insideLink) {
    if (page) { el.href = link({ sel: page.path, sub: '' }); el.dataset.sel = page.path; }
    else { el.href = api.fileUrl(store.pid, place); el.target = '_blank'; el.rel = 'noopener'; }
  }
  return el;
}

/* ---------- lineage: what the page rests on, what follows from it ---------- */
export function lineageHTML(store, page) {
  const lin = store.lineage(page.path);
  if (!lin.up.length && !lin.down.length) return html`<section class="lineage"><p class="quiet">Nothing links to or from this page yet.</p></section>`;
  const row = (x) => html`<a class="lin r-${x.page.review}" href="${link({ sel: x.page.path, sub: '' })}" data-sel="${x.page.path}" data-tip="${pageTip(x.page)}"><span class="lin-verb">${x.verb}</span>${pageGlyph(x.page, 16)}<span class="lin-name">${hasId(x.page) ? html`<b>${x.page.id}</b> ` : ''}${x.page.kind === 'source' || x.page.kind === 'result' ? shortTitle(x.page) : x.page.title}</span></a>`;
  const column = (name, list) => (list.length ? html`<div class="lin-group"><h3>${name}</h3><div class="lin-list">${list.map(row)}</div></div>` : '');
  return html`<section class="lineage" aria-label="Lineage">${column('Rests on', lin.up)}${column('Leads to', lin.down)}</section>`;
}

/* ---------- the trail: the pages opened on the way to this one ---------- */
export function noteTrail(store, path) {
  if (!path || !store.byPath.has(path)) return;
  const trail = app.ui.trail[store.pid] || (app.ui.trail[store.pid] = { list: [], at: -1 });
  const i = trail.list.indexOf(path);
  if (i >= 0) { trail.at = i; return; }
  trail.list = trail.list.slice(0, trail.at + 1).concat([path]).slice(-7);
  trail.at = trail.list.length - 1;
}
export function trailHTML(store, path) {
  const trail = app.ui.trail[store.pid];
  if (!trail) return '';
  const pages = trail.list.map((p) => store.byPath.get(p)).filter(Boolean);
  if (pages.length < 2) return '';
  return html`<nav class="trail" aria-label="The pages you came through"><span class="trail-h">Trail</span>${join(pages.map((p) => html`<a class="trail-i${p.path === path ? ' cur' : ''}" href="${link({ sel: p.path, sub: '' })}" data-sel="${p.path}"${p.path === path ? raw(' aria-current="page"') : ''} data-tip="${pageTip(p)}">${pageGlyph(p, 14)}<span>${label(p)}</span></a>`), html`<span class="trail-sep" aria-hidden="true">${icon('right')}</span>`)}</nav>`;
}

/* ---------- drafts: nothing typed is lost ---------- */
const draftKey = (pid, path) => 'po-draft:' + pid + ':' + path;
/* written at every keystroke: a reload a moment after typing must not lose the last words */
const saveDraft = (key, draft) => {
  const { preview, previewOf, saving, error, conflict, ...kept } = draft; // eslint-disable-line no-unused-vars
  storage.set(key, kept);
};
export function draftOf(store, page, view) {
  const key = draftKey(store.pid, page.path);
  const drafts = app.ui.drafts;
  const proposalId = app.ui.editProposal[page.path] || null;
  if (drafts[key] && (drafts[key].proposal || null) !== proposalId && !isDirty(drafts[key])) delete drafts[key];
  if (!drafts[key]) {
    const now = view.page;
    const proposal = proposalId ? (view.proposals || []).find((p) => p.id === proposalId) : null;
    const base = { title: now.title, summary: now.summary || '', state: now.state || '', body: now.body || '' };
    const start = proposal ? Object.assign({}, base, { body: proposal.body != null ? proposal.body : base.body }, pick(proposal.fields || {}, ['title', 'summary', 'state'])) : base;
    const saved = storage.get(key);
    if (saved && saved.base && (saved.proposal || null) === proposalId && isDirty(saved)) drafts[key] = Object.assign({}, saved, { restored: true });
    else drafts[key] = Object.assign({}, start, { base, base_updated: now.updated, proposal: proposalId, at: null, restored: false });
  }
  return drafts[key];
}
const pick = (object, names) => { const out = {}; names.forEach((n) => { if (object[n] != null) out[n] = object[n]; }); return out; };
export const isDirty = (draft) => !!draft && !!draft.base && (draft.proposal ? true : ['title', 'summary', 'state', 'body'].some((f) => (draft[f] || '') !== (draft.base[f] || '')));
export function touchDraft(store, path, field, value) {
  const key = draftKey(store.pid, path), draft = app.ui.drafts[key];
  if (!draft) return null;
  draft[field] = value;
  draft.at = new Date().toISOString();
  saveDraft(key, draft);
  return draft;
}
export function dropDraft(store, path) {
  const key = draftKey(store.pid, path);
  delete app.ui.drafts[key];
  delete app.ui.editProposal[path];
  storage.remove(key);
}
export const hasDraft = (store, path) => { const key = draftKey(store.pid, path); const d = app.ui.drafts[key] || storage.get(key); return !!(d && d.base && isDirty(d)); };

/* ---------- editor ---------- */
function editorHTML(store, page, view, mode) {
  const draft = draftOf(store, page, view);
  const dirty = isDirty(draft);
  const states = STATES[page.kind];
  const proposed = page.review === 'proposed';
  const what = draft.proposal ? 'Editing a suggested change. Your version is what gets applied.' : proposed ? 'Editing a proposal. Your version is what gets accepted.' : 'Editing. The current version goes to the history.';
  const submit = draft.proposal ? 'Save and accept the change' : proposed ? 'Save and accept' : 'Save';
  const conflict = draft.conflict;
  return html`<form class="editor ed-${mode}" data-form="edit" data-ref="${page.path}">
    <div class="ed-bar"><span class="ed-what">${what}</span>
      <span class="ed-draft" data-draft>${draft.saving ? 'Saving…' : draft.restored ? 'Your unsaved draft was restored' : dirty ? 'Draft kept on this device' : 'No changes yet'}</span>
      ${draft.restored || dirty ? html`<button class="btn btn-quiet" type="button" data-act="draft-discard" data-ref="${page.path}">Discard draft</button>` : ''}
      <button class="btn btn-quiet" type="button" data-act="edit-cancel">Close</button>
      <button class="btn btn-primary" type="submit"${draft.saving ? raw(' disabled') : ''}>${submit} <kbd>Ctrl S</kbd></button></div>
    ${draft.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span><b>Not saved.</b> ${draft.error} Your text is still here.</span></div>` : ''}
    ${conflict ? conflictHTML(draft, conflict) : draft.theirs ? theirsHTML(draft.theirs) : ''}
    <div class="ed-fields"><label><span>Title</span><input id="ed-title" name="title" value="${draft.title}" autocomplete="off"></label>
      <label><span>Summary</span><input id="ed-summary" name="summary" value="${draft.summary}" autocomplete="off" placeholder="One line, shown in lists and on the map"></label>
      ${states ? html`<label class="ed-state"><span>State</span><select id="ed-state" name="state">${states.map((s) => html`<option${s === draft.state ? raw(' selected') : ''}>${s}</option>`)}</select></label>` : ''}</div>
    <div class="ed-split"><div class="ed-pane"><div class="ed-cap">Markdown <span class="quiet">[[H2]] links a page</span></div><textarea id="ed-body" name="body" spellcheck="false" aria-label="Page body, Markdown">${draft.body}</textarea></div>
      <div class="ed-pane"><div class="ed-cap">Preview <span class="quiet" data-prev-state></span></div><div class="ed-prev prose" data-preview="${page.path}">${draft.preview != null ? raw(draft.preview) : html`<p class="quiet">Rendering…</p>`}</div></div></div>
  </form>`;
}
function conflictHTML(draft, conflict) {
  const rows = lineDiff(draft.base.body, conflict.body || '');
  return html`<div class="conflict" role="alert"><p><b>This page changed while you were editing.</b> ${cap(who((conflict.meta || {}).updated_by || conflict.author))} saved a newer version ${ago(conflict.updated)}. Nothing was overwritten, and your text is kept below.</p>
    <details open><summary>What changed in the page since you started</summary>${rows.length ? diffRows(rows) : html`<p class="quiet">Only its fields changed (title, summary or state), not the text.</p>`}</details>
    <div class="review-actions"><button class="btn btn-primary" type="button" data-act="conflict-mine">Save my version over it</button>
      <button class="btn" type="button" data-act="conflict-rebase">Keep editing on top of the new version</button>
      <button class="btn btn-neg-quiet" type="button" data-act="conflict-theirs">Discard my draft, show the new version</button></div></div>`;
}
/** After "keep editing on top of the new version": what that version changed stays in
    sight until the person saves, because their text does not contain it. */
function theirsHTML(theirs) {
  return html`<div class="conflict conflict-kept"><details open data-keep="theirs"><summary>You are editing on top of the version ${who(theirs.by || 'human')} saved ${ago(theirs.at)}. Your text does not contain what it changed:</summary>
    ${theirs.rows.length ? diffRows(theirs.rows) : html`<p class="quiet">Only its fields changed (title, summary or state), not the text.</p>`}</details>
    <p class="quiet">Copy over what you want to keep. Saving replaces that version with your text; it stays in the history.</p></div>`;
}
export const diffRows = (rows) => html`<div class="diff">${rows.map((r) => (r[0] === '…' ? html`<div class="d dgap">⋯</div>` : html`<div class="d d${r[0] === '+' ? 'a' : r[0] === '-' ? 'r' : 'c'}"><span class="d-s">${r[0] === ' ' ? '' : r[0] === '-' ? '−' : '+'}</span><span>${r[1]}</span></div>`))}</div>`;

/* ---------- history ---------- */
function historyHTML(store, page, view) {
  const versions = view.history || [];
  const meta = view.page.meta || {};
  const open = app.ui.histVersion[page.path] || '';
  const entry = open ? store.version(page.path, open) : null;
  return html`<div class="history"><div class="hist-bar"><h2>History</h2><button class="btn btn-quiet" data-act="history" data-ref="${page.path}">Back to the page</button></div>
    <ol class="hist">
      <li class="hist-i${(meta.updated_by || page.author) === 'human' ? ' by-you' : ''}"><div class="hist-h"><span class="hist-when">${dateTime(page.updated)}</span><span class="hist-who">${agentName(meta.updated_by || page.author)}</span><span class="hist-note">The page as it is now</span><span class="hist-cur">current</span></div></li>
      ${versions.map((v) => html`<li class="hist-i${v.author === 'human' ? ' by-you' : ''}${v.stamp === open ? ' open' : ''}"><div class="hist-h"><span class="hist-when">${dateTime(v.stamp)}</span><span class="hist-who">${agentName(v.author)}</span>
          <span class="hist-note">${v.stamp === open ? 'Compared with the page as it is now' : 'An earlier version'}</span><span class="hist-delta">${bytes(v.size)}</span>
          <button class="btn btn-quiet btn-s" data-act="version" data-ref="${page.path}" data-stamp="${v.stamp}" aria-expanded="${tf(v.stamp === open)}">${v.stamp === open ? 'Hide' : 'Compare'}</button>
          <button class="btn btn-quiet btn-s" data-act="restore" data-ref="${page.path}" data-stamp="${v.stamp}">Restore</button></div>
        ${v.stamp === open ? html`<div class="hist-body">${!entry || entry.loading ? html`<p class="quiet">Loading this version…</p>` : entry.error ? html`<p class="note note-neg">${entry.error.message}</p>` : entry.data.diff ? raw(entry.data.diff_html) : html`<p class="quiet">The text of this version is the same as now; only fields changed.</p>`}</div>` : ''}</li>`)}
    </ol>
    ${versions.length ? '' : html`<p class="quiet">This page has not been changed since it was written.</p>`}
    <p class="quiet">Every version is a plain file under <code>history/${page.path}/</code>. Restoring makes a new version; nothing is deleted.</p></div>`;
}

/* ---------- the body ---------- */
function bodyHTML(store, page, view) {
  const proposals = (view.proposals || []).filter((p) => p.body != null && store.pendingFor(page.path).some((x) => x.id === p.id));
  const shown = proposals.find((p) => !app.ui.asDiff[p.id]);
  if (shown) return html`<div class="prose prose-marked" data-prose="${page.path}">${raw(shown.marked_html)}</div>`;
  return html`<div class="prose" data-prose="${page.path}"${page.kind === 'finding' ? raw(' data-numbers="1"') : ''}>${raw(view.html)}</div>`;
}
function attachmentsHTML(store) {
  const files = (store.project.attachments || []);
  if (!files.length) return '';
  return html`<section class="attached"><h3 class="rail-h">Attached files</h3><ul class="att">${files.map((f) => html`<li><a class="file" href="${api.fileUrl(store.pid, 'inputs/' + f.name)}" target="_blank" rel="noopener">${icon('file')}<span>${f.name}</span><span class="quiet">${bytes(f.size)}</span></a></li>`)}</ul></section>`;
}

/** Notes that stand beside the text: suggested edits, the review of a proposal, the number
    check. In the wiki they go to the rail (the margin); elsewhere above the body. */
export function notesHTML(store, page, opt) {
  const entry = store.views.get(page.path);
  const view = entry && entry.data;
  return html`${reviewBar(store, page, opt)}${view && opt.part !== 'page' ? numberCheckHTML(store, view) : ''}`;
}

/* ---------- the page ---------- */
export function panelHTML(store, page, opt) {
  const mode = opt.mode, sub = opt.sub || '';
  const record = page.author === 'system';
  const entry = store.pageView(page.path);
  const view = entry && entry.data;
  const exp = page.kind === 'experiment' && page.id;
  const tools = html`
    ${exp && app.route.view !== 'experiments' ? html`<a class="ibtn" href="${link({ view: 'experiments', sel: page.path, sub: '' })}" data-tip="Plan, code and runs (4)" aria-label="Open the experiment">${icon('flask')}</a>` : ''}
    ${mode !== 'wiki' ? html`<a class="ibtn" href="${link({ view: 'wiki', sel: page.path, sub: '' })}" data-tip="Open in the wiki (2)" aria-label="Open in the wiki">${icon('open')}</a>` : ''}
    ${mode !== 'panel' ? html`<a class="ibtn" href="${link({ view: 'map', sel: page.path, sub: '' })}" data-tip="Show on the map (1)" aria-label="Show on the map">${icon('map')}</a>` : ''}
    ${record ? '' : html`<button class="ibtn${sub === 'edit' ? ' on' : ''}" data-act="edit" data-ref="${page.path}" data-tip="Edit (E)" aria-label="Edit" aria-pressed="${tf(sub === 'edit')}">${icon('pencil')}</button>`}
    <button class="ibtn${sub === 'history' ? ' on' : ''}" data-act="history" data-ref="${page.path}" data-tip="History (H)" aria-label="History" aria-pressed="${tf(sub === 'history')}">${icon('clock')}</button>
    ${page.kind === 'idea' || page.kind === 'result' ? '' : html`<button class="ibtn" data-act="delete-ask" data-ref="${page.path}" data-tip="Delete this page" aria-label="Delete this page">${icon('trash')}</button>`}
    ${mode === 'panel' ? html`<button class="ibtn" data-act="close-panel" data-tip="Close (Esc)" aria-label="Close">${icon('x')}</button>` : ''}`;
  /* one first-level heading per screen: the wiki's is the page; the map and the inbox have their own */
  const titleTag = mode === 'wiki' ? 'h1' : 'h2';
  const head = html`<header class="page-head">
    <div class="page-kicker"><span class="kicker-kind">${pageGlyph(page, 18)}<span>${kindName(page.kind)}${hasId(page) ? html` <b>${page.id}</b>` : ''}</span></span><span class="page-tools">${tools}</span></div>
    <${raw(titleTag)} class="page-title">${page.title}</${raw(titleTag)}>
    ${page.summary && page.summary !== page.title ? html`<p class="page-summary">${page.summary}</p>` : ''}
    <div class="page-status">${reviewTag(page)}${stateTag(page)}<span class="page-by">${byline(page)}</span></div>
    ${page.kind === 'experiment' && page.meta && page.meta.state_reason && (page.state === 'failed' || page.state === 'planned') ? html`<p class="page-reason${page.state === 'planned' ? ' page-reason-back' : ''}">${icon('warn')}${plain(page.meta.state_reason)}</p>` : ''}
    ${app.ui.confirmDelete === page.path ? html`<div class="note note-neg confirm" role="alertdialog" aria-label="Delete this page"><span><b>Delete this page?</b> Its last version stays under <code>history/</code>; pages that link to it keep the link as text.</span>
      <span class="row"><button class="btn btn-neg" data-act="delete" data-ref="${page.path}">Delete</button><button class="btn btn-quiet" data-act="delete-cancel">Cancel</button></span></div>` : ''}
  </header>`;
  let body;
  if (!view) {
    body = entry && entry.error
      ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The text of this page could not be loaded: ${entry.error.message} <button class="linkish" data-act="reload-page" data-ref="${page.path}">Try again</button></span></div>`
      : html`<div class="skeleton" aria-busy="true" aria-label="Loading the page"><i></i><i></i><i></i><i class="short"></i></div>`;
  } else if (sub === 'edit' && !record) body = editorHTML(store, page, view, mode);
  else if (sub === 'history') body = historyHTML(store, page, view);
  else {
    const draftWaiting = !record && hasDraft(store, page.path);
    body = html`${draftWaiting ? html`<div class="note note-draft">${icon('pencil')}<span>You have an unsaved draft of this page on this device.</span><span class="row"><button class="btn btn-s" data-act="edit" data-ref="${page.path}">Resume editing</button><button class="btn btn-quiet btn-s" data-act="draft-discard" data-ref="${page.path}">Discard it</button></span></div>` : ''}
      ${notesHTML(store, page, opt.notes === 'rail' ? { part: 'page' } : {})}
      ${bodyHTML(store, page, view)}
      ${page.kind === 'idea' ? attachmentsHTML(store) : ''}
      ${lineageHTML(store, page)}`;
  }
  const versions = view ? (view.history || []).length + 1 : null;
  const foot = html`<footer class="page-foot"><code>${page.path}</code>${versions ? html`<button class="linkish" data-act="history" data-ref="${page.path}">${plural(versions, 'version')}</button>` : ''}</footer>`;
  return html`<article class="page page-${mode} k-${page.kind} r-${page.review}${sub ? ' sub-' + sub : ''}" data-page="${page.path}">${head}${body}${sub ? '' : foot}</article>`;
}

/* ---------- after the markup is in the page ---------- */
const previews = new Map();
function renderPreview(store, el) {
  const path = el.dataset.preview;
  const key = draftKey(store.pid, path), draft = app.ui.drafts[key];
  if (!draft || draft.previewOf === draft.body) return;
  const text = draft.body;
  const state = $('[data-prev-state]', el.closest('.ed-pane'));
  const before = previews.get(key);
  if (before) before.abort();
  const control = new AbortController();
  previews.set(key, control);
  api.render(text, store.pid, path, control.signal).then((answer) => {
    draft.preview = answer.html;
    draft.previewOf = text;
    const now = $('[data-preview="' + CSS.escape(path) + '"]');
    if (now) { now.innerHTML = answer.html; decorate(store, now); }      // the server's rendering: the one markup inserted as is
    if (state) state.textContent = '';
  }, (error) => {
    if (error && error.name === 'AbortError') return;
    if (state) state.textContent = 'could not be rendered: ' + error.message;
  });
}
export const schedulePreview = debounce((store) => { $$('[data-preview]').forEach((el) => renderPreview(store, el)); }, 280);

/** Links of the server's rendering learn what they point at; numbers are marked. */
function decorate(store, root) {
  $$('a.wikilink[data-page]', root).forEach((a) => {
    const target = store.byPath.get(a.dataset.page);
    if (!target) return;
    a.dataset.tip = pageTip(target);
    a.dataset.sel = target.path;
    a.classList.toggle('r-proposed', target.review === 'proposed');
    a.classList.toggle('r-rejected', target.review === 'rejected');
    a.setAttribute('href', link({ sel: target.path, sub: '' }));
  });
  $$('img', root).forEach((img) => { img.loading = 'lazy'; img.decoding = 'async'; });
  $$('table', root).forEach((table) => { if (!table.parentElement.classList.contains('table-wrap')) { const wrap = document.createElement('div'); wrap.className = 'table-wrap'; table.replaceWith(wrap); wrap.append(table); } });
}
export function afterRender(root) {
  const store = app.store;
  $$('[data-follow]', root).forEach((el) => { el.scrollTop = el.scrollHeight; });
  $$('[data-w]', root).forEach((el) => { el.style.width = Math.max(0, Math.min(100, +el.dataset.w || 0)) + '%'; });
  if (!store || !store.loaded) return;
  $$('[data-prose]', root).forEach((el) => {
    decorate(store, el);
    if (el.dataset.numbers) { const view = (store.views.get(el.dataset.prose) || {}).data; if (view && view.number_check) markNumbers(el, view.number_check, store); }
  });
  $$('.prose-s[data-plan], .hist-body, .ed-prev', root).forEach((el) => decorate(store, el));
  /* The two lines that head a diff of versions name them by a raw stamp. The row above
     already says which version is compared, in the person's own time: they are left out. */
  $$('.hist-body pre.diff .diff-file', root).forEach((el) => {
    const next = el.nextSibling;
    if (next && next.nodeType === 3 && next.nodeValue.charAt(0) === '\n') next.nodeValue = next.nodeValue.slice(1);
    el.remove();
  });
  if ($('[data-preview]', root)) $$('[data-preview]', root).forEach((el) => renderPreview(store, el));
}
