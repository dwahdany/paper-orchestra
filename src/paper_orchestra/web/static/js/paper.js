/* The optional last step: write the wiki up as a paper with a bundled template. */

import { app, link } from './core.js';
import { api } from './api.js';
import { ago, cap, dateTime, duration, html, join, parseStamp, plain, plural, raw, setHTML } from './util.js';
import { icon, pageGlyph } from './vocab.js';
import { engineView } from './shell.js';

const STAGE_NAME = { ingest: 'Reading the wiki', outline: 'Outline', literature: 'Literature', plotting: 'Figures', section_writing: 'Sections', refinement: 'Refinement', finalize: 'Final PDF' };
let pollTimer = 0;

export function loadTemplates() {
  const ui = app.ui.paper;
  if (ui.templates || ui.loadingTemplates) return;
  ui.loadingTemplates = true;
  api.templates().then((data) => { ui.templates = (data.templates || []).map((t) => t.name); if (!ui.template) ui.template = ui.templates[0] || ''; }, (error) => { ui.templatesError = error; })
    .finally(() => { ui.loadingTemplates = false; if (app.route.view === 'paper') app.render('paper', true); });
}

/** A finding, as a chip that leads to its page. */
const findingChip = (page) => html`<a class="chip" href="${link({ view: 'wiki', sel: page.path, sub: '' })}" data-sel="${page.path}" data-tip="${page.title}">${pageGlyph(page, 12)}<b>${page.id || 'Finding'}</b></a>`;

/** The accepted findings a paper would not report, as the server lists them (`left_out`),
    each with the server's reason: every record such a finding is based on belongs to an
    experiment that was rejected or is gone. Said before a write-up starts, not after. */
const leftOut = (paper) => (Array.isArray(paper.left_out) ? paper.left_out.filter((item) => item && item.path) : []);
function leftOutHTML(store, paper) {
  const left = leftOut(paper);
  if (!left.length) return '';
  return html`<section class="paper-left" aria-label="Accepted findings a paper would not report"><h2 class="sec-t">Left out of a paper <span class="quiet">${plural(left.length, 'accepted finding')}</span></h2>
    ${left.map((item) => html`<div class="pn-row" data-left="${item.id || item.path}">${findingChip(store.byPath.get(item.path) || { path: item.path, id: item.id, title: item.title || '', kind: 'finding', review: 'accepted' })}<p><b>${item.title || ''}</b>${item.title ? ' ' : ''}<span class="pl-why">${cap(plain(String(item.reason || 'It would not be reported.')).replace(/[.\s]+$/, ''))}.</span></p></div>`)}
    <p class="quiet">A paper does not report such a finding. Accept the experiment again, or base the finding on a record that stands, and it is reported.</p></section>`;
}

/** What the number check says about the findings a paper would be written from, in the
    server's own sentence for each (`number_report` of the finding's page). Only the findings
    that have something to say are listed: a number that was found nowhere, or only outside
    the records the finding is based on. The reports are read when this view is shown. */
function numbersHTML(store, findings) {
  if (!findings.length) return '';
  const rows = findings.map((page) => { const entry = store.pageView(page.path); return { page, view: entry && entry.data, error: entry && entry.error }; });
  const waiting = rows.filter((r) => !r.view && !r.error).length;
  const failed = rows.filter((r) => r.error || (r.view && !r.view.number_report));
  const told = rows.filter((r) => r.view && r.view.number_report && ((r.view.number_report.unverified || []).length || (r.view.number_report.elsewhere || []).length));
  const clean = rows.length - waiting - failed.length - told.length;
  const name = findingChip;
  return html`<section class="paper-numbers" aria-label="The number check of the accepted findings"><h2 class="sec-t">Numbers in the accepted findings <span class="quiet">as the number check reports them</span></h2>
    ${told.map((r) => { const report = r.view.number_report; return html`<div class="pn-row" data-finding="${r.page.id || r.page.path}">${name(r.page)}<div><p>${report.summary}</p>
      ${(report.unverified || []).length ? html`<p class="pn-flag">${icon('warn')}<span><b>Not found:</b> ${join(report.unverified.map((n) => html`<b class="numcheck-n">${n}</b>`), ', ')}. In what a paper is written from, a number that was found nowhere is replaced by “[a number not on record]”.</span></p>` : ''}
      ${(report.elsewhere || []).length ? html`<p class="quiet"><b>Found only outside the records this finding is based on:</b> ${report.elsewhere.join(', ')}</p>` : ''}</div></div>`; })}
    ${failed.map((r) => html`<div class="pn-row">${name(r.page)}<p class="quiet">The numbers could not be checked${r.error ? ': ' + r.error.message : r.view.number_report_error ? ': ' + plain(r.view.number_report_error) : '.'}</p></div>`)}
    ${waiting ? html`<p class="quiet">Reading the number check of ${plural(waiting, 'finding')}…</p>` : ''}
    ${clean ? html`<p class="quiet pn-clean">${told.length || failed.length ? 'In the other ' + plural(clean, 'finding') : clean === 1 ? 'In the one accepted finding' : 'In all ' + clean + ' accepted findings'}, every number that was looked for was found in the records ${clean === 1 ? 'it is' : 'they are'} based on. Short and round numbers are not checked.</p>` : ''}
  </section>`;
}
/** What the last write-up said when it started: what it was written from, which accepted
    findings were left out and why, and which numbers were taken out (the server's sentence). */
function startedHTML(store) {
  const ev = store.activity.slice().reverse().find((x) => x.type === 'paper.started');
  if (!ev || !ev.summary) return '';
  return html`<p class="paper-started" data-started><b>${dateTime(ev.ts) ? 'Started ' + dateTime(ev.ts) + '.' : 'When it started:'}</b> ${plain(ev.summary)}</p>`;
}

/** A write-up that ended before it could record why in paper/state.json: it was refused
    (no finding that can be reported, a research cutoff that cannot be read) or its worker
    died. The worker's own last words, from worker.log, say which. */
function endedHTML(store, state, running) {
  const exit = (store.project.worker || {}).last_exit;
  if (running || !exit || exit.kind !== 'write' || exit.stopped || !exit.code) return '';
  /* a run that got as far as its own state says how it ended there */
  if (state && state.updated_at && (parseStamp(state.updated_at) || 0) >= (parseStamp(exit.at) || 0) - 3000 && state.status !== 'running') return '';
  const log = store.workerLog && typeof store.workerLog.text === 'string' ? store.workerLog.text : null;
  const lines = log == null ? [] : log.trimEnd().split('\n');
  const from = lines.map((line) => /^\[paper-orchestra\] .* starting: /.test(line)).lastIndexOf(true);
  const words = lines.slice(from + 1).join(' ').replace(/\s+/g, ' ').replace(/^Error:\s*/, '').trim().slice(0, 900);
  return html`<div class="note note-neg paper-ended" role="alert" data-ended>${icon('warn')}<span><b>The last write-up ended before it wrote anything${dateTime(exit.at) ? ' (' + dateTime(exit.at) + ')' : ''}.</b> ${words ? plain(words) : log == null ? 'Reading what its worker wrote…' : 'Its worker wrote no reason. Its log is worker.log in the project folder.'}</span></div>`;
}

export const paperView = {
  render(root) {
    const store = app.store, ui = app.ui.paper, e = engineView(store);
    loadTemplates();
    if (!store.paper) store.loadPaper();
    const paper = store.paper || {};
    const state = paper.state || null;
    const running = !!paper.running || e.mode === 'writing';
    clearInterval(pollTimer);
    if (running) pollTimer = setInterval(() => { if (app.store === store && app.route.view === 'paper') store.loadPaper(); else clearInterval(pollTimer); }, 3000);
    const busy = !!(store.project.worker || {}).state && store.project.worker.state !== 'idle';
    const accepted = store.pages.filter((p) => p.kind === 'finding' && p.review === 'accepted');
    const findings = accepted.length, left = leftOut(paper);
    const worker = store.project.worker || {};
    const ran = store.experiments.filter((x) => x.review === 'accepted' && x.state === 'succeeded').length;
    const attached = (store.project.attachments || []).length;
    const stages = state && state.stages ? Object.keys(state.stages) : [];
    const mark = (status) => (status === 'completed' ? icon('check') : status === 'failed' ? icon('x') : status === 'running' ? raw('<span class="live-dot" aria-hidden="true"></span>') : status === 'interrupted' ? icon('pause') : icon('ring'));
    setHTML(root, html`<div class="paper"><div class="paper-wrap">
      <header class="view-head"><h1>Write it up <span class="quiet">optional</span></h1>
        <p>A draft paper is written from what the project holds, with one of the bundled templates. Proposals you have not accepted are left out, and nothing in the wiki is changed.</p></header>
      <ul class="paper-from" aria-label="What the paper is written from">
        <li><b>${findings}</b><span class="pf-what">accepted ${findings === 1 ? 'finding' : 'findings'}</span><span class="pf-how">What the results mean. A paper needs at least one${left.length ? html` that can be reported: <b class="pf-left">${left.length === findings && findings > 1 ? 'all ' + left.length + ' are' : left.length === 1 && findings === 1 ? 'this one is' : left.length + ' of them ' + (left.length === 1 ? 'is' : 'are')} left out</b> (below).` : '.'}</span></li>
        <li><b>${ran}</b><span class="pf-what">${ran === 1 ? 'experiment' : 'experiments'} with a run that succeeded</span><span class="pf-how">Their metrics and figures go in as the runs wrote them.</span></li>
        <li><b>${attached}</b><span class="pf-what">${attached === 1 ? 'file' : 'files'} you attached</span><span class="pf-how">Text and tables go in as they are; images become figures marked “Provided by the researcher”. With these and a finding, a paper can be written without any run.</span></li></ul>
      ${paper.missing === 'results' ? html`<div class="note note-warn paper-gap">${icon('warn')}<span><b>Not ready: there are no results to report.</b> No accepted experiment has a run that succeeded, and no text file, table or image is attached to the project. <a href="${link({ view: 'experiments', sel: '', sub: '' })}">Approve an experiment and run it</a>; then a finding about its results must be accepted.</span></div>`
        : paper.missing === 'finding' && findings ? html`<div class="note note-warn paper-gap">${icon('warn')}<span><b>Not ready: none of the accepted findings can be reported.</b> ${left.length ? 'They are listed below, each with the reason.'
          : paper.missing_detail ? cap(plain(paper.missing_detail).replace(/[.\s]+$/, '')) + '. Accept the experiment again, or base a finding on a record that stands.'
            : (findings === 1 ? 'The one accepted finding rests' : 'Each of the ' + findings + ' accepted findings rests') + ' only on records of an experiment that was rejected or is gone, and a paper does not report such a finding. Accept the experiment again, or base a finding on a record that stands.'}</span></div>`
        : paper.missing === 'finding' ? html`<div class="note note-warn paper-gap">${icon('warn')}<span><b>Not ready: no finding is accepted.</b> The results are there, but a paper reports what they mean. <a href="${link({ view: 'inbox', sel: '', sub: '' })}">Accept a finding in the inbox</a>, or write one yourself as a new page.</span></div>` : ''}
      ${paper.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The state of the paper could not be read: ${paper.error.message}</span></div>` : ''}
      <form class="paper-form" data-form="paper">
        <label><span>Template</span><select id="paper-template" name="template" data-input="paper-template">${(ui.templates || []).map((t) => html`<option${t === ui.template ? raw(' selected') : ''}>${t}</option>`)}${ui.templates ? '' : html`<option value="">Loading…</option>`}</select></label>
        <label><span>Model</span><input id="paper-model" name="model" data-input="paper-model" value="${ui.model || ''}" placeholder="The project's writing model" autocomplete="off"></label>
        <button class="btn btn-primary" type="submit"${running || busy || !ui.template || paper.missing ? raw(' disabled') : ''}>${running ? 'Writing…' : state ? 'Write it again' : 'Start the write-up'}</button>
        ${running ? html`<button class="btn" type="button" data-act="stop">${icon('stop')}Stop</button>` : ''}
        ${busy && !running ? html`<span class="quiet">${worker.verified === false ? 'The project\'s files name a worker this server cannot verify; the write-up can start once it has ended or is forgotten (see the overview).' : 'The agents are at work; the write-up can start when they have stopped.'}</span>` : ''}
      </form>
      ${ui.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span><b>Not started.</b> ${ui.error}</span></div>` : ''}
      ${endedHTML(store, state, running)}
      ${ui.templatesError ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The templates could not be listed: ${ui.templatesError.message}</span></div>` : ''}
      ${leftOutHTML(store, paper)}
      ${numbersHTML(store, accepted)}
      ${state ? html`<section class="paper-state"><h2 class="sec-t">${running ? 'In progress' : state.status === 'succeeded' ? 'Finished' : state.status === 'failed' ? 'Failed' : state.status === 'interrupted' ? 'Interrupted' : 'Last run'} <span class="quiet">${state.updated_at ? ago(state.updated_at) : ''}</span></h2>
          ${startedHTML(store)}
          ${state.error ? html`<p class="note note-neg">${icon('warn')}<span>${plain(state.error)}</span></p>` : ''}
          <ol class="paper-stages">${stages.map((name) => { const s = state.stages[name] || {}; return html`<li class="ps ps-${String(s.status || 'pending').replace(/[^a-z]/g, '')}"><span class="ps-m">${mark(s.status)}</span><span class="ps-n">${STAGE_NAME[name] || name}</span><span class="ps-s">${s.status || 'pending'}${s.reused ? ' (reused)' : ''}${s.attempts > 1 ? ' · ' + s.attempts + ' attempts' : ''}</span><span class="ps-t n">${s.seconds ? duration(s.seconds) : ''}</span>${s.error ? html`<span class="ps-e">${s.error}</span>` : ''}</li>`; })}</ol></section>`
        : paper.exists === false ? html`<p class="quiet paper-none">No paper has been written for this project.</p>` : ''}
      ${paper.final_pdf ? html`<section class="paper-pdf"><h2 class="sec-t">The paper <a class="btn btn-s tabs-r" href="${api.fileUrl(store.pid, paper.final_pdf, true)}">Download the PDF</a></h2>
        <iframe class="pdf" title="The paper, as a PDF" src="${api.fileUrl(store.pid, paper.final_pdf)}"></iframe></section>` : ''}
    </div></div>`);
  },
};
