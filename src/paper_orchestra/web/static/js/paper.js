/* The optional last step: write the wiki up as a paper with a bundled template. */

import { app, link } from './core.js';
import { api } from './api.js';
import { ago, duration, html, raw, setHTML } from './util.js';
import { icon } from './vocab.js';
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
    const busy = (store.project.worker || {}).state && store.project.worker.state !== 'idle';
    const accepted = store.pages.filter((p) => p.review === 'accepted').length, findings = store.pages.filter((p) => p.kind === 'finding' && p.review === 'accepted').length;
    const stages = state && state.stages ? Object.keys(state.stages) : [];
    const mark = (status) => (status === 'completed' ? icon('check') : status === 'failed' ? icon('x') : status === 'running' ? raw('<span class="live-dot" aria-hidden="true"></span>') : status === 'interrupted' ? icon('pause') : icon('ring'));
    setHTML(root, html`<div class="paper"><div class="paper-wrap">
      <header class="view-head"><h1>Write it up <span class="quiet">optional</span></h1>
        <p>The paper pipeline writes a draft from what the project holds: the runs that succeeded, the accepted findings that say what they mean (${findings} of ${accepted} accepted pages), and what you attached. Attached text files and tables go into the materials as they are; attached images become figures marked “Provided by the researcher”. A project with attached results and an accepted finding can be written up without any run. Proposals you have not accepted are left out. Nothing in the wiki is changed.</p></header>
      ${paper.missing === 'results' ? html`<div class="note note-warn paper-gap">${icon('warn')}<span><b>Not ready: there are no results to report.</b> No accepted experiment has a run that succeeded, and no text file, table or image is attached to the project. <a href="${link({ view: 'experiments', sel: '', sub: '' })}">Approve an experiment and run it</a>; then a finding about its results must be accepted.</span></div>`
        : paper.missing === 'finding' ? html`<div class="note note-warn paper-gap">${icon('warn')}<span><b>Not ready: no finding is accepted.</b> The results are there, but a paper reports what they mean. <a href="${link({ view: 'inbox', sel: '', sub: '' })}">Accept a finding in the inbox</a>, or write one yourself as a new page.</span></div>` : ''}
      ${paper.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The state of the paper could not be read: ${paper.error.message}</span></div>` : ''}
      <form class="paper-form" data-form="paper">
        <label><span>Template</span><select id="paper-template" name="template" data-input="paper-template">${(ui.templates || []).map((t) => html`<option${t === ui.template ? raw(' selected') : ''}>${t}</option>`)}${ui.templates ? '' : html`<option value="">Loading…</option>`}</select></label>
        <label><span>Model</span><input id="paper-model" name="model" data-input="paper-model" value="${ui.model || ''}" placeholder="The project's writing model" autocomplete="off"></label>
        <button class="btn btn-primary" type="submit"${running || busy || !ui.template || paper.missing ? raw(' disabled') : ''}>${running ? 'Writing…' : state ? 'Write it again' : 'Start the write-up'}</button>
        ${running ? html`<button class="btn" type="button" data-act="stop">${icon('stop')}Stop</button>` : ''}
        ${busy && !running ? html`<span class="quiet">The agents are at work; the write-up can start when they have stopped.</span>` : ''}
      </form>
      ${ui.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span><b>Not started.</b> ${ui.error}</span></div>` : ''}
      ${ui.templatesError ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The templates could not be listed: ${ui.templatesError.message}</span></div>` : ''}
      ${state ? html`<section class="paper-state"><h2 class="sec-t">${running ? 'In progress' : state.status === 'succeeded' ? 'Finished' : state.status === 'failed' ? 'Failed' : state.status === 'interrupted' ? 'Interrupted' : 'Last run'} <span class="quiet">${state.updated_at ? ago(state.updated_at) : ''}</span></h2>
          ${state.error ? html`<p class="note note-neg">${icon('warn')}<span>${state.error}</span></p>` : ''}
          <ol class="paper-stages">${stages.map((name) => { const s = state.stages[name] || {}; return html`<li class="ps ps-${String(s.status || 'pending').replace(/[^a-z]/g, '')}"><span class="ps-m">${mark(s.status)}</span><span class="ps-n">${STAGE_NAME[name] || name}</span><span class="ps-s">${s.status || 'pending'}${s.reused ? ' (reused)' : ''}${s.attempts > 1 ? ' · ' + s.attempts + ' attempts' : ''}</span><span class="ps-t n">${s.seconds ? duration(s.seconds) : ''}</span>${s.error ? html`<span class="ps-e">${s.error}</span>` : ''}</li>`; })}</ol></section>`
        : paper.exists === false ? html`<p class="quiet paper-none">No paper has been written for this project.</p>` : ''}
      ${paper.final_pdf ? html`<section class="paper-pdf"><h2 class="sec-t">The paper <a class="btn btn-s tabs-r" href="${api.fileUrl(store.pid, paper.final_pdf, true)}">Download the PDF</a></h2>
        <iframe class="pdf" title="The paper, as a PDF" src="${api.fileUrl(store.pid, paper.final_pdf)}"></iframe></section>` : ''}
    </div></div>`);
  },
};
