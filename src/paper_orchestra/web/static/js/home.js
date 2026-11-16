/* The screens outside a project: the composer that starts one, the library, the setup check. */

import { app } from './core.js';
import { api } from './api.js';
import { ago, bytes, html, money, plural, raw, setHTML, storage } from './util.js';
import { STAGES, icon, phaseName } from './vocab.js';
import { stripMapSmall } from './overview.js';

/* ---------- start: one box for the idea ---------- */
const EXAMPLE = 'Learning-rate warmup is assumed to matter most for large batches. I suspect that with adaptive optimisers it is small batches that need it, because early second-moment estimates are noisy.';
const FILE_NOTE = { md: 'notes · read as text', txt: 'text', csv: 'table · read as text', tsv: 'table · read as text', json: 'data · read as text', png: 'image · shown to the model', jpg: 'image · shown to the model', jpeg: 'image · shown to the model', gif: 'image · shown to the model', webp: 'image · shown to the model', pdf: 'paper · read as a PDF', ipynb: 'notebook · read as text', py: 'code · read as text', tex: 'LaTeX · read as text', bib: 'references · read as text' };
const REVIEW_KINDS = [['topic', 'Topics'], ['question', 'Questions'], ['hypothesis', 'Hypotheses'], ['experiment', 'Experiments'], ['finding', 'Findings']];
const COMPOSER_KEY = 'po-composer';

/** The composer's state. The text and the options survive a reload; files cannot (the
    browser does not let a page keep them), and the composer says so when that happens. */
export function composer() {
  if (!app.ui.start) {
    const saved = storage.get(COMPOSER_KEY, {}) || {};
    app.ui.start = {
      text: saved.text || '', files: [], adv: !!saved.adv, lostFiles: saved.fileCount || 0,
      title: saved.title || '', effort: saved.effort || '', cutoff: saved.cutoff || '', compute: saved.compute || '',
      model_usd: saved.model_usd || '', exp_minutes: saved.exp_minutes || '', exp_usd: saved.exp_usd || '',
      auto: saved.auto || {}, uploading: false, progress: 0, error: null, request: null,
    };
  }
  return app.ui.start;
}
export function keepComposer() {
  const s = composer();
  storage.set(COMPOSER_KEY, { text: s.text, adv: s.adv, title: s.title, effort: s.effort, cutoff: s.cutoff, compute: s.compute, model_usd: s.model_usd, exp_minutes: s.exp_minutes, exp_usd: s.exp_usd, auto: s.auto, fileCount: s.files.length });
}
export function clearComposer() { app.ui.start = null; storage.remove(COMPOSER_KEY); }
export const composerReady = (s) => (s.text.trim().length > 0 || s.files.length > 0) && !s.uploading;
export const totalSize = (s) => s.files.reduce((a, f) => a + f.size, 0);
/** The settings the person changed, as the partial object the API merges into its defaults. */
export function composerSettings(s) {
  const settings = {};
  if (s.effort) settings.models = { effort: s.effort };
  if (s.cutoff.trim()) settings.research_cutoff = s.cutoff.trim();
  if (s.compute) settings.compute = s.compute;
  const limits = {};
  if (String(s.model_usd).trim() !== '') limits.model_usd = Number(String(s.model_usd).replace(/[$,\s]/g, ''));
  if (String(s.exp_minutes).trim() !== '') limits.experiment_seconds = Math.round(Number(String(s.exp_minutes).replace(/[^\d.]/g, '')) * 60);
  if (String(s.exp_usd).trim() !== '') limits.experiment_usd = Number(String(s.exp_usd).replace(/[$,\s]/g, ''));
  if (Object.keys(limits).length) settings.limits = limits;
  const review = {};
  Object.keys(s.auto || {}).forEach((kind) => { if (s.auto[kind]) review[kind] = 'auto'; });
  if (Object.keys(review).length) settings.review = review;
  return settings;
}

const ext = (name) => (String(name).split('.').length > 1 ? String(name).split('.').pop().toLowerCase().slice(0, 5) : 'file');

export const startView = {
  render(root, route) {
    const s = composer();
    if (route.q && route.q.adv) s.adv = true;
    const ready = composerReady(s);
    const n = s.files.length;
    const typed = Number(String(s.model_usd).replace(/[$,\s]/g, ''));
    const limit = String(s.model_usd).trim() !== '' && typed > 0 ? '$' + typed.toFixed(2) : '$25.00';
    setHTML(root, html`<div class="start"><div class="start-col">
      <h1>What do you want to find out?</h1>
      <p class="start-lede">Describe the idea in a few sentences and add anything you already have. Agents research it and fill a wiki with sources, questions, hypotheses and experiments. They only suggest: nothing is accepted, and nothing runs, without you.</p>
      <form class="composer${s.uploading ? ' busy' : ''}" data-form="start" data-drop>
        <label class="sr" for="idea-text">Your idea</label><textarea id="idea-text" name="text" data-input="start-text" rows="5" placeholder="What do you suspect, and why? What would convince you?"${s.uploading ? raw(' readonly') : ''}>${s.text}</textarea>
        ${n ? html`<ul class="files" aria-label="Attached files">${s.files.map((f, i) => html`<li class="filechip"><span class="ft">${ext(f.name)}</span><span class="fn" data-tip="${f.name}">${f.name}</span><span class="fs">${bytes(f.size)} · ${FILE_NOTE[ext(f.name)] || 'file'}</span>
          <button class="ibtn" type="button" data-act="file-del" data-i="${i}" aria-label="Remove ${f.name}"${s.uploading ? raw(' disabled') : ''}>${icon('x')}</button></li>`)}</ul>` : ''}
        ${s.lostFiles && !n ? html`<p class="composer-note">${icon('warn')}Your text was kept across the reload. The ${plural(s.lostFiles, 'file')} you had attached could not be: a page cannot keep files. Add ${s.lostFiles === 1 ? 'it' : 'them'} again. <button class="linkish" type="button" data-act="lost-ok">Dismiss</button></p>` : ''}
        ${s.uploading ? html`<div class="upload" role="status"><progress id="upload-bar" max="1" value="${s.progress}" aria-label="Upload progress"></progress><span id="upload-say">${s.progress >= 1 ? 'Sent. The project is being created…' : 'Sending ' + (n ? plural(n, 'file') + ', ' + bytes(totalSize(s)) : 'the idea') + '…'}</span><button class="btn btn-quiet btn-s" type="button" data-act="upload-cancel">Cancel</button></div>` : ''}
        ${s.error ? html`<div class="note note-neg composer-error" role="alert">${icon('warn')}<span><b>The project was not created.</b> ${s.error} Your text and files are still here.</span></div>` : ''}
        <div class="composer-bar"><button class="btn btn-quiet" type="button" data-act="file-add"${s.uploading ? raw(' disabled') : ''}>${icon('plus')}Add files</button><span class="quiet drop-hint">or drop notes, data, figures, papers, code, a folder</span>
          <input type="file" id="file-input" class="sr" multiple tabindex="-1" aria-label="Files to attach">
          <span class="composer-go">${s.text || n ? '' : html`<button class="btn btn-quiet" type="button" data-act="start-example">Use an example</button>`}
            <button class="btn btn-primary btn-l" type="submit"${ready ? '' : raw(' disabled')}>Start research ${icon('right')}</button></span></div>
        <div class="drop-veil" aria-hidden="true">Drop to attach</div></form>
      <p class="start-est">${ready ? 'First it reads ' + (n ? 'your ' + plural(n, 'file') + ' and ' : '') + 'the idea, then looks for prior work, and pauses at the first checkpoint. It stops when ' + limit + ' has been spent on models.' : 'A few sentences are enough. You can edit the idea later; it stays yours and no agent changes it.'}</p>
      <details class="adv"${s.adv ? raw(' open') : ''}><summary data-act="adv">Options <span class="quiet">title, model effort, limits, what needs your review</span></summary><div class="adv-grid">
        <label><span>Title</span><input id="opt-title" data-input="opt" data-name="title" value="${s.title}" placeholder="Named from the idea if left empty" autocomplete="off"></label>
        <label><span>Model effort</span><select id="opt-effort" data-input="opt" data-name="effort">${[['', 'As the project default'], ['low', 'low'], ['medium', 'medium'], ['high', 'high'], ['xhigh', 'extra high'], ['max', 'max']].map(([v, t]) => html`<option value="${v}"${s.effort === v ? raw(' selected') : ''}>${t}</option>`)}</select></label>
        <label><span>Only literature before</span><input id="opt-cutoff" data-input="opt" data-name="cutoff" value="${s.cutoff}" placeholder="No cutoff (or 2024-06)" autocomplete="off"></label>
        <label><span>Experiments run on</span><select id="opt-compute" data-input="opt" data-name="compute">${[['', 'As the project default'], ['local', 'This machine (local CPU)'], ['modal', 'Modal sandboxes']].map(([v, t]) => html`<option value="${v}"${s.compute === v ? raw(' selected') : ''}>${t}</option>`)}</select></label>
        <label><span>Spend limit for models</span><input id="opt-model-usd" data-input="opt" data-name="model_usd" value="${s.model_usd}" placeholder="25.00 dollars" inputmode="decimal" autocomplete="off"></label>
        <label><span>Limit per experiment run</span><input id="opt-exp-min" data-input="opt" data-name="exp_minutes" value="${s.exp_minutes}" placeholder="15 minutes" inputmode="decimal" autocomplete="off"></label>
        <label><span>Spend limit for experiments</span><input id="opt-exp-usd" data-input="opt" data-name="exp_usd" value="${s.exp_usd}" placeholder="5.00 dollars" inputmode="decimal" autocomplete="off"></label>
        <fieldset class="adv-review"><legend>Needs your review before it counts</legend>${REVIEW_KINDS.map(([kind, name]) => html`<label class="tick"><input type="checkbox" data-input="opt-review" data-kind="${kind}"${s.auto[kind] ? '' : raw(' checked')}> ${name}</label>`)}
          <label class="tick"><input type="checkbox" disabled> Sources <span class="quiet">verified by code, accepted on creation</span></label></fieldset></div></details>
      <div class="start-route" aria-label="What happens next"><div class="smap smap-route">
        ${[['Idea', 'yours'], ['Sources', 'verified papers'], ['Topics', 'what is known'], ['Questions', 'what is open'], ['Hypotheses', 'you accept or reject'], ['Experiments', 'you approve each run'], ['Results', 'written by code'], ['Findings', 'traced to results']].map((x, i) => html`<div class="smap-st${i ? ' none' : ' here'}"><div class="smap-dot"></div><div class="smap-name">${x[0]}</div><div class="smap-hint">${x[1]}</div></div>`)}</div></div>
    </div></div>`);
  },
};
export { EXAMPLE };

/* ---------- library ---------- */
const details = new Map();   // project id -> counts, fetched for the rows on screen
export function loadLibrary(force) {
  const lib = app.ui.library || (app.ui.library = { projects: null, error: null, loading: false, at: 0 });
  if (lib.loading || (!force && lib.projects && Date.now() - lib.at < 4000)) return;
  lib.loading = true;
  api.projects().then((data) => {
    lib.projects = data.projects || [];
    lib.error = null;
    lib.at = Date.now();
    lib.projects.slice(0, 16).forEach((row) => {
      const known = details.get(row.id);
      if (known && known.updated === row.updated) return;
      details.set(row.id, { updated: row.updated, counts: known ? known.counts : null });
      api.project(row.id).then((p) => { details.set(row.id, { updated: row.updated, counts: p.counts }); if (app.route.name === 'library') app.render('library', true); }, () => {});
    });
  }, (error) => { lib.error = error; }).finally(() => {
    lib.loading = false;
    const now = JSON.stringify([lib.projects, lib.error && lib.error.message]);   // looked at again every few seconds: drawn again only when it changed
    if (now !== lib.drawn && app.route.name === 'library') { lib.drawn = now; app.render('library', true); }
  });
}

function rowState(row, waiting) {
  const running = row.worker && row.worker !== 'idle';
  if (running || row.status === 'running') return ['live', 'Round ' + (row.round || 1) + ' · ' + phaseName(row.phase), 'running now'];
  if (row.status === 'failed') return ['neg', 'Round ' + (row.round || 1) + ' · ' + phaseName(row.next_phase || row.phase) + ' failed', 'needs you'];
  if (row.status === 'stopped') return [row.error ? 'neg' : 'pause', 'Round ' + (row.round || 1) + ' · stopped', row.error ? 'the agents stopped unexpectedly' : 'stopped by you'];
  if (row.status === 'checkpoint') { const n = waiting == null ? row.pending : waiting; return ['pause', 'Round ' + (row.round || 1) + ' · checkpoint', n ? n + ' waiting for you' : 'ready to continue']; }
  return ['idle', 'Not started', 'ready'];
}

export const libraryView = {
  render(root) {
    loadLibrary(false);
    const lib = app.ui.library;
    const head = html`<header class="view-head lib-head"><div><h1>Projects</h1><p>Each project is a folder of plain files${app.ui.health && app.ui.health.workspace ? html` in <code>${app.ui.health.workspace}</code>` : ''}. Open one to see its map.</p></div>
      <a class="btn btn-primary" href="#/start">${icon('plus')}New project</a></header>`;
    let body;
    if (!lib.projects && lib.error) body = html`<div class="note note-neg" role="alert">${icon('warn')}<span>The projects could not be listed: ${lib.error.message} <button class="linkish" data-act="reload-library">Try again</button></span></div>`;
    else if (!lib.projects) body = html`<div class="skeleton" aria-busy="true" aria-label="Loading the projects"><i></i><i></i><i class="short"></i></div>`;
    else if (!lib.projects.length) body = html`<div class="blank lib-blank"><h2>No project yet</h2><p>A project starts from an idea: a few sentences, and whatever notes, data or papers you already have.</p><p><a class="btn btn-primary btn-l" href="#/start">Start the first one ${icon('right')}</a></p></div>`;
    else {
      body = html`<div class="lib-list">${lib.projects.map((row) => {
        const d = details.get(row.id), counts = d && d.counts, st = rowState(row, counts ? (counts.pending || {}).total : null);
        const by = counts ? counts.by_kind || {} : null;
        const say = row.status === 'failed' || (row.status === 'stopped' && row.error) ? row.error : row.checkpoint && row.checkpoint.summary ? row.checkpoint.summary : row.status === 'idle' ? 'The idea is in place; research has not started.' : '';
        return html`<a class="lib-row" href="#/p/${encodeURIComponent(row.id)}/map"><div class="lib-main"><h2>${row.title || row.dir}</h2><p>${say}</p></div>
          <div class="lib-map">${stripMapSmall(by, counts ? (counts.pending || {}).by_kind : null)}<span class="lib-counts">${by ? (STAGES.slice(1).some((k) => by[k]) ? (by.source || 0) + ' sources · ' + (by.hypothesis || 0) + ' hypotheses · ' + (by.experiment || 0) + ' experiments · ' + (by.finding || 0) + ' findings' : 'idea only') : ' '}</span></div>
          <div class="lib-st st-${st[0]}"><span class="lib-st1">${st[0] === 'live' ? html`<span class="live-dot"></span>` : ''}${st[1]}</span><span class="lib-st2">${st[2]}</span></div>
          <div class="lib-num"><span>${money((row.spent || {}).model_usd || 0)}</span><span class="quiet">${ago(row.updated)}</span></div></a>`;
      })}</div>`;
    }
    setHTML(root, html`<div class="library"><div class="lib-wrap">${head}${body}</div></div>`);
  },
};

/* ---------- setup ---------- */
export function loadDoctor(force) {
  const doc = app.ui.doctor || (app.ui.doctor = { data: null, error: null, loading: false, at: 0 });
  if (doc.loading || (!force && doc.data)) return;
  doc.loading = true;
  api.doctor().then((data) => { doc.data = data; doc.error = null; doc.at = Date.now(); }, (error) => { doc.error = error; })
    .finally(() => { doc.loading = false; if (app.route.name === 'setup') app.render('setup', true); });
}
const MARKS = { ok: ['check', 'Usable'], warn: ['warn', 'Works, limited'], missing: ['x', 'Not set up'], optional: ['minus', 'Optional, not set up'] };
const item = (status, name, note, detail, fix) => html`<div class="doc-i s-${status}"><span class="doc-m">${icon(MARKS[status][0])}</span><div class="doc-b"><h3>${name}${note ? html`<span class="quiet">${note}</span>` : ''}</h3>
  ${String(detail || '').split('\n').filter((l) => l.trim()).map((l) => html`<p>${l}</p>`)}
  ${fix ? html`<div class="doc-fix"><code>${fix}</code><button class="ibtn" data-act="copy" data-text="${fix}" data-tip="Copy" aria-label="Copy the command">${icon('copy')}</button></div>` : ''}</div><span class="doc-s">${MARKS[status][1]}</span></div>`;

export const setupView = {
  render(root) {
    loadDoctor(false);
    const doc = app.ui.doctor, d = doc.data;
    const head = html`<header class="view-head lib-head"><div><h1>Setup check</h1><p>What this machine can use. Keys are read from the environment and never shown, stored or sent anywhere but their own service.</p></div>
      <button class="btn" data-act="recheck"${doc.loading ? raw(' disabled') : ''}>${doc.loading ? 'Checking…' : 'Check again'}</button></header>`;
    if (!d) {
      setHTML(root, html`<div class="setup"><div class="setup-wrap">${head}${doc.error ? html`<div class="note note-neg" role="alert">${icon('warn')}<span>The check could not be run: ${doc.error.message}</span></div>` : html`<div class="skeleton" aria-busy="true" aria-label="Checking"><i></i><i></i><i class="short"></i></div>`}</div></div>`);
      return;
    }
    const prov = d.provider || {}, local = (d.compute || {}).local || {}, modal = (d.compute || {}).modal || {};
    const models = html`<section class="doc-g"><h2>Models</h2>
      ${item(prov.usable ? ((prov.warnings || []).length ? 'warn' : 'ok') : 'missing', cap(prov.provider || 'Provider') + ' · ' + (d.model || prov.model || ''), prov.credential_source ? 'credentials from ' + prov.credential_source : '', prov.usable ? ((prov.warnings || []).join('\n') || 'The research agents can use this model.') : prov.error || 'No credentials were found.', prov.usable ? '' : 'export ANTHROPIC_API_KEY=…')}
      ${(d.other_providers || []).map((o) => item(o.credential_source ? 'ok' : 'optional', cap(o.provider) === 'Openai' ? 'OpenAI' : cap(o.provider), o.sdk_installed ? 'package installed' : 'package not installed', o.credential_source ? 'Credentials from ' + o.credential_source + '. Research agents need a Claude model; other providers serve the paper pipeline only.' : 'No credentials. Only needed if you write papers with this provider.', ''))}</section>`;
    const compute = html`<section class="doc-g"><h2>Where experiments run</h2>
      ${item(local.available ? ((local.jobs || {}).network_isolation === false ? 'warn' : 'ok') : 'missing', 'This machine', local.python ? local.python : '', ((local.jobs || {}).summary || '') + ((local.problems || []).length ? '\n' + local.problems.join('\n') : '') + (local.error ? '\n' + local.error : ''), '')}
      ${item(modal.available ? 'ok' : 'missing', 'Modal', modal.modal_version ? 'modal ' + modal.modal_version : 'package not installed', modal.available ? (modal.jobs || {}).summary || 'Sandboxes can be started.' : ((modal.problems || []).join('\n') || modal.error || 'Not available.'), modal.available ? '' : 'modal setup')}
      ${item(local.pdflatex ? 'ok' : 'optional', 'LaTeX', local.pdflatex || '', local.pdflatex ? 'A paper can be compiled on this machine.' : 'pdflatex was not found. Only needed for the optional paper step, and Modal can compile instead.', '')}</section>`;
    const lit = html`<section class="doc-g"><h2>Literature</h2>
      ${item(d.semantic_scholar_api_key ? 'ok' : 'warn', 'Semantic Scholar', d.semantic_scholar_api_key ? 'API key found' : 'no API key', d.semantic_scholar_api_key ? 'Papers are verified at the rate your key allows.' : 'Papers are still verified, at the slow rate of the public interface. A key makes the research phase faster.', d.semantic_scholar_api_key ? '' : 'export SEMANTIC_SCHOLAR_API_KEY=…')}</section>`;
    const pk = d.packages || {};
    setHTML(root, html`<div class="setup"><div class="setup-wrap">${head}${models}${compute}${lit}
      <p class="quiet setup-foot">Checked ${ago(new Date(doc.at).toISOString())} · PaperOrchestra ${d.version || ''} · Python ${d.python || ''}${app.ui.health && app.ui.health.workspace ? html` · workspace <code>${app.ui.health.workspace}</code>` : ''} · server at ${location.host}
        ${Object.keys(pk).length ? html`<br>Packages: ${Object.keys(pk).filter((k) => pk[k]).map((k) => k + ' ' + pk[k]).join(', ')}` : ''}</p></div></div>`);
  },
};
const cap = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '');
