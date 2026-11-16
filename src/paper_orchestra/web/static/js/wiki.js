/* The wiki view: the index of pages by kind, the page, and the margin (suggested edits,
   number check, lineage, where the page sits on the map, details). */

import { app, link } from './core.js';
import { $, dateTime, html, plural, raw, setHTML } from './util.js';
import { EXTRA_KINDS, KIND, STAGES, agentName, icon, pageGlyph } from './vocab.js';
import { afterRender, lineageHTML, notesHTML, pageName, pageTip, panelHTML, trailHTML } from './page.js';

/** A small inset that shows where the page sits in the flow. */
export function locator(store, path) {
  const W = 252, H = 104, lin = store.lineage(path);
  const near = {};
  lin.up.concat(lin.down).forEach((x) => (near[x.page.path] = true));
  const pos = {};
  let marks = '', lines = '';
  STAGES.forEach((k, i) => {
    const list = store.pages.filter((pg) => pg.kind === k);
    const x = 12 + (i * (W - 24)) / 7, pitch = Math.min(8, (H - 30) / Math.max(1, list.length));
    list.forEach((pg, j) => {
      const y = (H - 12) / 2 + (j - (list.length - 1) / 2) * pitch;
      pos[pg.path] = [x, y];
      marks += '<circle class="loc-m' + (pg.path === path ? ' me' : near[pg.path] ? ' near' : '') + (pg.review === 'proposed' ? ' prop' : '') + '" cx="' + x + '" cy="' + y.toFixed(1) + '" r="' + (pg.path === path ? 4 : near[pg.path] ? 2.6 : list.length > 40 ? 1.1 : 1.7) + '"/>';
    });
  });
  Object.keys(near).forEach((k) => { if (pos[k] && pos[path]) lines += '<path class="loc-l" d="M' + pos[path][0] + ' ' + pos[path][1].toFixed(1) + 'L' + pos[k][0] + ' ' + pos[k][1].toFixed(1) + '"/>'; });
  const letters = STAGES.map((k, i) => '<text class="loc-t" x="' + (12 + (i * (W - 24)) / 7) + '" y="' + (H - 2) + '" text-anchor="middle">' + k[0].toUpperCase() + '</text>').join('');
  return raw('<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="Position of this page on the map">' + lines + marks + letters + '</svg>');
}

const wideRail = () => window.matchMedia('(min-width: 1281px)').matches;

export const wikiView = {
  render(root, route) {
    const store = app.store;
    const idea = store.pages.find((pg) => pg.kind === 'idea') || store.pages[0];
    const missing = route.ref && !route.sel;
    const page = store.byPath.get(route.sel) || (missing ? null : idea);
    const filter = (app.ui.wikiFilter || '').toLowerCase().trim();
    const lessons = store.pages.filter((pg) => pg.kind === 'lesson').length;
    const kinds = STAGES.concat(EXTRA_KINDS.filter((k) => k !== 'lesson' || app.ui.showLessons || (page && page.kind === 'lesson')));
    const groups = kinds.map((k) => {
      const all = store.pages.filter((x) => x.kind === k);
      const list = all.filter((x) => !filter || (x.title + ' ' + (x.id || '') + ' ' + (x.summary || '')).toLowerCase().indexOf(filter) >= 0);
      if (!list.length) return '';
      return html`<div class="idx-group"><h3>${k === 'idea' ? 'Idea' : KIND[k].many}<span>${k === 'idea' ? '' : filter && list.length !== all.length ? list.length + ' of ' + all.length : all.length}</span></h3>
        ${list.map((x) => {
          const change = store.pendingFor(x.path).length;
          const cur = page && x.path === page.path;
          return html`<a class="idx r-${x.review}${cur ? ' cur' : ''}" href="${link({ view: 'wiki', sel: x.path, sub: '' })}"${cur ? raw(' aria-current="page"') : ''} data-tip="${pageTip(x)}">
            ${pageGlyph(x, 16)}<span class="idx-name">${pageName(x)}</span>${x.review === 'proposed' ? html`<span class="flag flag-prop">proposed</span>` : change ? html`<span class="flag flag-prop">change</span>` : ''}</a>`;
        })}</div>`;
    });
    const any = groups.some((g) => g !== '');
    const editing = !!page && route.sub === 'edit' && page.author !== 'system';
    const rail = !editing && !!page && wideRail();
    const view = page ? (store.views.get(page.path) || {}).data : null;
    const margin = rail ? notesHTML(store, page, { margin: true, part: 'edits' }) : null;
    const meta = page ? page.meta || {} : {};
    setHTML(root, html`<div class="wiki${editing ? ' wiki-editing' : ''}">
      <nav class="wiki-index" aria-label="Pages by kind"><div class="idx-top"><label class="sr" for="idx-filter">Filter pages</label><input id="idx-filter" data-input="wiki-filter" placeholder="Filter pages" value="${app.ui.wikiFilter || ''}" autocomplete="off">
        <button class="btn btn-s" data-act="new-page" data-tip="Add a page of your own">${icon('plus')}New</button></div>
        <div class="idx-scroll">${any ? groups : html`<p class="quiet idx-none">No page matches.</p>`}
          ${lessons && !app.ui.showLessons && !(page && page.kind === 'lesson') ? html`<button class="linkish idx-more" data-act="show-lessons">Show ${plural(lessons, 'lesson')} for the agents</button>` : ''}</div></nav>
      <div class="wiki-main">${page ? html`${trailHTML(store, page.path)}<div class="wiki-page">${panelHTML(store, page, { mode: 'wiki', sub: route.sub, notes: rail ? 'rail' : 'inline' })}</div>`
        : html`<div class="blank"><h1>There is no page “${route.ref}”</h1><p>It may have been deleted or renamed. Its last version, if it existed, is kept under <code>history/</code>.</p><p><a class="btn" href="${link({ view: 'wiki', sel: '', sub: '' })}">Open the idea</a></p></div>`}</div>
      ${rail ? html`<aside class="wiki-rail" aria-label="Margin">
        ${String(margin).trim() ? html`<div class="rail-notes">${margin}</div>` : ''}
        <h2 class="rail-h">Lineage</h2>${lineageHTML(store, page)}
        <h2 class="rail-h">On the map</h2><a class="locator" href="${link({ view: 'map', sel: page.path, sub: '' })}" data-tip="Show on the map (1)">${locator(store, page.path)}</a>
        <h2 class="rail-h">Details</h2><dl class="details"><dt>File</dt><dd><code>${page.path}</code></dd><dt>Author</dt><dd>${agentName(page.author)}</dd>
          ${page.round ? html`<dt>Round</dt><dd>${page.round}</dd>` : ''}<dt>Changed</dt><dd>${dateTime(page.updated)}${meta.updated_by ? ' by ' + agentName(meta.updated_by) : ''}</dd>
          <dt>Versions</dt><dd>${view ? html`<button class="linkish" data-act="history" data-ref="${page.path}">${(view.history || []).length + 1}</button>` : '…'}</dd></dl></aside>` : ''}
    </div>`);
    afterRender(root);
    const cur = $('.idx.cur', root);
    if (cur && cur.scrollIntoView && app.lastRenderKey !== app.renderKey) cur.scrollIntoView({ block: 'nearest' });
  },
};
