/* The map view. Draws the layout in SVG, keeps pan and zoom, traces the lineage of the
   selection, and lets a proposal be decided where it stands.

   At scale the map stays usable by folding: the sources a topic cites are drawn as one
   bundle that opens when the topic (or one of its sources) is selected; a round filter and
   a lineage filter hide the rest; the canvas scrolls vertically under pinned headings; the
   layout is cached by a signature of what is drawn and seeded with the previous ordering.

   A crowded map draws the argument only: which experiment tests which hypothesis, which run
   belongs to which experiment, what a finding rests on and what it supports or refutes. The
   reference lines (cites, about, answers, mentions, and every line that ends at a source)
   are drawn for the page that is selected or, while none is, pointed at. Asked for all of
   them, the map keeps its plan and draws them over it, thin, along the gutters and between
   the rows: no page changes its place. A stage with more pages than the sheet has rows for
   shows each as one line: mark, id, title. */

import { app, link, select } from './core.js';
import { $, $$, esc, html, plural, raw, setHTML, tf } from './util.js';
import { KIND, STAGES, TONE, glyph, glyphInner, icon, kindName, relOf } from './vocab.js';
import { layout, routeRefs } from './layout.js';
import { afterRender, label, panelHTML, shortTitle, statusLine, trailHTML } from './page.js';
import { continueLabel, engineView, strip } from './shell.js';

const R = { idea: 9, source: 4.6, topic: 9, question: 8.4, hypothesis: 8, experiment: 9, result: 5.6, finding: 4.6, bundle: 6 };
const SLOT = { idea: 26, source: 16, topic: 24, question: 22, hypothesis: 22, experiment: 24, result: 16, finding: 22 };
const CAP = { idea: 86, source: 150, topic: 150, question: 156, hypothesis: 176, experiment: 168, result: 96, finding: 176 };
const MIN = { idea: 30, source: 84, topic: 72, question: 46, hypothesis: 92, experiment: 92, result: 48, finding: 60 };
const HERO = { question: 1, hypothesis: 1, experiment: 1, finding: 1 };
const TOP = 44, BTN = 58, ROOM = 80, NEAR_STEPS = 2;
/* a map with more marks or more reference lines than this is crowded */
const CROWD_NODES = 50, CROWD_REFS = 40;
/** A line that says where something was read or what it is about, not what was argued. */
const isRef = (type) => type === 'answers' || relOf(type).fam === 'read';

let ctx, fam;
const tw = (text, font) => { ctx = ctx || document.createElement('canvas').getContext('2d'); ctx.font = font; return ctx.measureText(text).width; };
const fonts = () => { fam = fam || getComputedStyle(document.body).fontFamily; return { id: '600 11.5px ' + fam, t: '400 11.5px ' + fam, s: '400 10px ' + fam, src: '400 11px ' + fam }; };
const fit = (text, font, max) => {
  if (tw(text, font) <= max) return text;
  let lo = 0, hi = text.length;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (tw(text.slice(0, m).trimEnd() + '…', font) <= max) lo = m; else hi = m - 1; }
  return lo <= 2 ? '' : text.slice(0, lo).trimEnd() + '…';
};
const wrap2 = (text, font, max) => {
  const words = text.split(' ');
  let a = '';
  while (words.length && tw((a ? a + ' ' : '') + words[0], font) <= max) a = (a ? a + ' ' : '') + words.shift();
  if (!a) return [fit(text, font, max)];
  return words.length ? [a, fit(words.join(' '), font, max)] : [a];
};

const V = { st: { k: 1, tx: 0, ty: 0, sig: '', lay: null }, prevKeys: {}, root: null, store: null, route: null };

function mount(root, store) {
  setHTML(root, html`<div class="map-view" data-pid="${store.pid}">
    <h1 class="sr">Map of ${store.project.title}</h1>
    <div class="strip-slot"></div>
    <div class="map-body">
      <div class="map-canvas" tabindex="0" role="group" aria-label="Research map. Tab goes to the selected page; the arrow keys move between pages; Enter opens one.">
        <svg class="map-svg" role="group" aria-label="Research map" tabindex="-1" focusable="false"></svg>
        <div class="map-empty"></div><div class="map-tip" role="tooltip"></div>
        <div class="map-foot"><div class="map-key"></div><div class="map-tools"></div></div>
      </div>
      <aside class="panel" aria-label="Selected page"></aside>
    </div></div>`);
  V.st = { k: 1, tx: 0, ty: 0, sig: '', lay: null, order: (V.st && V.st.pid === store.pid && V.st.order) || null, pid: store.pid };
  const canvas = $('.map-canvas', root), svg = $('.map-svg', root);
  let drag = null;
  svg.addEventListener('pointerdown', (ev) => {
    if (ev.target.closest('.node') || ev.button) return;
    drag = { x: ev.clientX, y: ev.clientY, tx: V.st.tx, ty: V.st.ty, moved: false };
    svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.moved) { V.st.tx = drag.tx + dx; V.st.ty = drag.ty + dy; V.st.user = true; canvas.classList.add('panning'); applyT(); }
  });
  const endDrag = (ev) => {
    const was = drag; drag = null; canvas.classList.remove('panning');
    /* a click on the empty sheet clears the selection; not the second click of a double
       click on a node, under which the map has just moved */
    if (was && !was.moved && ev.type === 'pointerup' && !ev.target.closest('.node') && V.route.sel && performance.now() - (V.pickedAt || 0) > 500) select('');
  };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('wheel', (ev) => {
    if (ev.target.closest('.map-foot')) return;
    ev.preventDefault();
    if (ev.ctrlKey || ev.metaKey) zoomAt(Math.exp(-ev.deltaY * 0.01), ev.offsetX, ev.offsetY);
    else { V.st.tx -= ev.deltaX; V.st.ty -= ev.deltaY; V.st.user = true; applyT(); }
  }, { passive: false });
  svg.addEventListener('click', (ev) => {
    if (ev.target.closest('.n-btn')) return; // handled by the action dispatcher
    const n = ev.target.closest('.node');
    if (!n || ev.detail > 1) return;          // the second click of a double click would unselect what the first selected
    V.pickedAt = performance.now();
    if (n.dataset.bundle) openBundle(n.dataset.key);
    else select(n.dataset.key === V.route.sel ? '' : n.dataset.key);
  });
  /* A page is pointed at when the pointer comes onto it, or when the person moves the map
     under the pointer. A page that a new plan put under a pointer at rest was not: selecting
     a page opens it beside the map, the map is laid out again, and what then lies under the
     pointer shows neither its tip nor its lines until the pointer moves. */
  let at = null, waits = false;
  const moved = (ev) => { const was = at; at = [ev.clientX, ev.clientY]; return !was || was[0] !== at[0] || was[1] !== at[1]; };
  const point = (ev) => {
    const n = ev.target.closest('.node'), e = ev.target.closest('.edge-hit');
    if (n) tip(n.dataset.key); else if (e) edgeTip(e, ev); else hideTip();
    if ((n ? n.dataset.key : '') !== V.pointed) pointAt(n ? n.dataset.key : '');
  };
  svg.addEventListener('mousemove', (ev) => {
    if (!moved(ev)) return;
    V.shifted = false;
    if (waits) { waits = false; point(ev); }
  });
  svg.addEventListener('mouseover', (ev) => {
    const real = moved(ev);
    if (!real && V.shifted) { waits = true; return; }
    if (real) V.shifted = false;
    waits = false;
    point(ev);
  });
  svg.addEventListener('mouseleave', () => { hideTip(); pointAt(''); });
  svg.addEventListener('focusin', (ev) => {
    const n = ev.target.closest('.node');
    if (!n) return;
    /* Tab can reach a node that lies outside the sheet: the map moves so that it shows */
    if (ev.target.matches(':focus-visible')) reveal(n.dataset.key);
    if (!ev.target.closest('.n-btn')) tip(n.dataset.key);
    pointAt(n.dataset.key);
  });
  svg.addEventListener('focusout', () => { hideTip(); pointAt(''); });
  svg.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    const b = ev.target.closest('.n-btn'), n = ev.target.closest('.node');
    if (b) { ev.preventDefault(); ev.stopPropagation(); b.dispatchEvent(new MouseEvent('click', { bubbles: true })); }
    else if (n) { ev.preventDefault(); if (n.dataset.bundle) openBundle(n.dataset.key); else select(n.dataset.key); }
  });
  /* reference lines: for the selected page only, or all of them */
  $('.map-foot', root).addEventListener('click', (ev) => {
    if (!ev.target.closest('[data-act="map-refs"]') || ev.detail > 1) return;
    app.ui.map.refs = V.st.refsAlways ? 'sel' : 'always';
    /* a crowded map keeps its plan, and with it where the person had moved it */
    if (!V.st.crowded) { V.st.user = false; V.st.fitted = false; }
    app.render('ui');
  });
  if (window.ResizeObserver) new ResizeObserver(() => { cancelAnimationFrame(V._raf); V._raf = requestAnimationFrame(() => { if (V.root && V.root.contains(canvas)) draw(); }); }).observe(canvas);
}

/** Move the map just far enough that a node is on the sheet (below the headings, above the legend). */
function reveal(key) {
  const s = V.st, n = s.lay && s.lay.nodes[key];
  if (!n || !s.W) return;
  const x = s.tx + s.k * n.x, y = s.ty + s.k * n.y;
  const top = 34 + 28, bottom = s.H - (s.bottom || 0) - 28, left = 24, right = Math.max(left, s.W - 180);
  const dx = x < left ? left - x : x > right ? right - x : 0, dy = y < top ? top - y : y > bottom ? bottom - y : 0;
  if (!dx && !dy) return;
  s.tx += dx; s.ty += dy; s.user = true;
  applyT();
}

function openBundle(key) {
  const b = V.st.bundles && V.st.bundles[key];
  if (!b) return;
  if (b.home && V.store.byPath.has(b.home)) select(b.home);
  else { app.ui.map.open[key] = !app.ui.map.open[key]; app.render('map'); }
}

function zoomAt(f, cx, cy) {
  const s = V.st, k = Math.max(0.4, Math.min(2.2, s.k * f));
  s.tx = cx - ((cx - s.tx) * k) / s.k; s.ty = cy - ((cy - s.ty) * k) / s.k; s.k = k; s.user = true;
  applyT();
}
function zoom(dir) {
  const c = $('.map-canvas', V.root);
  if (!c || !V.st.lay) return;
  if (dir === 0) {
    /* fit: the automatic view, or, when the plan is larger than the sheet, all of it at once */
    const s = V.st, lay = s.lay;
    const kx = c.clientWidth / lay.width, ky = (c.clientHeight - (s.bottom || 0)) / Math.max(1, lay.bottomY + 12);
    const big = (kx < 0.99 || ky < 0.99) && !s.fitted;
    s.fitted = big; s.user = big; s.k = big ? Math.max(0.4, Math.min(1, kx, ky)) : 1; s.tx = 0; s.ty = 0;
    if (big) applyT(); else draw();
  } else zoomAt(dir > 0 ? 1.2 : 1 / 1.2, c.clientWidth / 2, c.clientHeight / 2);
}

function applyT() {
  const s = V.st, root = V.root;
  const vp = $('.vp', root);
  if (!vp) return;
  /* the map can be moved, never lost: some of it stays on the sheet */
  if (s.lay && s.W) {
    const w = s.lay.width * s.k, h = (s.lay.bottomY + 12) * s.k, viewH = s.H - (s.bottom || 0);
    s.tx = Math.max(Math.min(0, s.W - w) - s.W * 0.5, Math.min(s.W * 0.5, s.tx));
    s.ty = Math.max(Math.min(0, viewH - h) - viewH * 0.35, Math.min(viewH * 0.35, s.ty));
  }
  vp.setAttribute('transform', 'translate(' + s.tx + ' ' + s.ty + ') scale(' + s.k + ')');
  /* The headings stand over their columns. A map drawn smaller has its headings drawn
     smaller with it (down to a size that can still be read); smaller still, they lose
     their counts and then their names, so that none ever runs into the next. */
  const hk = Math.max(0.8, Math.min(1, s.k));
  $$('.head', root).forEach((h) => h.setAttribute('transform', 'translate(' + (s.tx + s.k * +h.dataset.x) + ' ' + (1 - hk) * 17 + ') scale(' + hk + ')'));
  const canvas = $('.map-canvas', root);
  canvas.classList.toggle('heads-s', s.k < 0.8);
  canvas.classList.toggle('heads-xs', s.k < 0.68);
}

/* lineage of the selection: upstream and downstream, with the direct links marked.
   Only the nearest steps unless the whole lineage is asked for. */
function trace(lay, edges, sel, steps) {
  const near = new Set(), kin = new Set(), strong = new Set(), mid = new Set();
  let more = false;
  if (!sel || !lay.nodes[sel]) return { near, kin, strong, mid, more };
  const walk = (dir) => {
    const seen = new Set([sel]);
    let frontier = new Set([sel]), depth = 0;
    while (frontier.size) {
      depth++;
      const next = new Set();
      edges.forEach((e, i) => {
        const from = dir > 0 ? e.lo : e.hi, to = dir > 0 ? e.hi : e.lo;
        if (!frontier.has(from)) return;
        if (depth > steps) { if (!seen.has(to)) more = true; return; }
        (from === sel ? strong : mid).add(i);
        if (!seen.has(to)) { seen.add(to); next.add(to); (from === sel ? near : kin).add(to); }
      });
      if (depth > steps) break;
      frontier = next;
    }
  };
  walk(1); walk(-1);
  near.forEach((k) => kin.delete(k));
  return { near, kin, strong, mid, more };
}

/* everything upstream and downstream of a page, over the wiki's own relations */
function closure(store, start) {
  const keep = new Set([start]);
  const stage = (path) => { const p = store.byPath.get(path); return p ? STAGES.indexOf(p.kind) : -1; };
  [1, -1].forEach((dir) => {
    let frontier = [start];
    const seen = new Set([start]);
    while (frontier.length) {
      const next = [];
      frontier.forEach((path) => {
        const a = store.adj.get(path);
        if (!a) return;
        a.out.concat(a.inn).forEach((e) => {
          const other = e.s === path ? e.t : e.s;
          if (seen.has(other)) return;
          const d = stage(other) - stage(path);
          if (dir > 0 ? d > 0 : d < 0) { seen.add(other); keep.add(other); next.push(other); }
        });
      });
      frontier = next;
    }
  });
  return keep;
}

/** What is on the map: pages, folded bundles, and the edges between them. */
function scene(store, sel) {
  const m = app.ui.map, showRej = app.ui.showRejected;
  let pages = store.pages.filter((pg) => STAGES.indexOf(pg.kind) >= 0 && (showRej || pg.review !== 'rejected'));
  const rounds = Array.from(new Set(pages.map((pg) => pg.round).filter((r) => r != null && r > 0))).sort((a, b) => a - b);
  if (m.round != null && rounds.indexOf(m.round) < 0) m.round = null;
  if (m.round != null) pages = pages.filter((pg) => pg.kind === 'idea' || pg.round === m.round || pg.path === sel);
  if (m.lineageOnly && sel && store.byPath.has(sel)) { const keep = closure(store, sel); pages = pages.filter((pg) => keep.has(pg.path)); }
  const on = new Set(pages.map((pg) => pg.path));
  const sources = pages.filter((pg) => pg.kind === 'source');
  const fold = m.fold == null ? sources.length > 20 || pages.length > 70 : m.fold;
  /* typed relations first; a mention is drawn only where nothing else connects the two pages */
  const typed = new Set();
  store.edges.forEach((e) => { if (e.type !== 'mentions') { typed.add(e.s + '|' + e.t); typed.add(e.t + '|' + e.s); } });
  let edges = store.edges.filter((e) => on.has(e.s) && on.has(e.t) && (e.type !== 'mentions' || (!typed.has(e.s + '|' + e.t) && pages.length <= 90)));
  const seenEdge = new Set();
  edges = edges.filter((e) => { const k = e.s + '|' + e.t + '|' + e.type; if (seenEdge.has(k)) return false; seenEdge.add(k); return true; });

  const bundles = {}, inBundle = {};
  if (fold && sources.length) {
    const citers = {};
    edges.forEach((e) => { const t = store.byPath.get(e.t); if (t && t.kind === 'source') (citers[e.t] = citers[e.t] || []).push(e.s); });
    const weight = {};
    Object.values(citers).forEach((list) => list.forEach((p) => (weight[p] = (weight[p] || 0) + 1)));
    sources.forEach((src) => {
      const cands = (citers[src.path] || []).slice().sort((a, b) => {
        const ka = store.byPath.get(a).kind === 'topic' ? 0 : 1, kb = store.byPath.get(b).kind === 'topic' ? 0 : 1;
        return ka - kb || weight[b] - weight[a] || a.localeCompare(b);
      });
      const home = cands[0] || '';
      const key = 'bundle:' + home;
      (bundles[key] = bundles[key] || { key, home, members: [] }).members.push(src.path);
    });
    Object.values(bundles).forEach((b) => {
      b.open = b.members.length < 2 || !!m.open[b.key] || (sel && (sel === b.home || b.members.indexOf(sel) >= 0));
      if (!b.open) b.members.forEach((p) => (inBundle[p] = b.key));
    });
  }
  const nodes = pages.filter((pg) => !inBundle[pg.path]).map((pg) => ({ key: pg.path, page: pg, kind: pg.kind }));
  Object.values(bundles).forEach((b) => { if (!b.open) nodes.push({ key: b.key, bundle: b, kind: 'bundle' }); });
  if (Object.keys(inBundle).length) {
    const seen = new Set();
    edges = edges.map((e) => ({ s: inBundle[e.s] || e.s, t: inBundle[e.t] || e.t, type: e.type })).filter((e) => {
      const k = e.s + '|' + e.t + '|' + (e.type === 'mentions' ? 'cites' : e.type);
      if (e.s === e.t || seen.has(k)) return false;
      seen.add(k); return true;
    });
  }
  /* a crowded map keeps its reference lines back until a page is selected or pointed at */
  const read = (key) => !!bundles[key] || (store.byPath.get(key) || {}).kind === 'source';
  const held = (e) => isRef(e.type) || read(e.s) || read(e.t);
  const refs = edges.filter(held), strong = edges.filter((e) => !held(e));
  const crowded = strong.length > 0 && (nodes.length > CROWD_NODES || refs.length > CROWD_REFS);
  if (crowded) m.refTool = true;
  const always = m.refs ? m.refs === 'always' : !crowded;
  /* Every line, asked for on a crowded map: the plan stays as it is and the reference lines
     are drawn over it, so that the pages do not change places under the person who asked. */
  const over = always && crowded, apart = !always || over;
  return { nodes, edges: apart ? strong : edges, refs: apart ? refs : [], refsAlways: always, refsOver: over, crowded, refTool: !!(refs.length && (m.refTool || m.refs)), bundles, rounds, fold, sources: sources.length, hidden: store.pages.filter((pg) => STAGES.indexOf(pg.kind) >= 0).length - pages.length };
}

function draw() {
  const root = V.root, store = V.store, route = V.route, s = V.st;
  const canvas = $('.map-canvas', root), svg = $('.map-svg', root);
  if (!canvas) return;
  const sel = route.sel && store.byPath.has(route.sel) ? route.sel : '';
  const view = $('.map-view', root);
  view.classList.toggle('has-panel', !!sel);
  const heading = $('.map-view > h1', root);
  if (heading) heading.textContent = 'Map of ' + store.project.title;      // the intake may have named the project since
  /* the strip's log follows what arrives, unless the person scrolled up in it */
  const oldLog = $('.strip-log', root);
  const logTop = oldLog ? oldLog.scrollTop : 0, logStuck = !oldLog || oldLog.scrollTop + oldLog.clientHeight >= oldLog.scrollHeight - 24;
  setHTML($('.strip-slot', root), strip(store));
  const newLog = $('.strip-log', root);
  /* a running phase is followed; the log of a failure opens at its top, where the advice and
     the worker's own words are */
  const follows = (el) => !!el && el.hasAttribute('data-follow');
  if (newLog) newLog.scrollTop = follows(newLog) ? (logStuck ? newLog.scrollHeight : logTop) : oldLog && !follows(oldLog) ? logTop : 0;
  $$('.strip-slot .strip-pre', root).forEach((el) => { el.scrollTop = el.scrollHeight; });
  const panel = $('.panel', root);
  const keep = panel.dataset.key === sel ? panel.scrollTop : 0;
  setHTML(panel, sel ? html`${trailHTML(store, sel)}${panelHTML(store, store.byPath.get(sel), { mode: 'panel', sub: route.sub })}` : html``);
  panel.dataset.key = sel; panel.scrollTop = keep;
  afterRender(panel);

  const m = app.ui.map, showRej = app.ui.showRejected;
  const sc = scene(store, sel);
  const rejected = store.pages.filter((pg) => pg.review === 'rejected' && STAGES.indexOf(pg.kind) >= 0).length;
  setHTML($('.map-tools', root), html`
    ${rejected ? html`<button class="tool${showRej ? ' on' : ''}" data-act="toggle-rejected" aria-pressed="${tf(showRej)}" data-tip="Show or hide rejected pages">Rejected ${showRej ? 'shown' : 'hidden'}</button>` : ''}
    ${sc.sources > 6 ? html`<button class="tool${sc.fold ? ' on' : ''}" data-act="toggle-fold" aria-pressed="${tf(sc.fold)}" data-tip="Draw the sources a topic cites as one bundle; it opens when the topic is selected">Sources ${sc.fold ? 'folded' : 'unfolded'}</button>` : ''}
    ${sc.refTool ? html`<button class="tool${sc.refsAlways ? ' on' : ''}" data-act="map-refs" aria-pressed="${tf(sc.refsAlways)}" data-tip="${sc.refsAlways ? 'Every line is drawn. Press to draw the lines to sources and those that cite, mention, answer or are about a page for the selected page only.' : 'Lines to sources and lines that cite, mention, answer or are about a page are drawn for the page you select; while nothing is selected, for the page you point at. Press to draw all of them.'}">Reference lines: ${sc.refsAlways ? 'always' : 'on selection'}</button>` : ''}
    ${sc.rounds.length > 1 ? html`<label class="tool tool-sel"><span class="sr">Round shown</span><select data-input="map-round" id="map-round" aria-label="Round shown"><option value="">All rounds</option>${sc.rounds.map((r) => html`<option value="${r}"${m.round === r ? raw(' selected') : ''}>Round ${r}</option>`)}</select></label>` : ''}
    <label class="tool tool-sel"><span class="sr">Lineage of the selection</span><select data-input="map-lineage" id="map-lineage" aria-label="Lineage of the selection">
      <option value="near"${!m.deep && !m.lineageOnly ? raw(' selected') : ''}>Lineage: nearest steps</option><option value="all"${m.deep && !m.lineageOnly ? raw(' selected') : ''}>Lineage: all of it</option><option value="only"${m.lineageOnly ? raw(' selected') : ''}>Lineage only, hide the rest</option></select></label>
    <span class="tool-note">Scroll or drag to move</span>
    <span class="tool-group"><button class="tool tool-i" data-act="zoom-out" data-tip="Zoom out (−)" aria-label="Zoom out">${icon('minus')}</button>
      <button class="tool tool-i" data-act="zoom-fit" data-tip="Fit the whole map (0)" aria-label="Fit the whole map">${icon('fit')}</button>
      <button class="tool tool-i" data-act="zoom-in" data-tip="Zoom in (+)" aria-label="Zoom in">${icon('plus')}</button></span>`);
  const keyEl = $('.map-key', root), keyOf = sc.refsAlways ? 'all' : 'sel';
  if (keyEl.dataset.refs !== keyOf) { setHTML(keyEl, keyHTML(!sc.refsAlways)); keyEl.dataset.refs = keyOf; }
  canvas.classList.toggle('narrow', canvas.clientWidth < 780);
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  const bottom = $('.map-foot', root).offsetHeight + 12;
  const F = fonts();
  /* a stage with more pages than the sheet has rows for: one line each, mark, id and title */
  const dense = sc.nodes.length > 60, rows = Math.max(6, Math.floor((H - TOP - 96) / 38)), perKind = {};
  sc.nodes.forEach((n) => { if (n.page) perKind[n.kind] = (perKind[n.kind] || 0) + 1; });
  const tight = (kind) => dense && kind !== 'source' && kind !== 'idea' && perKind[kind] > rows;
  const info = {}, idCol = {};
  sc.nodes.forEach((n) => { if (n.page && HERO[n.kind]) idCol[n.kind] = Math.max(idCol[n.kind] || 0, tw(n.page.id || '', F.id)); });
  const nodes = sc.nodes.map((n) => {
    if (n.bundle) {
      const text = plural(n.bundle.members.length, 'source');
      const w = tw(text, F.src) + 2;
      info[n.key] = { bundle: n.bundle, st: { full: '', short: '', ref: '' }, idText: '', title: text, idW: 0, lines: 1 };
      return { key: n.key, kind: 'source', layer: 1, r: R.bundle, hu: 11, hd: 11, w: Math.ceil(w), min: Math.ceil(w), hard: Math.ceil(w) };
    }
    const pg = n.page;
    const st = statusLine(store, pg);
    const idText = pg.kind === 'source' ? '' : pg.kind === 'idea' ? 'Idea' : pg.kind === 'result' ? shortTitle(pg) : pg.id || '';
    const title = pg.kind === 'source' ? shortTitle(pg) : pg.kind === 'idea' || pg.kind === 'result' ? '' : pg.title;
    const idW = HERO[pg.kind] ? idCol[pg.kind] : idText ? tw(idText, F.id) : 0;
    const tW = title ? tw(title, pg.kind === 'source' ? F.src : F.t) : 0;
    const btn = st.ref ? BTN : 0;
    const stFull = st.full ? tw(st.full, F.s) + btn : 0;
    let w, min, hard;
    if (tight(pg.kind)) {
      /* the marks of a proposal stand at the end of its line (a topic is decided on its page) */
      const held = pg.kind === 'topic' ? 0 : btn, one = idText && title ? idW + 5 + tW : Math.max(idW, tW);
      w = Math.min(CAP[pg.kind], one) + held; min = Math.max(MIN[pg.kind], idW + held);
      info[pg.path] = { pg, st, idText, title, idW, lines: 1, tight: true, held };
      return { key: pg.path, kind: pg.kind, r: R[pg.kind], hu: held ? 14 : 12, hd: held ? 14 : 12, w: Math.ceil(w), min: Math.ceil(Math.min(min, w)) };
    }
    if (pg.kind === 'source') { w = Math.min(CAP.source, tW); min = w; hard = MIN.source; }
    else if (pg.kind === 'idea') { w = Math.max(idW, stFull); min = w; }
    else if (pg.kind === 'result') { w = Math.max(idW, stFull); min = w; hard = Math.max(MIN.result, Math.ceil(stFull)); }   // "succeeded" is never cut
    else if (pg.kind === 'topic') { w = tW <= 120 ? tW : Math.max(MIN.topic, Math.min(CAP.topic, tW / 2 + 16)); min = Math.min(w, MIN.topic); }
    else {
      const one = idW + 5 + tW;
      w = Math.max(stFull, one <= CAP[pg.kind] ? one : Math.min(CAP[pg.kind], idW + 5 + tW / 2 + 16));
      min = Math.max(MIN[pg.kind], idW);
    }
    info[pg.path] = { pg, st, idText, title, idW, lines: 1 };
    return { key: pg.path, kind: pg.kind, r: R[pg.kind], hu: 19, hd: 17, w: Math.ceil(w), min: Math.ceil(Math.min(min, w)), hard: hard == null ? undefined : Math.min(hard, Math.ceil(w)) };
  });
  const reviewOf = (key) => (info[key] && info[key].pg ? info[key].pg.review : 'accepted');
  const refs = sc.refs.filter((e) => info[e.s] && info[e.t]);
  /* the look of a line: lines of one look may run together where the map is crowded */
  const edges = sc.edges.filter((e) => info[e.s] && info[e.t]).map((e) => {
    const a = reviewOf(e.s), b = reviewOf(e.t);
    return { s: e.s, t: e.t, type: e.type, look: relOf(e.type).fam + (a === 'rejected' || b === 'rejected' ? '-dead' : a === 'proposed' || b === 'proposed' ? '-prop' : '') };
  });
  const sig = [W, H, bottom, nodes.map((n) => n.key + (info[n.key].pg ? info[n.key].pg.review : '') + n.w + (info[n.key].tight ? 't' : '')).join(), edges.map((e) => e.s + e.t + e.type).join(), sc.refsAlways && !sc.refsOver ? '' : 'refs:' + refs.map((e) => e.s + e.t).join()].join('|');
  const drawnSig = s.sig;
  if (sig !== s.sig) {
    /* a column is never narrower than its own heading */
    const colMin = {};
    const fh = '600 10.5px ' + fam;
    STAGES.forEach((k) => {
      if (k === 'idea') return;
      const count = sc.nodes.filter((n) => n.page && n.kind === k).length + (k === 'source' ? Object.values(sc.bundles).filter((b) => !b.open).reduce((a, b) => a + b.members.length, 0) : 0);
      const wait = sc.nodes.some((n) => n.page && n.kind === k && n.page.review === 'proposed');
      const name = KIND[k].many.toUpperCase();
      colMin[k] = Math.ceil(tw(name, fh) + name.length * 0.8 + 8 + tw(String(count), fh) + (wait ? 20 : 0) + 4 - 5 - 4 - (k === 'finding' ? 0 : 22) + 8);
    });
    const opt = { W, H, top: TOP, bottom, padL: 14, padR: 10, min: MIN, colMin, slot: SLOT, gutterGrow: 56, gapMax: 12, gapMin: dense ? 2 : 3, prevOrder: s.order, quiet: sc.refsAlways && !sc.refsOver ? null : refs, packed: Object.fromEntries(STAGES.map((k) => [k, tight(k)])) };
    /* first pass finds the width each column gets; then titles are wrapped and heights are known */
    const first = layout(nodes, edges, opt);
    nodes.forEach((n) => {
      const it = info[n.key], lw = first.cols[n.layer == null ? STAGES.indexOf(n.kind) : n.layer].lw;
      if (it.bundle || it.tight) return;
      if (n.kind === 'source') { n.hu = 10; n.hd = 10; return; }
      if (n.kind === 'idea') { n.hu = 18; n.hd = 18; return; }
      if (n.kind === 'topic') { it.lines = dense ? 1 : wrap2(it.title, F.t, lw).length; n.hu = it.lines > 1 ? 32 : 19; n.hd = 10; return; }
      const room = lw - it.idW - 5;
      it.lines = !dense && HERO[n.kind] && it.title && room >= ROOM ? wrap2(it.title, F.t, room).length : 1;
      n.hu = it.lines > 1 ? 32 : 19; n.hd = it.st.full ? 17 : 10;
    });
    opt.prevOrder = first.order;
    s.lay = layout(nodes, edges, opt);
    s.order = Object.assign({}, s.order, s.lay.order);
    s.ext = {}; s.lines = {};
    nodes.forEach((n) => { s.ext[n.key] = [n.hu, n.hd]; s.lines[n.key] = info[n.key].lines; });
    s.sig = sig;
    if (!s.user) { s.k = 1; s.tx = 0; s.ty = 0; }
  }
  /* the plan or the selection changed under the pointer: what it pointed at may lie elsewhere now */
  if (sig !== drawnSig || sel !== V.sel) { V.pointed = ''; V.shifted = true; hideTip(); }
  V.sel = sel;
  s.W = W; s.H = H; s.bottom = bottom; s.bundles = sc.bundles; s.folded = sc.fold; s.refsAlways = sc.refsAlways; s.crowded = sc.crowded;
  const lay = s.lay;
  Object.keys(info).forEach((k) => (info[k].lines = s.lines[k] || 1));
  /* if the plan is larger than the sheet, keep the selection in sight */
  if (!s.user) {
    s.k = lay.width > W ? Math.max(0.86, W / lay.width) : 1;
    /* on a narrow sheet the page lies over the map: keep the selection in the part that shows */
    const cover = sel && getComputedStyle(panel).position === 'absolute' ? panel.offsetWidth : 0;
    const overflow = lay.width * s.k - (W - cover);
    s.tx = 0; s.ty = 0;
    const at = sel && lay.nodes[sel];
    if (overflow > 0 && at) s.tx = -Math.max(0, Math.min(overflow, at.x * s.k - (W - cover) * 0.5));
    const tall = (lay.bottomY + 12) * s.k - (H - bottom);
    if (tall > 0 && at) s.ty = -Math.max(0, Math.min(tall, at.y * s.k - (H - bottom) * 0.5));
  }
  /* the lines of the plan, and after them the reference lines of the selected page */
  s.refs = sc.refsOver ? [] : refs;
  let lines = lay.edges;
  if (sc.refsOver) {
    /* all of them: those that end at one page leave it as one line and part where they arrive */
    const ends = {};
    refs.forEach((e) => (ends[e.t] = ends[e.t] || []).push(e));
    lines = lines.concat(...Object.keys(ends).map((key) => routeRefs(lay, key, ends[key])));
  } else if (sel && refs.length) lines = lines.concat(routeRefs(lay, sel, refs.filter((e) => e.s === sel || e.t === sel)));
  s.drawn = lines;
  const tr = trace(lay, lines, sel, m.deep || m.lineageOnly ? Infinity : NEAR_STEPS);
  const flash = app.ui.flash && Date.now() - app.ui.flash.at < 1600 ? app.ui.flash : null;
  const prev = V.prevKeys[store.pid];
  const empty = store.pages.filter((pg) => STAGES.indexOf(pg.kind) >= 0).length <= 1;

  /* headers: the name of each stage is also the key to its glyph */
  let heads = '';
  const fh = '600 10.5px ' + fam;
  lay.cols.forEach((c, i) => {
    const pagesOf = sc.nodes.filter((n) => n.page && n.kind === c.kind);
    const folded = c.kind === 'source' ? Object.values(sc.bundles).filter((b) => !b.open).reduce((a, b) => a + b.members.length, 0) : 0;
    const count = pagesOf.length + folded;
    const waiting = pagesOf.filter((n) => n.page.review === 'proposed').length;
    const name = c.kind === 'idea' ? 'IDEA' : KIND[c.kind].many.toUpperCase();
    const span = (i < lay.cols.length - 1 ? lay.cols[i + 1].x0 : lay.width) - c.x0 - c.slot - 10;
    const nameW = tw(name, fh) + name.length * 0.8, nW = c.kind === 'idea' ? 0 : 8 + tw(String(count), fh), wW = waiting ? 9 + tw(waiting + ' proposed', fh) : 0;
    const showW = waiting && nameW + nW + wW <= span;
    heads += '<g class="head' + (count ? '' : ' head-empty') + '" data-x="' + c.x0 + '" transform="translate(' + c.x0 + ' 0)">' +
      '<g transform="translate(' + c.slot / 2 + ' 20) scale(.82)">' + glyphInner(c.kind, 'accepted', '') + '</g>' +
      '<text x="' + (c.slot + 4) + '" y="24"><tspan class="head-name">' + name + '</tspan>' +
      (c.kind === 'idea' ? '' : '<tspan class="head-n" dx="6">' + count + '</tspan>') +
      (showW ? '<tspan class="head-w" dx="7">' + waiting + ' proposed</tspan>' : waiting ? '<tspan class="head-w" dx="5">+' + waiting + '</tspan>' : '') + '</text></g>';
  });

  /* edges, quietest first */
  const famOrder = { read: 0, reason: 1, test: 2, pos: 3, neg: 4 };
  const order = lines.map((e, i) => i).sort((a, b) => famOrder[relOf(lines[a].type).fam] - famOrder[relOf(lines[b].type).fam] || (tr.strong.has(a) ? 1 : 0) - (tr.strong.has(b) ? 1 : 0));
  let ed = '', hits = '';
  const marks = {};
  order.forEach((i) => {
    const e = lines[i], rel = relOf(e.type);
    const provisional = reviewOf(e.s) === 'proposed' || reviewOf(e.t) === 'proposed';
    const dead = reviewOf(e.s) === 'rejected' || reviewOf(e.t) === 'rejected';
    const type = String(e.type).replace(/[^a-z_]/g, '');
    const cls = 'edge f-' + rel.fam + ' t-' + type + (provisional ? ' e-prop' : '') + (dead ? ' e-dead' : '') + (sel ? (tr.strong.has(i) ? ' hl' : tr.mid.has(i) ? ' kin' : ' dim') : '');
    ed += '<path class="' + cls + '" d="' + e.d + '"/>';
    hits += '<path class="edge-hit" d="' + e.d + '" data-s="' + esc(e.s) + '" data-t="' + esc(e.t) + '" data-type="' + type + '"/>';
    if ((rel.fam === 'pos' || rel.fam === 'neg') && e.mid) {
      /* lines that run together carry one mark between them; it is lit while one of them is */
      const at = rel.fam + ' ' + e.mid[0] + ' ' + e.mid[1], dim = !!sel && !tr.strong.has(i) && !tr.mid.has(i);
      if (!marks[at] || !dim) marks[at] = '<g class="badge b-' + rel.fam + (dim ? ' dim' : '') + '" transform="translate(' + e.mid[0] + ' ' + e.mid[1] + ')"><circle r="6.5"/><path d="' + (rel.fam === 'pos' ? 'M-3 0h6M0-3v6' : 'M-2.4-2.4l4.8 4.8M2.4-2.4l-4.8 4.8') + '"/></g>';
    }
  });
  const badges = Object.values(marks).join('');

  /* nodes. The map is one stop for Tab: the selected page (or the idea) takes the keyboard,
     the arrow keys move from there, and Tab goes on to the tools and the page. A map of 150
     pages is not 150 presses of Tab away from its own page panel. */
  const ideaKey = (store.pages.find((pg) => pg.kind === 'idea') || {}).path;
  const rove = sel && lay.nodes[sel] ? sel : ideaKey && lay.nodes[ideaKey] ? ideaKey : (sc.nodes.find((x) => x.page && lay.nodes[x.key]) || {}).key;
  let nd = '';
  sc.nodes.forEach((sn) => {
    const key = sn.key, n = lay.nodes[key], it = info[key];
    if (!n) return;
    const tab = key === rove ? 0 : -1;
    const col = lay.cols[n.layer], ext = s.ext[key];
    const lw = col.lw, lx = col.slot / 2 + 5;
    const dimCls = sel ? (key === sel ? ' sel' : tr.near.has(key) ? ' near' : tr.kin.has(key) ? ' kin' : ' dim') : '';
    const bg = '<rect class="n-bg" x="' + (-col.slot / 2 - 2) + '" y="' + -ext[0] + '" width="' + (col.slot + 5 + lw + 5) + '" height="' + (ext[0] + ext[1]) + '" rx="4"/>';
    if (it.bundle) {
      const text = fit(it.title, F.src, lw);
      const home = it.bundle.home && store.byPath.get(it.bundle.home);
      const aria = it.title + (home ? ' cited by ' + label(home) : ' that nothing cites yet') + '. Opens them.';
      nd += '<g class="node k-bundle' + dimCls + '" data-key="' + esc(key) + '" data-bundle="1" transform="translate(' + n.x + ' ' + n.y + ')" tabindex="0" role="button" aria-label="' + esc(aria) + '">' + bg +
        glyphInner('bundle', 'accepted', '') + '<rect class="n-knock" x="' + (lx - 2) + '" y="-7" width="' + (tw(text, F.src) + 6) + '" height="14"/><text class="n-l1 n-src n-bundle" x="' + lx + '" y="4">' + esc(text) + '</text></g>';
      return;
    }
    const pg = it.pg;
    let l1 = '', l2 = '', btn = '';
    const decide = (bx, by, name) => '<g class="n-btn n-yes" data-act="accept" data-ref="' + esc(it.st.ref) + '" transform="translate(' + bx + ' ' + by + ')" tabindex="' + tab + '" role="button" aria-label="Accept ' + name + '"><title>Accept (A)</title><rect class="n-hit" x="-4.5" y="-6.5" width="28" height="28"/><rect class="n-face" width="19" height="15" rx="3"/><path d="M5.4 7.9l2.7 2.7 5.4-6.2"/></g>' +
      '<g class="n-btn n-no" data-act="reject-ask" data-ref="' + esc(it.st.ref) + '" data-key="' + esc(pg.path) + '" transform="translate(' + (bx + 28) + ' ' + by + ')" tabindex="' + tab + '" role="button" aria-label="Reject ' + name + '"><title>Reject, with a reason (R)</title><rect class="n-hit" x="-4.5" y="-6.5" width="28" height="28"/><rect class="n-face" width="19" height="15" rx="3"/><path d="M6.1 4.1l6.8 6.8M12.9 4.1l-6.8 6.8"/></g>';
    if (it.tight) {
      /* one line: the mark says the state, the words are in the tip and on the page */
      const held = it.held;
      const idShown = pg.kind === 'result' && it.idW > lw - held + 2 ? it.idText.replace(/^.*· /, '') : it.idText;
      let room = lw - held - (idShown ? it.idW + 5 : 0);
      /* The marks of a page that is itself proposed need no word: its hatching says what they
         decide. On an accepted page they decide a change to it, and the line says which
         before it says the title: "testing → supported", or "→ supported" (the mark shows the
         state it has). The longer of the two that leaves the title room to be read; where
         neither does, the longer that fits, and no stump of a title beside it; then "change".
         A stage drawn at its narrowest has room for the id and the marks only (one word in one
         row is not worth a wider column in every row): there the dashed ring around the mark
         says that a change waits, and the tip and the page say which. */
      const fits = (text) => !!text && tw(text, F.s) + 8 <= room;
      const forms = held && pg.review === 'accepted' ? [it.st.full, it.st.short].filter(fits) : [];
      const says = forms.find((text) => room - tw(text, F.s) - 8 >= 60) || forms[0] || (held && pg.review === 'accepted' && fits('change') ? 'change' : '');
      if (says) room -= tw(says, F.s) + 8;
      const t = it.title && room >= (says ? 60 : 34) ? fit(it.title, F.t, room) : '';
      const end = t ? (idShown ? it.idW + 5 : 0) + tw(t, F.t) : tw(idShown, F.id);
      l1 = '<rect class="n-knock" x="' + (lx - 2) + '" y="-8" width="' + (held ? lw + 4 : end + 6) + '" height="16"/>' +
        (idShown ? '<text class="n-l1" x="' + lx + '" y="4"><tspan class="n-id">' + esc(idShown) + '</tspan></text>' : '') +
        (t ? '<text class="n-l1" x="' + (lx + (idShown ? it.idW + 5 : 0)) + '" y="4">' + esc(t) + '</text>' : '') +
        (says ? '<text class="n-l2" x="' + (lx + lw - 47 - 8 - tw(says, F.s)) + '" y="3.5">' + esc(says) + '</text>' : '');
      if (held) btn = decide(lx + lw - 47, -7.5, esc(label(pg)));
    } else if (pg.kind === 'source') { const t = fit(it.title, F.src, lw); l1 = '<rect class="n-knock" x="' + (lx - 2) + '" y="-7" width="' + (tw(t, F.src) + 5) + '" height="14"/><text class="n-l1 n-src" x="' + lx + '" y="4">' + esc(t) + '</text>'; }
    else if (pg.kind === 'topic') {
      const lines = wrap2(pg.title, F.t, lw).slice(0, it.lines);
      if (lines.length < 2 && it.lines < 2) lines[0] = fit(pg.title, F.t, lw);
      l1 = lines.map((t, i) => '<text class="n-l1" x="' + lx + '" y="' + (lines.length === 1 ? -6 : i ? -6 : -19) + '">' + esc(t) + '</text>').join('');
    } else {
      const room = lw - it.idW - 5;
      let lines = it.title && room >= ROOM ? wrap2(it.title, F.t, room) : [];
      if (lines.length > it.lines) lines = [fit(it.title, F.t, room)];
      const y0 = lines.length > 1 ? -19 : -6;
      const idShown = it.idW > lw + 2 && pg.kind === 'result' ? it.idText.replace(/^.*· /, '') : it.idText;
      l1 = '<text class="n-l1" x="' + lx + '" y="' + y0 + '"><tspan class="n-id">' + esc(idShown) + '</tspan></text>' +
        lines.map((t, i) => '<text class="n-l1" x="' + (lx + it.idW + 5) + '" y="' + (y0 + i * 13) + '">' + esc(t) + '</text>').join('');
    }
    if (!it.tight && pg.kind !== 'topic' && pg.kind !== 'source' && it.st.full) {
      const room = lw - (it.st.ref ? BTN : 0);
      const fits = (text) => tw(text, F.s) <= room;
      /* A state is said in whole words or not at all: "prop…" tells nobody anything. Where a
         column is too narrow for the word and the two marks, the marks stay: the hatching
         and the dashed outline already say that the page waits for a decision. */
      const words = fits(it.st.full) ? it.st.full : fits(it.st.short) ? it.st.short
        : it.st.ref ? (pg.review === 'accepted' && fits('change') ? 'change' : '')
        : tw(it.st.short, F.s) <= lw + 8 ? it.st.short : fit(it.st.short, F.s, Math.max(room, 30));
      l2 = words ? '<text class="n-l2" x="' + lx + '" y="11.5">' + esc(words) + '</text>' : '';
      if (it.st.ref) {
        /* the marks are small; what is pressed is a square around each: 28 units, which is
           24 px where the map is drawn at its smallest automatic scale */
        btn = decide(lx + (words ? tw(words, F.s) + 8 : 1), 1, esc(label(pg)));
      }
    }
    const change = store.pendingFor(pg.path).length > 0;
    const cls = 'node k-' + pg.kind + ' r-' + (pg.review === 'proposed' || pg.review === 'rejected' ? pg.review : 'accepted') + ' t-' + (TONE[pg.state] || 'none') + (change ? ' has-change' : '') + (it.tight ? ' n-tight' : '') + dimCls +
      (flash && flash.key === pg.path ? ' flash-' + flash.decision : '') + (prev && !prev.has(pg.path) ? ' pop' : '');
    const aria = kindName(pg.kind) + ' ' + (pg.id && pg.kind !== 'source' ? pg.id : '') + ', ' + pg.title + ', ' + pg.review + (pg.state ? ', ' + pg.state : '') + (change ? ', change proposed' : '');
    nd += '<g class="' + cls + '" data-key="' + esc(pg.path) + '" transform="translate(' + n.x + ' ' + n.y + ')" tabindex="' + tab + '" role="button" aria-label="' + esc(aria) + '"' + (pg.path === sel ? ' aria-pressed="true"' : '') + '>' +
      bg + glyphInner(pg.kind, pg.review, pg.state, { change }) + l1 + l2 + btn + '</g>';
  });
  V.prevKeys[store.pid] = new Set(sc.nodes.filter((n) => n.page).map((n) => n.key));

  /* an empty project shows the route still to be built */
  let ghost = '';
  const ideaPath = (store.pages.find((pg) => pg.kind === 'idea') || {}).path;
  if (empty) {
    const y = ideaPath && lay.nodes[ideaPath] ? lay.nodes[ideaPath].y : H / 2;
    ghost = '<path class="ghost-line" d="M' + (lay.cols[0].gx + 12) + ' ' + y + 'H' + lay.cols[7].gx + '"/>';
    lay.cols.slice(1).forEach((c) => {
      ghost += '<g class="ghost" transform="translate(' + c.gx + ' ' + y + ')"><circle class="ghost-bg" r="11"/>' + glyphInner(c.kind, 'accepted', '') +
        '<text y="28" text-anchor="middle">' + esc(KIND[c.kind].hint) + '</text></g>';
    });
  }

  const focused = document.activeElement && svg.contains(document.activeElement) ? document.activeElement : null;
  const held = focused && focused.closest('.node') ? { key: focused.closest('.node').dataset.key, act: focused.classList.contains('n-btn') ? focused.dataset.act : '' } : null;
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  /* built from numbers, class names and escaped text only */
  svg.innerHTML =
    '<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect class="hatch-bg" width="5" height="5"/><path class="hatch-ln" d="M0 0v5"/></pattern></defs>' +
    '<g class="vp"><g class="ghosts">' + ghost + '</g><g class="edges">' + ed + '</g><g class="refs"></g><g class="hits">' + hits + '</g><g class="badges">' + badges + '</g><g class="nodes">' + nd + '</g></g>' +
    '<rect class="head-bg" width="' + W + '" height="34"/><path class="head-rule" d="M0 34.5H' + W + '"/><g class="heads">' + heads + '</g>';
  /* The drawing was written again, and with it the node that had the keyboard: the keyboard
     is put back where it was. When the arrow keys moved the selection from a node, it goes
     along to the new one. */
  const wanted = V.focusNext || (held && held.key);
  if (wanted) {
    let el = svg.querySelector('.node[data-key="' + CSS.escape(wanted) + '"]');
    if (el && !V.focusNext && held.act) el = el.querySelector('.n-btn[data-act="' + held.act + '"]') || el;
    if (el && document.activeElement !== el && (V.focusNext || !document.activeElement || document.activeElement === document.body)) el.focus({ preventScroll: true });
    if (el) V.focusNext = null;
  }
  canvas.classList.toggle('has-sel', !!sel);
  canvas.classList.toggle('dense', dense);
  pointAt(V.pointed);
  canvas.classList.toggle('wider', lay.width * s.k > W + 2 || (lay.bottomY + 12) * s.k > H - bottom + 2);
  canvas.classList.toggle('trace-more', !!(sel && tr.more));

  const emptyEl = $('.map-empty', root);
  setHTML(emptyEl, empty ? emptyHTML(store) : sc.hidden && !sc.nodes.some((n) => n.page && n.page.kind !== 'idea') ? html`<div class="empty-card"><p>Nothing is drawn with these filters.</p><div class="row"><button class="btn" data-act="map-reset">Show everything</button></div></div>` : html``);
  if (emptyEl.firstChild) emptyEl.style.top = (empty && ideaPath && lay.nodes[ideaPath] ? lay.nodes[ideaPath].y + 58 : H * 0.36) + 'px';
  applyT();
}

function keyHTML(refsOnSelection) {
  const sw = (cls, lab, note) => html`<span class="key-i"><svg width="30" height="12" viewBox="0 0 30 12" aria-hidden="true"><path class="edge ${cls}" d="M1 6H29"/>${
    cls.indexOf('f-pos') >= 0 ? raw('<g class="badge b-pos" transform="translate(15 6) scale(.8)"><circle r="6.5"/><path d="M-3 0h6M0-3v6"/></g>') : cls.indexOf('f-neg') >= 0 ? raw('<g class="badge b-neg" transform="translate(15 6) scale(.8)"><circle r="6.5"/><path d="M-2.4-2.4l4.8 4.8M2.4-2.4l-4.8 4.8"/></g>') : ''}</svg>${lab}${note ? html` <i class="key-note">${note}</i>` : ''}</span>`;
  const g = (review, state, lab) => html`<span class="key-i">${glyph('hypothesis', review, state, 16)}${lab}</span>`;
  return html`<div class="key-row"><span class="key-h">Review</span>${g('accepted', 'open', 'accepted')}${g('proposed', 'open', 'proposed')}${g('rejected', 'open', 'rejected')}</div>
    <div class="key-row key-state"><span class="key-h">State</span>${g('accepted', 'open', 'open')}${g('accepted', 'testing', 'testing, running')}${g('accepted', 'supported', 'supported, succeeded')}${g('accepted', 'refuted', 'refuted, failed')}</div>
    <div class="key-row"><span class="key-h">Lines</span>${sw('f-read', refsOnSelection ? 'cites, about, lines to sources' : 'cites, about', refsOnSelection ? 'on selection' : '')}${sw('f-reason', 'motivated by, follows')}${sw('f-test', 'tests, run of, based on')}${sw('f-pos', 'supports')}${sw('f-neg', 'refutes')}</div>`;
}

function emptyHTML(store) {
  const e = engineView(store);
  const files = (store.project.attachments || []).length;
  const limits = (store.project.settings || {}).limits || {};
  const idea = store.pages.find((pg) => pg.kind === 'idea');
  if (e.mode === 'running') return html`<div class="empty-card"><h2>Research has started.</h2><p>Intake reads ${files ? 'your ' + plural(files, 'file') + ' and ' : ''}the idea, then the scout looks for prior work. Pages appear here as they are written.</p></div>`;
  return html`<div class="empty-card"><h2>The map is empty beyond your idea.</h2>
    <p>${e.fresh ? 'Research has not started.' : 'Nothing has been written yet.'} Intake reads ${files ? 'your ' + plural(files, 'file') : 'the idea'}, then the scout looks for prior work, verifies every paper and writes down what is known and what is open. It pauses at the first checkpoint and nothing is accepted without you.</p>
    <p class="empty-meta">The limit for models in this project is $${limits.model_usd != null ? Number(limits.model_usd).toFixed(2) : '…'}.</p>
    <div class="row"><button class="btn btn-primary" data-act="continue">${e.fresh ? 'Start research' : continueLabel(e.next, store)} ${icon('right')}</button>${idea ? html`<a class="btn" href="${link({ view: 'wiki', sel: idea.path, sub: 'edit' })}">Edit the idea first</a>` : ''}</div></div>`;
}

/** The reference lines of the page the pointer or the keyboard is on, while they are held
    back and nothing is selected. A selection has its own drawn; beside them, the lines of a
    second page could not be told from its lineage, and would end at pages that are dimmed. */
function pointAt(key) {
  const s = V.st, g = V.root && $('.map-svg .refs', V.root);
  V.pointed = key || '';
  if (!g) return;
  const own = key && !V.sel && s.lay && s.refs && s.refs.length ? routeRefs(s.lay, key, s.refs.filter((e) => e.s === key || e.t === key)) : [];
  const review = (k) => (V.store.byPath.get(k) || {}).review;
  const state = (e) => (review(e.s) === 'rejected' || review(e.t) === 'rejected' ? ' e-dead' : review(e.s) === 'proposed' || review(e.t) === 'proposed' ? ' e-prop' : '');
  g.replaceChildren(...own.map((e) => {
    const path = document.createElementNS(g.namespaceURI, 'path');
    path.setAttribute('class', 'edge f-' + relOf(e.type).fam + ' t-' + String(e.type).replace(/[^a-z_]/g, '') + state(e) + ' ref');
    path.setAttribute('d', e.d);
    return path;
  }));
}

function tip(key) {
  const store = V.store, n = V.st.lay && V.st.lay.nodes[key], el = $('.map-tip', V.root);
  if (!n || key === V.route.sel) return hideTip();
  const bundle = V.st.bundles && V.st.bundles[key];
  if (bundle) {
    const home = bundle.home && store.byPath.get(bundle.home);
    const names = bundle.members.map((p) => shortTitle(store.byPath.get(p)));
    setHTML(el, html`<div class="tip-k">${plural(bundle.members.length, 'source')}${home ? ' · cited by ' + label(home) : ' · not cited yet'}</div><div class="tip-s">${names.slice(0, 8).join(', ')}${names.length > 8 ? ' and ' + (names.length - 8) + ' more' : ''}</div><div class="tip-s">Select ${home ? 'the ' + kindName(home.kind).toLowerCase() : 'it'} to unfold them.</div>`);
  } else {
    const pg = store.byPath.get(key);
    if (!pg) return hideTip();
    const st = statusLine(store, pg).full;
    setHTML(el, html`<div class="tip-k">${kindName(pg.kind)}${pg.id && pg.kind !== 'source' ? ' ' + pg.id : ''}${st ? ' · ' + st : ''}</div><div class="tip-t">${pg.title}</div>${pg.summary && pg.summary !== pg.title ? html`<div class="tip-s">${pg.summary}</div>` : ''}`);
  }
  show(el, V.st.tx + V.st.k * (n.x + 14), V.st.ty + V.st.k * (n.y + 16));
}
function edgeTip(e, ev) {
  const store = V.store, el = $('.map-tip', V.root);
  const name = (key) => { const pg = store.byPath.get(key); if (pg) return label(pg); const b = V.st.bundles && V.st.bundles[key]; return b ? plural(b.members.length, 'source') : key; };
  setHTML(el, html`<div class="tip-e">${name(e.dataset.s)} <b>${relOf(e.dataset.type).out}</b> ${name(e.dataset.t)}</div>`);
  show(el, ev.offsetX + 12, ev.offsetY + 14);
}
function show(el, x, y) {
  const c = $('.map-canvas', V.root);
  el.classList.add('on');
  el.style.left = Math.max(8, Math.min(x, c.clientWidth - el.offsetWidth - 8)) + 'px';
  el.style.top = Math.max(40, Math.min(y, c.clientHeight - el.offsetHeight - 8)) + 'px';
}
function hideTip() { const el = V.root && $('.map-tip', V.root); if (el) el.classList.remove('on'); }

export const mapView = {
  render(root, route) {
    const store = app.store;
    const first = root.firstElementChild;
    if (!first || first.dataset.pid !== store.pid || !first.classList.contains('map-view')) mount(root, store);
    V.root = root; V.store = store; V.route = route;
    draw();
  },
  zoom,
  /** Arrow keys walk the map: left and right along the lines, up and down within a stage. */
  key(ev) {
    const lay = V.st.lay, store = V.store;
    if (!lay) return false;
    const sel = V.route.sel;
    if (ev.key === '+' || ev.key === '=') { zoom(1); return true; }
    if (ev.key === '-') { zoom(-1); return true; }
    if (ev.key === '0') { zoom(0); return true; }
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].indexOf(ev.key) < 0) return false;
    if (!sel || !lay.nodes[sel]) {
      const idea = store.pages.find((pg) => pg.kind === 'idea');
      const active = document.activeElement;
      if (idea && active && active.closest && active.closest('.map-svg, .map-canvas') && !active.closest('.map-foot')) { V.focusNext = idea.path; active.blur(); }
      if (idea) select(idea.path);
      return true;
    }
    const me = lay.nodes[sel];
    let to = null;
    const isPage = (k) => store.byPath.has(k);
    if (ev.key === 'ArrowUp' || ev.key === 'ArrowDown') {
      const colKeys = lay.columnOrder[me.layer].filter(isPage), i = colKeys.indexOf(sel);
      to = colKeys[i + (ev.key === 'ArrowDown' ? 1 : -1)];
    } else {
      const right = ev.key === 'ArrowRight';
      const cand = (V.st.drawn || lay.edges).filter((e) => (right ? e.lo : e.hi) === sel).map((e) => (right ? e.hi : e.lo)).filter(isPage);
      /* the nearest stage first, then the nearest line */
      cand.sort((a, b) => Math.abs(lay.nodes[a].layer - me.layer) - Math.abs(lay.nodes[b].layer - me.layer) || Math.abs(lay.nodes[a].y - me.y) - Math.abs(lay.nodes[b].y - me.y));
      to = cand[0];
      if (!to) { const next = (lay.columnOrder[me.layer + (right ? 1 : -1)] || []).filter(isPage); to = next.slice().sort((a, b) => Math.abs(lay.nodes[a].y - me.y) - Math.abs(lay.nodes[b].y - me.y))[0]; }
    }
    if (to) {
      const active = document.activeElement;
      if (active && active.closest && active.closest('.map-svg, .map-canvas') && !active.closest('.map-foot')) { V.focusNext = to; active.blur(); }
      select(to);
    }
    return true;
  },
  /** Forget pan and zoom (a new filter, a new project). */
  reset() { V.st.user = false; V.st.fitted = false; },
  /** Whether sources are drawn folded into bundles at the moment. */
  folded() { return !!V.st.folded; },
};
