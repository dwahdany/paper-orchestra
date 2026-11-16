/* The map layout.
   A layered (Sugiyama-style) layout with the layer fixed by the kind of page, so the drawing
   reads left to right the way research flows. Steps:
     1. long edges get a dummy node in every layer they pass, so they run in their own lane
     2. order inside layers: barycentre sweeps + adjacent transpositions, fewest crossings kept;
        the ordering of the previous layout is the starting order, so that the drawing does
        not jump when one page is added
     3. vertical positions: neighbours pull, minimum separations hold (isotonic regression)
     4. long edges are straightened where their lane is free
     5. gutters between layers are routed orthogonally: edges that share an endpoint share a
        trunk, trunks get tracks by interval colouring, track order is chosen for fewest crossings
     6. column widths answer to the width available: labels shrink before anything overlaps
   Pure function: no DOM. Written to stay quick at a few hundred nodes. */

import { STAGES } from './vocab.js';

const LANE = 7, TRACK = 6, CHAMFER = 5;
const avg = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

/* crossings between two orderings: inversions of the lower positions, edges sorted by the upper */
function inversions(pairs) {
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let size = 0;
  for (const p of pairs) if (p[1] > size) size = p[1];
  const tree = new Int32Array(size + 2);
  let count = 0, seen = 0;
  for (const p of pairs) {
    /* how many earlier edges end strictly below this one */
    let i = p[1] + 1, le = 0;
    for (; i > 0; i -= i & -i) le += tree[i];
    count += seen - le;
    for (i = p[1] + 1; i < tree.length; i += i & -i) tree[i]++;
    seen++;
  }
  return count;
}

export function layout(nodes, edges, opt) {
  const L = STAGES.length;
  const N = {};
  const layers = Array.from({ length: L }, () => []);
  const prev = opt.prevOrder || null;
  nodes.forEach((n) => {
    const r = { key: n.key, layer: n.layer == null ? STAGES.indexOf(n.kind) : n.layer, real: true, hu: n.hu, hd: n.hd, w: n.w, min: n.min, hard: n.hard == null ? n.min : n.hard, r: n.r, up: [], dn: [], y: 0 };
    N[n.key] = r;
    layers[r.layer].push(r);
  });
  if (prev) {
    layers.forEach((arr) => {
      arr.forEach((n, i) => { n.seed = prev[n.key] == null ? 1e6 + i : prev[n.key]; });
      arr.sort((a, b) => a.seed - b.seed);
    });
  }

  /* 1. chains: every edge becomes a path from its lower layer to its higher layer */
  const chains = [], intra = [];
  let dummy = 0;
  edges.forEach((e) => {
    let a = N[e.s], b = N[e.t];
    if (!a || !b || a === b) return;
    if (a.layer === b.layer) { intra.push({ e, a, b }); return; }
    if (a.layer > b.layer) { const t = a; a = b; b = t; }
    const path = [a];
    for (let l = a.layer + 1; l < b.layer; l++) {
      const d = { key: '~' + dummy++, layer: l, real: false, hu: 0, hd: 0, up: [], dn: [], y: 0 };
      layers[l].push(d);
      path.push(d);
    }
    path.push(b);
    for (let i = 0; i < path.length - 1; i++) { path[i].dn.push(path[i + 1]); path[i + 1].up.push(path[i]); }
    chains.push({ e, path });
  });
  if (prev) {
    /* a lane starts where its ends were: between the neighbours it connects */
    layers.forEach((arr) => arr.forEach((n, i) => (n.pos = i)));
    layers.forEach((arr, l) => {
      const reals = arr.filter((n) => n.real).length, above = l ? Math.max(1, layers[l - 1].length) : 1;
      let r = 0;
      arr.forEach((n) => { n.seed = n.real ? r++ : (n.up.length ? (avg(n.up.map((x) => x.pos)) / above) * reals : reals) + 0.5; });
      if (l) arr.sort((a, b) => a.seed - b.seed);
      arr.forEach((n, i) => (n.pos = i));
    });
  }

  /* 2. ordering */
  const setPos = () => layers.forEach((arr) => arr.forEach((n, i) => (n.pos = i)));
  setPos();
  const crossBetween = (l) => {
    const pairs = [];
    layers[l].forEach((u) => u.dn.forEach((v) => pairs.push([u.pos, v.pos])));
    return pairs.length > 1 ? inversions(pairs) : 0;
  };
  const crossings = () => { let c = 0; for (let l = 0; l < L - 1; l++) c += crossBetween(l); return c; };
  const sortLayer = (l, side) => {
    layers[l].forEach((n) => {
      const nb = side === 'up' ? n.up : n.dn;
      const all = n.up.length + n.dn.length;
      n.bc = nb.length ? avg(nb.map((x) => x.pos)) : all ? n.pos : 1e6 + n.pos; // unconnected pages sink
    });
    layers[l] = layers[l].map((n, i) => [n, i]).sort((a, b) => a[0].bc - b[0].bc || a[1] - b[1]).map((x) => x[0]);
    layers[l].forEach((n, i) => (n.pos = i));
  };
  let best = null, bestC = Infinity;
  const keep = () => { const c = crossings(); if (c < bestC) { bestC = c; best = layers.map((arr) => arr.slice()); } };
  keep();
  const total = layers.reduce((a, arr) => a + arr.length, 0);
  const sweeps = total > 900 ? 4 : total > 400 ? 8 : 14;
  for (let it = 0; it < sweeps && bestC > 0; it++) {
    for (let l = 1; l < L; l++) sortLayer(l, 'up');
    keep();
    for (let l = L - 2; l >= 0; l--) sortLayer(l, 'dn');
    keep();
  }
  best.forEach((arr, l) => (layers[l] = arr.filter((n) => n.up.length + n.dn.length).concat(arr.filter((n) => !(n.up.length + n.dn.length)))));
  setPos();
  /* adjacent transpositions: only the edges of the two nodes can change their crossings */
  const pairCross = (xs, ys) => { let c = 0; for (const x of xs) for (const y of ys) if (x.pos > y.pos) c++; return c; };
  for (let pass = 0, improved = true; improved && pass < 8; pass++) {
    improved = false;
    for (let l = 0; l < L; l++) {
      const arr = layers[l];
      for (let i = 0; i < arr.length - 1; i++) {
        const a = arr[i], b = arr[i + 1];
        if (!(a.up.length + a.dn.length) || !(b.up.length + b.dn.length)) continue;
        const before = pairCross(a.up, b.up) + pairCross(a.dn, b.dn);
        if (!before) continue;
        const after = pairCross(b.up, a.up) + pairCross(b.dn, a.dn);
        if (after < before) { arr[i] = b; arr[i + 1] = a; b.pos = i; a.pos = i + 1; improved = true; }
      }
    }
  }
  bestC = crossings();

  /* 3. vertical positions */
  const availH = Math.max(200, opt.H - opt.top - opt.bottom);
  const baseSep = (a, b, G) => (!a.real && !b.real ? LANE : !a.real ? b.hu + 6 : !b.real ? a.hd + 6 : a.hd + b.hu + G);
  let G = opt.gapMax;
  layers.forEach((arr) => {
    let base = 0, rr = 0;
    for (let i = 1; i < arr.length; i++) { base += baseSep(arr[i - 1], arr[i], 0); if (arr[i - 1].real && arr[i].real) rr++; }
    if (rr) G = Math.min(G, (availH * 0.94 - base - 30) / rr);
  });
  G = Math.max(opt.gapMin == null ? 3 : opt.gapMin, Math.min(opt.gapMax, G));
  const sep = (a, b) => baseSep(a, b, G);
  layers.forEach((arr) => {
    let y = 0;
    arr.forEach((n, i) => { if (i) y += sep(arr[i - 1], n); n.y = y; });
    const mid = y / 2;
    arr.forEach((n) => (n.y -= mid));
  });
  const place = (arr, want) => {
    /* minimise sum w (y - want)^2 subject to y[i] - y[i-1] >= sep: pool adjacent violators */
    const off = [0];
    for (let i = 1; i < arr.length; i++) off[i] = off[i - 1] + sep(arr[i - 1], arr[i]);
    const blocks = [];
    arr.forEach((n, i) => {
      blocks.push({ sum: want[i].w * (want[i].t - off[i]), w: want[i].w, n: 1 });
      while (blocks.length > 1) {
        const b = blocks[blocks.length - 1], a = blocks[blocks.length - 2];
        if (a.sum / a.w <= b.sum / b.w) break;
        a.sum += b.sum; a.w += b.w; a.n += b.n; blocks.pop();
      }
    });
    let i = 0;
    blocks.forEach((b) => { for (let k = 0; k < b.n; k++, i++) arr[i].y = b.sum / b.w + off[i]; });
  };
  const relax = (l, side) => {
    const arr = layers[l];
    const conn = arr.filter((n) => n.up.length + n.dn.length);
    if (conn.length) {
      place(conn, conn.map((n) => {
        let nb = side === 'up' ? n.up : side === 'dn' ? n.dn : n.up.concat(n.dn);
        if (!nb.length) nb = n.up.concat(n.dn);
        return { t: avg(nb.map((x) => x.y)), w: n.real ? 1 : 3 };
      }));
    }
    /* pages nothing links to yet are packed after the rest of their stage */
    arr.forEach((n, i) => { if (!(n.up.length + n.dn.length) && i > 0 && conn.length) n.y = arr[i - 1].y + sep(arr[i - 1], n); });
  };
  const rounds = total > 900 ? 3 : 8;
  for (let it = 0; it < rounds; it++) {
    for (let l = 1; l < L; l++) relax(l, 'up');
    for (let l = L - 2; l >= 0; l--) relax(l, 'dn');
  }
  for (let it = 0; it < 6; it++) for (let l = 0; l < L; l++) relax(l, 'both');

  /* 4. straighten long edges where the lane is free */
  const straighten = () => chains.filter((c) => c.path.length > 2).sort((a, b) => b.path.length - a.path.length).forEach((c) => {
    const ds = c.path.slice(1, -1);
    let lo = -Infinity, hi = Infinity;
    ds.forEach((d) => {
      const arr = layers[d.layer], i = d.pos;
      if (i > 0) lo = Math.max(lo, arr[i - 1].y + sep(arr[i - 1], d));
      if (i < arr.length - 1) hi = Math.min(hi, arr[i + 1].y - sep(d, arr[i + 1]));
    });
    if (lo > hi) return;
    const a = c.path[0].y, b = c.path[c.path.length - 1].y;
    const y = a >= lo && a <= hi ? a : b >= lo && b <= hi ? b : Math.max(lo, Math.min(hi, avg(ds.map((d) => d.y))));
    ds.forEach((d) => (d.y = y));
  });
  straighten(); straighten();
  /* a page with one neighbour sits level with it when it can */
  layers.forEach((arr) => arr.forEach((n, i) => {
    if (!n.real) return;
    const nb = n.up.concat(n.dn);
    if (nb.length !== 1) return;
    const lo = i > 0 ? arr[i - 1].y + sep(arr[i - 1], n) : -Infinity;
    const hi = i < arr.length - 1 ? arr[i + 1].y - sep(n, arr[i + 1]) : Infinity;
    if (nb[0].y >= lo && nb[0].y <= hi) n.y = nb[0].y;
  }));
  straighten();

  let minY = Infinity, maxY = -Infinity;
  layers.forEach((arr) => arr.forEach((n) => { minY = Math.min(minY, n.y - n.hu); maxY = Math.max(maxY, n.y + n.hd); }));
  if (minY === Infinity) { minY = 0; maxY = 0; }
  const height = maxY - minY;
  const shiftY = opt.top + Math.max(10, (availH - height) / 2) - minY;
  layers.forEach((arr) => arr.forEach((n) => (n.y = Math.round((n.y + shiftY) * 2) / 2)));

  /* 5. gutters: nets, tracks, track order */
  const gutters = [];
  for (let g = 0; g < L - 1; g++) {
    const segs = [], byEnds = new Map();
    layers[g].forEach((u) => u.dn.forEach((v) => { const s = { u, v, net: null }; segs.push(s); byEnds.set(u.key + '>' + v.key, s); }));
    let rest = segs.filter((s) => Math.abs(s.u.y - s.v.y) >= 0.75);
    const nets = [];
    while (rest.length) {
      const cnt = {};
      rest.forEach((s) => { cnt['v' + s.v.key] = (cnt['v' + s.v.key] || 0) + 1; cnt['u' + s.u.key] = (cnt['u' + s.u.key] || 0) + 1; });
      let bk = null;
      Object.keys(cnt).forEach((k) => { if (!bk || cnt[k] > cnt[bk] || (cnt[k] === cnt[bk] && k[0] === 'v' && bk[0] === 'u')) bk = k; });
      const mem = rest.filter((s) => (bk[0] === 'v' ? 'v' + s.v.key : 'u' + s.u.key) === bk);
      let lo = Infinity, hi = -Infinity;
      mem.forEach((s) => { lo = Math.min(lo, s.u.y, s.v.y); hi = Math.max(hi, s.u.y, s.v.y); });
      const net = { mem, lo, hi, track: 0, x: 0 };
      mem.forEach((s) => (s.net = net));
      nets.push(net);
      rest = rest.filter((s) => !s.net);
    }
    nets.sort((a, b) => a.lo - b.lo);
    const ends = [];
    nets.forEach((net) => {
      let t = ends.findIndex((hi) => hi + 8 < net.lo);
      if (t < 0) { t = ends.length; ends.push(0); }
      ends[t] = net.hi;
      net.track = t;
    });
    const T = ends.length;
    let order = Array.from({ length: T }, (_, i) => i);
    const cost = (perm) => {
      let c = 0;
      nets.forEach((n1) => {
        const p1 = perm[n1.track];
        n1.mem.forEach((s) => nets.forEach((n2) => {
          if (n2 === n1) return;
          const p2 = perm[n2.track];
          if (p2 < p1 && s.u.y > n2.lo + 1 && s.u.y < n2.hi - 1) c++;
          if (p2 > p1 && s.v.y > n2.lo + 1 && s.v.y < n2.hi - 1) c++;
        }));
      });
      return c;
    };
    if (T > 1 && T <= 6 && nets.length <= 60) {
      let bestP = order.slice(), bestCost = cost(order);
      const permute = (arr, k) => {
        if (k === arr.length) { const c = cost(arr); if (c < bestCost) { bestCost = c; bestP = arr.slice(); } return; }
        for (let i = k; i < arr.length; i++) { [arr[k], arr[i]] = [arr[i], arr[k]]; permute(arr, k + 1); [arr[k], arr[i]] = [arr[i], arr[k]]; }
      };
      permute(order.slice(), 0);
      order = bestP;
    }
    nets.forEach((net) => (net.slot = order[net.track]));
    gutters.push({ segs, byEnds, nets, T });
  }

  /* 6. horizontal: columns answer to the width available */
  const cols = [];
  for (let l = 0; l < L; l++) {
    const real = layers[l].filter((n) => n.real);
    const kind = STAGES[l];
    const nat = real.length ? Math.max.apply(null, real.map((n) => n.w)) : opt.min[kind];
    const floor = (opt.colMin && opt.colMin[kind]) || 0;
    const min = Math.max(floor, real.length ? Math.max.apply(null, real.map((n) => n.min)) : opt.min[kind]);
    const hard = Math.max(floor, real.length ? Math.max.apply(null, real.map((n) => n.hard)) : opt.min[kind]);
    cols.push({ kind, n: real.length, slot: opt.slot[kind] || 22, nat: Math.max(nat, min), min, hard });
  }
  const trackW = (g) => Math.min(TRACK, 90 / Math.max(1, g.T - 1));
  const gw = gutters.map((g) => 22 + Math.max(0, g.T - 1) * trackW(g));
  const fixed = opt.padL + opt.padR + cols.reduce((a, c) => a + c.slot + 5 + 4, 0) + gw.reduce((a, b) => a + b, 0);
  const natSum = cols.reduce((a, c) => a + c.nat, 0), minSum = cols.reduce((a, c) => a + c.min, 0), hardSum = cols.reduce((a, c) => a + c.hard, 0);
  /* labels give way in two steps: titles wrap and shorten first, keys are cut only after that */
  let extra = opt.W - fixed - natSum, f = 1, f2 = 1, grow = 0, lead = 0;
  if (extra < 0) {
    f = Math.max(0, Math.min(1, (opt.W - fixed - minSum) / Math.max(1, natSum - minSum)));
    if (opt.W - fixed - minSum < 0) f2 = Math.max(0, Math.min(1, (opt.W - fixed - hardSum) / Math.max(1, minSum - hardSum)));
  } else { grow = Math.min(opt.gutterGrow, extra / (L - 1)); lead = (extra - grow * (L - 1)) / 2; }
  let x = opt.padL + lead;
  cols.forEach((c, l) => {
    c.lw = Math.floor(f2 < 1 ? c.hard + f2 * (c.min - c.hard) : c.min + f * (c.nat - c.min));
    c.x0 = x; c.gx = x + c.slot / 2; c.lx = x + c.slot + 5; c.px = c.lx + c.lw + 4;
    x = c.px;
    if (l < L - 1) {
      const w = gw[l] + grow, g = gutters[l];
      g.x0 = x; g.x1 = x + w;
      const step = trackW(g), span = Math.max(0, g.T - 1) * step;
      g.nets.forEach((net) => (net.x = Math.round(x + (w - span) / 2 + net.slot * step)));
      x += w;
    }
  });
  const width = x + opt.padR + lead;

  /* 7. paths */
  const dOf = (pts) => {
    const out = [pts[0]];
    for (let i = 1; i < pts.length - 1; i++) {
      const p = pts[i], a = pts[i - 1], b = pts[i + 1];
      const l1 = Math.abs(p[0] - a[0]) + Math.abs(p[1] - a[1]), l2 = Math.abs(b[0] - p[0]) + Math.abs(b[1] - p[1]);
      if (!l1 || !l2) continue;
      const r = Math.min(CHAMFER, l1 / 2, l2 / 2);
      out.push([p[0] + ((a[0] - p[0]) / l1) * r, p[1] + ((a[1] - p[1]) / l1) * r], [p[0] + ((b[0] - p[0]) / l2) * r, p[1] + ((b[1] - p[1]) / l2) * r]);
    }
    out.push(pts[pts.length - 1]);
    return 'M' + out.map((p) => Math.round(p[0] * 10) / 10 + ' ' + Math.round(p[1] * 10) / 10).join('L');
  };
  const outEdges = chains.map((c) => {
    const a = c.path[0], b = c.path[c.path.length - 1];
    const pts = [[cols[a.layer].gx + a.r, a.y]];
    for (let i = 0; i < c.path.length - 1; i++) {
      const u = c.path[i], v = c.path[i + 1];
      const seg = gutters[u.layer].byEnds.get(u.key + '>' + v.key);
      if (seg && seg.net) pts.push([seg.net.x, u.y], [seg.net.x, v.y]);
    }
    pts.push([cols[b.layer].gx - b.r, b.y]);
    /* the longest level run carries the badge or the label */
    let mid = null, len = -1;
    for (let i = 0; i < pts.length - 1; i++) {
      if (pts[i][1] !== pts[i + 1][1]) continue;
      const l = Math.abs(pts[i + 1][0] - pts[i][0]);
      if (l > len) { len = l; mid = [(pts[i][0] + pts[i + 1][0]) / 2, pts[i][1]]; }
    }
    if (c.path.length > 2) { const d = c.path[Math.floor(c.path.length / 2)]; mid = [(cols[d.layer].x0 + cols[d.layer].px) / 2, d.y]; }
    return { s: c.e.s, t: c.e.t, type: c.e.type, lo: a.key, hi: b.key, d: dOf(pts), mid };
  });
  intra.forEach((c) => {
    const col = cols[c.a.layer], x0 = col.gx - c.a.r, bend = col.x0 - 9;
    outEdges.push({ s: c.e.s, t: c.e.t, type: c.e.type, lo: c.a.key, hi: c.b.key, d: dOf([[x0, c.a.y], [bend, c.a.y], [bend, c.b.y], [col.gx - c.b.r, c.b.y]]), mid: [bend, (c.a.y + c.b.y) / 2] });
  });

  const outNodes = {};
  layers.forEach((arr, l) => arr.forEach((n) => { if (n.real) outNodes[n.key] = { x: cols[l].gx, y: n.y, layer: l, pos: n.pos }; }));
  const columnOrder = layers.map((arr) => arr.filter((n) => n.real).map((n) => n.key));
  const order = {};
  layers.forEach((arr) => arr.forEach((n, i) => { if (n.real) order[n.key] = i; }));
  return { nodes: outNodes, cols, edges: outEdges, width, height: maxY - minY + opt.top + opt.bottom, bottomY: maxY + shiftY, crossings: bestC, columnOrder, order, G };
}
