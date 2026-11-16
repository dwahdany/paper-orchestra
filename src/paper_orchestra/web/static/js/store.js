/* One in-memory store per open project.

   It is filled from the project view, the pages (front matter only), the graph, the edit
   proposals, the requests, the experiments and the recent activity. Page bodies and
   experiment details are fetched when they are shown and cached until the page changes.
   The event stream patches the store: activity lines are appended, log lines go straight to
   the open log, and anything that changes the wiki triggers one coalesced refetch of the
   (cheap) lists. On a reconnect everything is fetched again.

   Decisions are optimistic. A decision sits in `overlay` from the moment it is made; the
   derived state already shows it. It is sent after a short grace period (the only "undo"
   there is: the API has none), and rolled back with the server's message if it is refused. */

import { api, Unreachable } from './api.js';
import { askHealth } from './core.js';
import { STAGES, relOf } from './vocab.js';

const VERDICT = { supports: 1, refutes: 1, answers: 1 };
const STRUCTURAL = /^(page\.|review\.decided|proposal\.created|experiment\.state|run\.started|run\.finished|request\.(created|withdrawn)|phase\.|checkpoint|budget\.warning|error|paper\.|project\.)/;
const MAX_ACTIVITY = 600;

export class Store {
  constructor(pid, onChange) {
    this.pid = pid;
    this.onChange = onChange;            // (reason, background) -> void
    this.loaded = false;
    this.loadError = null;
    this.server = null;                  // the lists as the server last gave them
    this.project = null;
    this.pages = [];
    this.byPath = new Map();
    this.refs = new Map();
    this.edges = [];
    this.proposals = [];                 // pending edit proposals (after the overlay)
    this.requests = [];
    this.experiments = [];
    this.expById = new Map();
    this.activity = [];                  // oldest first
    this.lastSeq = 0;
    this.moreActivity = false;
    this.views = new Map();              // path -> {key, data, loading, error}
    this.exps = new Map();               // experiment id -> {key, data, loading, error}
    this.versions = new Map();           // path@stamp -> {data, loading, error}
    this.overlay = new Map();            // ref -> a decision not yet confirmed by a refetch
    this.pendingItems = [];
    this.changeFor = new Map();
    this.adj = new Map();
    this.tails = {};                     // experiment id -> the last log line seen
    this.workerLog = null;
    this.paper = null;
    this.live = 'connecting';            // connecting | live | reconnecting
    this.source = null;
    this.closed = false;
    this.refreshError = null;
    this._timer = 0;
    this._busy = false;
    this._again = false;
    /* The state comes with the event stream; what a press of Continue will do (`next_step`)
       comes with the project view, which is fetched. `behind` is true from a state that moved
       until a project view read after it has arrived: the forecast in hand belongs to the
       state before, and nothing is said from it meanwhile. */
    this.behind = false;
    this._moves = 0;
  }

  /* ---------- loading ---------- */
  async fetchLists() {
    const pid = this.pid;
    const [project, pages, graph, proposals, requests, experiments] = await Promise.all([
      api.project(pid), api.pages(pid), api.graph(pid), api.proposals(pid, 'proposed'), api.requests(pid), api.experiments(pid),
    ]);
    return { project, pages: pages.pages, edges: graph.edges, dangling: graph.dangling || [], proposals: proposals.proposals, requests: requests.requests, experiments: experiments.experiments };
  }

  async load() {
    try {
      const [lists, activity] = await Promise.all([this.fetchLists(), api.activity(this.pid, { limit: 200 })]);
      if (this.closed) return;
      this.server = lists;
      this.activity = activity.events || [];
      this.moreActivity = !!activity.has_more;
      this.lastSeq = this.activity.reduce((max, ev) => Math.max(max, ev.seq || 0), 0);
      this.loaded = true;
      this.loadError = null;
      this.derive();
      this.connect();
      this.lookAtWorker();
    } catch (error) {
      if (this.closed) return;
      this.loadError = error;
    }
    this.onChange('load');
  }

  /** Fetch the lists again, soon. Calls made while one refetch runs are folded into the next. */
  refresh() {
    if (this.closed || this._timer) return;
    this._timer = setTimeout(() => { this._timer = 0; this.refreshNow(); }, 160);
  }
  async refreshNow() {
    if (this.closed || !this.loaded) return;
    if (this._busy) { this._again = true; return this._waiting || Promise.resolve(); }
    this._busy = true;
    this._waiting = (async () => {
      const moves = this._moves;
      try {
        const lists = await this.fetchLists();
        if (this.closed) return;
        this.server = lists;
        this.refreshError = null;
        if (moves === this._moves) this.behind = false;
        for (const [ref, entry] of this.overlay) if (entry.done) this.overlay.delete(ref);
        this.derive();
        this.lookAtWorker();
      } catch (error) {
        if (!(error instanceof Unreachable)) this.refreshError = error;
        /* the project is gone, or its files can no longer be opened (the server says why) */
        if (error && (error.status === 404 || error.code === 'project_unreadable')) { this.loaded = false; this.loadError = error; }
      } finally {
        this._busy = false;
        this.onChange('refresh', true);
        if (this._again) { this._again = false; this.refresh(); }
      }
    })();
    return this._waiting;
  }

  close() {
    this.closed = true;
    clearTimeout(this._timer);
    if (this.source) { this.source.close(); this.source = null; }
  }

  /* ---------- derived state ---------- */
  derive() {
    const s = this.server;
    if (!s) return;
    this.project = s.project;
    this.pages = s.pages.map((page) => Object.assign({}, page));
    this.byPath = new Map(this.pages.map((page) => [page.path, page]));
    this.refs = new Map();
    const name = (key, path) => { const k = String(key).toLowerCase(); if (!this.refs.has(k)) this.refs.set(k, path); };
    this.pages.forEach((page) => { name(page.path, page.path); name(page.path.replace(/^wiki\//, '').replace(/\.md$/, ''), page.path); });
    this.pages.forEach((page) => { if (page.id) name(page.id, page.path); });
    this.pages.forEach((page) => name(page.path.replace(/^.*\//, '').replace(/\.md$/, ''), page.path));

    let proposals = s.proposals.filter((p) => p.review === 'proposed' && this.byPath.has(p.target));
    for (const [ref, decision] of this.overlay) {
      const proposal = proposals.find((p) => p.id === ref);
      if (proposal) {
        proposals = proposals.filter((p) => p !== proposal);
        const page = this.byPath.get(proposal.target);
        if (page && decision.decision === 'accept') ['state', 'title', 'summary'].forEach((f) => { if (proposal.fields && proposal.fields[f] != null) page[f] = proposal.fields[f]; });
        continue;
      }
      const page = this.byPath.get(ref);
      if (!page || page.review !== 'proposed') continue;
      page.review = decision.decision === 'accept' ? 'accepted' : 'rejected';
      page.optimistic = true;
      const fields = (decision.edits && decision.edits.fields) || {};
      ['state', 'title', 'summary'].forEach((f) => { if (fields[f] != null) page[f] = fields[f]; });
      if (decision.decision === 'accept' && decision.approve_run) page.state = 'approved';
    }
    this.proposals = proposals;

    this.edges = s.edges.filter((e) => this.byPath.has(e.source) && this.byPath.has(e.target)).map((e) => ({ s: e.source, t: e.target, type: e.type }));
    this.adj = new Map();
    const side = (path) => { let a = this.adj.get(path); if (!a) { a = { out: [], inn: [] }; this.adj.set(path, a); } return a; };
    this.edges.forEach((e) => { side(e.s).out.push(e); side(e.t).inn.push(e); });

    this.changeFor = new Map();
    proposals.forEach((p) => { const list = this.changeFor.get(p.target) || []; list.push(p); this.changeFor.set(p.target, list); });

    const items = [];
    this.pages.forEach((page) => { if (page.review === 'proposed') items.push({ ref: page.path, type: 'page', page, kind: page.kind, author: page.author, round: page.round, when: page.updated }); });
    proposals.forEach((p) => { const page = this.byPath.get(p.target); items.push({ ref: p.id, type: 'edit', prop: p, page, kind: page.kind, author: p.author, round: p.round, when: p.created }); });
    const order = (it) => { const i = STAGES.indexOf(it.kind); return (i < 0 ? 50 : i) * 10 + (it.type === 'page' ? 0 : 1); };
    this.pendingItems = items.sort((a, b) => order(a) - order(b) || String(a.when).localeCompare(String(b.when)));

    this.experiments = s.experiments.map((row) => {
      const page = this.byPath.get(row.path);
      if (!page) return row;
      /* an interrupted run is marked on the row of either list: the page's mark follows the one that says so */
      if (row.interrupted && page.state === 'running' && page.interrupted == null) page.interrupted = true;
      return Object.assign({}, row, { review: page.review, state: page.state, title: page.title });
    });
    this.expById = new Map(this.experiments.map((row) => [row.id, row]));
    this.requests = s.requests;
  }

  /** The path of the page a reference names: a path, an id (`H2`), or a link target. */
  resolve(ref) {
    if (!ref) return '';
    return this.refs.get(String(ref).toLowerCase()) || this.refs.get(String(ref).replace(/^wiki\//, '').replace(/\.md$/, '').toLowerCase()) || '';
  }
  page(ref) { return this.byPath.get(ref) || this.byPath.get(this.resolve(ref)) || null; }
  pending() { return this.pendingItems; }
  pendingFor(path) { return this.changeFor.get(path) || []; }
  kindPages(kind) { return this.pages.filter((p) => p.kind === kind); }

  /** Neighbours of a page, split by the direction of the research flow. A page rests on what
      its own relations point at; verdicts (supports, refutes, answers) count as what came of it. */
  lineage(path) {
    const up = [], down = [];
    const a = this.adj.get(path);
    if (!a) return { up, down };
    const seen = new Set();
    a.out.forEach((e) => {
      if (e.type === 'mentions' && a.out.some((o) => o.t === e.t && o.type !== 'mentions')) return;
      const key = e.type + '>' + e.t;
      if (seen.has(key)) return;
      seen.add(key);
      (VERDICT[e.type] ? down : up).push({ verb: relOf(e.type).out, page: this.byPath.get(e.t), type: e.type });
    });
    a.inn.forEach((e) => {
      if (e.type === 'mentions' && a.inn.some((o) => o.s === e.s && o.type !== 'mentions')) return;
      const key = e.type + '<' + e.s;
      if (seen.has(key)) return;
      seen.add(key);
      down.push({ verb: relOf(e.type).inn, page: this.byPath.get(e.s), type: e.type });
    });
    const stage = (x) => { const i = STAGES.indexOf(x.page.kind); return i < 0 ? 50 : i; };
    return { up: up.sort((x, y) => stage(x) - stage(y)), down: down.sort((x, y) => stage(x) - stage(y)) };
  }

  /* ---------- pages and experiments, fetched when shown ---------- */
  viewKey(path) {
    const page = this.server.pages.find((p) => p.path === path);
    if (!page) return null;
    return page.updated + '|' + page.review + '|' + this.server.proposals.filter((p) => p.target === path && p.review === 'proposed').map((p) => p.id).join(',');
  }
  pageView(path) {
    const key = this.viewKey(path);
    if (key == null) return null;
    let entry = this.views.get(path);
    if (!entry) { entry = { key: null, gen: 0, data: null, loading: false, error: null }; this.views.set(path, entry); }
    if (entry.key !== key && !entry.loading) {
      entry.loading = true;
      const gen = entry.gen;   // what comes back may be older than an invalidation made meanwhile: then it is fetched again
      api.page(this.pid, path).then((data) => { entry.data = data; entry.error = null; }, (error) => { entry.error = error; })
        .finally(() => { entry.loading = false; entry.key = entry.gen === gen ? key : null; if (!this.closed) this.onChange('page', true); });
    }
    return entry;
  }
  invalidate(path) { const entry = this.views.get(path); if (entry) { entry.key = null; entry.gen += 1; } }

  expKey(eid) {
    const row = this.server.experiments.find((r) => r.id === eid);
    if (!row) return null;
    const run = row.last_run || {};
    return [row.updated, row.review, row.state, row.runs, run.status, run.finished, (run.repairs || []).length, row.interrupted ? 'interrupted' : '', row.cancel_does || ''].join('|');
  }
  /** GET .../experiments/{eid} takes up to half a second: only for the experiment that is open. */
  experiment(eid) {
    const key = this.expKey(eid);
    if (key == null) return null;
    let entry = this.exps.get(eid);
    if (!entry) { entry = { key: null, gen: 0, data: null, loading: false, error: null }; this.exps.set(eid, entry); }
    if (entry.key !== key && !entry.loading) {
      entry.loading = true;
      const gen = entry.gen;   // saved code does not change the row's key: a fetch that began before the save must not count as current
      api.experiment(this.pid, eid).then((data) => { entry.data = data; entry.error = null; }, (error) => { entry.error = error; })
        .finally(() => { entry.loading = false; entry.key = entry.gen === gen ? key : null; if (!this.closed) this.onChange('experiment', true); });
    }
    return entry;
  }
  invalidateExperiment(eid) { const entry = this.exps.get(eid); if (entry) { entry.key = null; entry.gen += 1; } }
  /** The number check of a finding rests on other pages: the records of the runs, whether
      their experiment is rejected, the sources. When one of those changed, what was read
      of the findings is read again when it is next shown. */
  invalidateFindings() {
    for (const [path, entry] of this.views) { const page = this.byPath.get(path); if (page && page.kind === 'finding') { entry.key = null; entry.gen += 1; } }
  }

  /** What rests on a rejected experiment (`dependents` of its view) is made of other pages:
      the findings based on its runs, the verdicts they carry. When one of those changed, the
      experiments that are rejected are read again when they are next shown. */
  invalidateRejected() {
    for (const [eid, entry] of this.exps) { const row = this.expById.get(eid); if (row && row.review === 'rejected') { entry.key = null; entry.gen += 1; } }
  }

  /** A page that was deleted is gone at once, here and in every other tab that hears of it:
      it must not be asked for again on the way to the next refetch of the lists. */
  forget(path) {
    if (!path || !this.server || !this.server.pages.some((p) => p.path === path)) return;
    this.server.pages = this.server.pages.filter((p) => p.path !== path);
    this.views.delete(path);
    this.derive();
  }

  /** Forget what could not be loaded, so that it is asked for again when it is next shown. */
  retryFailed() {
    for (const entry of this.views.values()) if (entry.error && !entry.loading) { entry.error = null; entry.key = null; entry.gen += 1; }
    for (const entry of this.exps.values()) if (entry.error && !entry.loading) { entry.error = null; entry.key = null; entry.gen += 1; }
    for (const [key, entry] of this.versions) if (entry.error) this.versions.delete(key);
    if (this.paper && this.paper.error) this.paper = null;
  }

  version(path, stamp) {
    const key = path + '@' + stamp;
    let entry = this.versions.get(key);
    if (!entry) {
      entry = { data: null, loading: true, error: null };
      this.versions.set(key, entry);
      api.version(this.pid, path, stamp).then((data) => { entry.data = data; }, (error) => { entry.error = error; })
        .finally(() => { entry.loading = false; if (!this.closed) this.onChange('version', true); });
    }
    return entry;
  }

  async loadEarlierActivity() {
    if (!this.activity.length || !this.moreActivity) return;
    const got = await api.activity(this.pid, { before: this.activity[0].seq, limit: 150 });
    this.activity = (got.events || []).concat(this.activity);
    this.moreActivity = !!got.has_more;
    this.onChange('activity');
  }

  async loadPaper() {
    try { this.paper = await api.paper(this.pid); } catch (error) { this.paper = { error }; }
    this.onChange('paper', true);
  }

  /** Why a worker ended, in its own words (missing credentials, a crash before the engine wrote anything). */
  lookAtWorker() {
    const p = this.project;
    if (!p) return;
    const exit = p.worker && p.worker.last_exit;
    const state = p.state || {};
    /* (how a write-up ended is said by the paper view, from the project's events: its worker's log is not read for it) */
    const wanted = state.status === 'failed' || (state.status === 'stopped' && state.error) || (exit && !exit.stopped && exit.code !== 0 && exit.kind !== 'write');
    const key = wanted ? [state.status, state.error, exit && exit.at].join('|') : null;
    if (!wanted) { this.workerLog = null; return; }
    if (this.workerLog && this.workerLog.key === key) return;
    this.workerLog = { key, text: null };
    api.workerLog(this.pid).then((log) => { if (this.workerLog && this.workerLog.key === key) { this.workerLog.text = log.text || ''; this.workerLog.problem = log.problem || ''; this.onChange('worker-log', true); } }, () => {});
  }

  /* ---------- the event stream ---------- */
  connect() {
    if (this.closed || this.source || typeof EventSource === 'undefined') return;
    const source = new EventSource(api.eventsUrl(this.pid, this.lastSeq));
    this.source = source;
    source.onopen = () => {
      const again = this.live === 'reconnecting';
      this.live = 'live';
      if (again) this.refreshNow();      // whatever happened meanwhile: fetch it all again
      this.onChange('live', true);
    };
    source.onmessage = (message) => {
      let event;
      try { event = JSON.parse(message.data); } catch (e) { return; }
      if (event && typeof event === 'object') this.handle(event);
    };
    source.addEventListener('state', (message) => {
      let data;
      try { data = JSON.parse(message.data); } catch (e) { return; }
      this.takeState(data);
    });
    source.onerror = () => {
      if (this.closed) return;
      this.live = 'reconnecting';
      askHealth().catch(() => {});       // tells the shell whether the server is there at all
      if (source.readyState === 2) {     // closed for good (refused): open a new one, resuming after what we have
        this.source = null;
        source.close();
        setTimeout(() => this.connect(), 3000);
      }
      this.onChange('live', true);
    };
  }

  takeState(data) {
    if (!this.project || !data || !data.state) return;
    const before = this.project.state || {};
    const moved = before.status !== data.state.status || before.phase !== data.state.phase || before.round !== data.state.round
      || (before.phases || []).length !== (data.state.phases || []).length
      || before.next_phase !== data.state.next_phase
      || ('state_source' in data && (this.project.state_source || null) !== (data.state_source || null))
      || ((this.project.worker || {}).state !== (data.worker || {}).state)
      || ((this.project.worker || {}).verified !== (data.worker || {}).verified);
    this.project.state = data.state;
    this.project.checkpoint = data.state.checkpoint;
    /* where that reading came from, and what is wrong with state.json, arrive with it */
    if ('state_problem' in data) this.project.state_problem = data.state_problem;
    if ('state_source' in data) this.project.state_source = data.state_source;
    if (data.worker) this.project.worker = data.worker;
    if (this.server) this.server.project = this.project;
    if (moved) { this._moves++; this.behind = true; this.refresh(); }
    this.lookAtWorker();
    this.onChange('state', true);
  }

  handle(event) {
    const seq = event.seq || 0;
    if (seq && seq <= this.lastSeq) return;
    if (seq) this.lastSeq = seq;
    const type = String(event.type || '');
    if (type === 'run.log') {
      const data = event.data || {};
      const lines = Array.isArray(data.lines) ? data.lines : [];
      if (data.experiment && lines.length) this.tails[data.experiment] = lines[lines.length - 1];
      this.onChange('run.log', true, event);   // appended to the open log without a render
      return;
    }
    this.activity.push(event);
    if (type === 'agent.activity' && /named the project/i.test(event.summary || '')) this.refresh();
    if (this.activity.length > MAX_ACTIVITY) this.activity.splice(0, this.activity.length - MAX_ACTIVITY);
    if (type === 'page.deleted') this.forget((event.data || {}).path || (event.refs || [])[0]);
    if (STRUCTURAL.test(type)) {
      (event.refs || []).forEach((ref) => this.invalidate(ref));
      const data = event.data || {};
      if (data.path) this.invalidate(data.path);
      if (data.target && typeof data.target === 'string') this.invalidate(data.target);
      const eid = data.experiment || (type === 'experiment.state' ? data.id : null);
      if (eid) this.invalidateExperiment(eid);
      if (type === 'run.finished' || (/^(review\.decided|page\.)/.test(type) && /^(experiment|result|source)$/.test(data.kind || 'experiment'))) this.invalidateFindings();
      if (/^(review\.decided|page\.)/.test(type) && /^(finding|hypothesis|result)$/.test(data.kind || '')) this.invalidateRejected();
      if (this.paper && /^(paper\.|review\.decided|run\.finished|page\.)/.test(type)) this.loadPaper();
      this.refresh();
    }
    this.onChange('event', true, event);
  }

  /* ---------- decisions ---------- */
  /** Record a decision in the overlay; the view shows it at once. */
  decide(entry) {
    this.overlay.set(entry.ref, entry);
    this.derive();
  }
  /** Take back decisions that were not sent yet. Returns how many. */
  retract(entries) {
    let n = 0;
    entries.forEach((entry) => {
      if (entry.sent || entry.cancelled) return;
      entry.cancelled = true;
      if (this.overlay.get(entry.ref) === entry) { this.overlay.delete(entry.ref); n++; }
    });
    if (n) this.derive();
    return n;
  }
  unsent() { return Array.from(this.overlay.values()).filter((entry) => !entry.sent); }

  /** Send decisions. Resolves to `{failed: [{entry, result}]}`; rejects (after rolling
      everything back) when the server could not be reached or refused the whole request. */
  async send(entries, options) {
    entries = entries.filter((entry) => !entry.sent && !entry.cancelled);
    if (!entries.length) return { failed: [], results: [] };
    entries.forEach((entry) => { entry.sent = true; });
    let answer;
    try {
      answer = await api.review(this.pid, entries.map((entry) => ({
        ref: entry.ref, decision: entry.decision, reason: entry.reason || null, edits: entry.edits || null,
        approve_run: entry.approve_run == null ? null : entry.approve_run, base_updated: entry.base_updated || null,
      })), options);
    } catch (error) {
      entries.forEach((entry) => { if (this.overlay.get(entry.ref) === entry) this.overlay.delete(entry.ref); });
      this.derive();
      this.onChange('decision-failed');
      throw error;
    }
    const failed = [];
    (answer.results || []).forEach((result, i) => {
      const entry = entries[i];
      if (!entry) return;
      if (result.ok) entry.done = true;
      else { if (this.overlay.get(entry.ref) === entry) this.overlay.delete(entry.ref); failed.push({ entry, result }); }
      this.invalidate(entry.target || entry.ref);
      if (result.page && result.page.path) this.invalidate(result.page.path);
    });
    if (failed.length) this.derive();
    await this.refreshNow();
    return { failed, results: answer.results || [] };
  }
}
