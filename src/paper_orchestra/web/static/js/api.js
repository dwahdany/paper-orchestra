/* The HTTP API, as the server defines it (paper_orchestra/web/api.py).

   Two kinds of failure are told apart everywhere: the server answered with a refusal
   (`ApiError`, with its status, code and message, and the body for a 409), or it did not
   answer at all (`Unreachable`). The second one flips the connection state, which the
   shell shows as a banner; nothing the person typed is thrown away in either case.

   One refusal changes the connection state as well: 401. The page was let in when it was
   opened, so a 401 afterwards means that the server was restarted and has a new access
   token, which this page's cookie is not (`connection.denied`). The shell says so, and
   nothing is asked again by itself until the person has opened the server's new link. */

import { pathOf, plain } from './util.js';

export class ApiError extends Error {
  constructor(status, code, message, body) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.body = body || {};
  }
}
export class Unreachable extends Error {
  constructor(message) {
    super(message || 'The server is not answering.');
    this.name = 'Unreachable';
  }
}

/** What a person is told when the server answers 401 while the app is open. */
export const RESTARTED = 'This server was restarted; open the new link printed in its terminal.';

const listeners = new Set();
let state = 'ok';            // ok | gone (no answer) | denied (401: the server has a new access token)
export const connection = {
  get ok() { return state === 'ok'; },
  /** The server answers, and refuses this page: it was restarted and has a new access token. */
  get denied() { return state === 'denied'; },
  on(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  set(ok) { connection.to(ok ? 'ok' : 'gone'); },
  /** Listeners are told whether all is well, as before; `denied` says which trouble it is. */
  to(next) { if (next !== state) { state = next; listeners.forEach((fn) => fn(next === 'ok')); } },
};

function refusal(status, data) {
  const error = (data && data.error) || {};
  /* the server's own sentence is for somebody who opens the app without its link; a person
     with the app open is told what happened and what to do */
  if (status === 401) return new ApiError(401, error.code || 'token_required', RESTARTED, data);
  const message = error.message || (status === 413 ? 'The upload is too large.' : `The server answered with status ${status}.`);
  return new ApiError(status, error.code || 'error', plain(message), data);
}

async function request(method, url, body, options = {}) {
  /* Refused once, a read is not sent again: every poll and every refetch would be refused
     for as long as this page stays open. What the person asks to be done is still tried,
     and `health` asks whether the server lets this page in again. */
  if (state === 'denied' && method === 'GET' && !options.probe) throw refusal(401, null);
  let response;
  try {
    response = await fetch(url, {
      method,
      headers: body === undefined ? { Accept: 'application/json' } : { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: options.signal,
      keepalive: !!options.keepalive,
    });
  } catch (error) {
    if (error && error.name === 'AbortError') throw error;
    connection.set(false);
    throw new Unreachable();
  }
  connection.to(response.status === 401 ? 'denied' : 'ok');
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
  if (!response.ok) throw refusal(response.status, data);
  return data;
}

const project = (pid) => '/api/projects/' + encodeURIComponent(pid);
const query = (params) => {
  const parts = Object.entries(params || {}).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(v));
  return parts.length ? '?' + parts.join('&') : '';
};

export const api = {
  health: () => request('GET', '/api/health', undefined, { probe: true }),
  doctor: () => request('GET', '/api/doctor'),
  templates: () => request('GET', '/api/templates'),
  /** What a new project starts with (`settings`), the phases in order, whom a request can be
      for, and the values the settings for model effort and compute accept. */
  defaults: () => request('GET', '/api/defaults'),
  render: (markdown, pid, path, signal) => request('POST', '/api/render', { markdown, project: pid || null, path: path || null }, { signal }),

  projects: () => request('GET', '/api/projects'),
  project: (pid) => request('GET', project(pid)),
  patchProject: (pid, patch) => request('PATCH', project(pid), patch),
  /** `body.phase` names the phase to run; without it the engine decides (what is approved runs first). */
  advance: (pid, body) => request('POST', project(pid) + '/advance', body || {}),
  stop: (pid) => request('POST', project(pid) + '/stop'),
  /** Clear the name of a worker the server cannot verify. Nothing is signalled. */
  forgetWorker: (pid) => request('POST', project(pid) + '/worker/forget'),
  workerLog: (pid) => request('GET', project(pid) + '/worker/log'),
  activity: (pid, params) => request('GET', project(pid) + '/activity' + query(params)),
  eventsUrl: (pid, after) => project(pid) + '/events' + query({ after: after || 0 }),

  pages: (pid, params) => request('GET', project(pid) + '/pages' + query(params)),
  page: (pid, path) => request('GET', project(pid) + '/pages/' + pathOf(path)),
  putPage: (pid, path, body) => request('PUT', project(pid) + '/pages/' + pathOf(path), body),
  deletePage: (pid, path) => request('DELETE', project(pid) + '/pages/' + pathOf(path)),
  createPage: (pid, body) => request('POST', project(pid) + '/pages', body),
  version: (pid, path, stamp) => request('GET', project(pid) + '/pages/' + pathOf(path) + '/versions/' + encodeURIComponent(stamp)),

  review: (pid, items, options) => request('POST', project(pid) + '/review', { items }, options),
  proposals: (pid, review) => request('GET', project(pid) + '/proposals' + query({ review })),
  graph: (pid) => request('GET', project(pid) + '/graph?include_rejected=1'),
  requests: (pid) => request('GET', project(pid) + '/requests'),
  /** `role` is whom the instruction is for: `all`, or one of the roles of `/api/defaults`. */
  addRequest: (pid, text, target, role) => request('POST', project(pid) + '/requests', Object.assign({ text, target: target || null }, role ? { for: role } : {})),
  deleteRequest: (pid, rid) => request('DELETE', project(pid) + '/requests/' + encodeURIComponent(rid)),
  search: (pid, q, signal) => request('GET', project(pid) + '/search' + query({ q, limit: 12 }), undefined, { signal }),

  experiments: (pid) => request('GET', project(pid) + '/experiments'),
  experiment: (pid, eid) => request('GET', project(pid) + '/experiments/' + encodeURIComponent(eid)),
  putCode: (pid, eid, name, text) => request('PUT', project(pid) + '/experiments/' + encodeURIComponent(eid) + '/code/' + pathOf(name), { text }),
  /** Make the code the agent of a run left the experiment's code, as the person's edit. */
  adoptCode: (pid, eid, run) => request('POST', project(pid) + '/experiments/' + encodeURIComponent(eid) + '/runs/' + encodeURIComponent(run) + '/adopt-code'),
  approve: (pid, eid, body) => request('POST', project(pid) + '/experiments/' + encodeURIComponent(eid) + '/approve', body || {}),
  run: (pid, eid) => request('POST', project(pid) + '/experiments/' + encodeURIComponent(eid) + '/run'),
  /** Take back the approval of an experiment that has not started, or cancel its run: the
      server does what `cancel_does` of the experiment says. `expect` is what the person was
      told the press does (`withdraw` or `cancel`): when the experiment has moved on and the
      press would do the other, nothing is done and the answer is a 409 that says so. */
  cancel: (pid, eid, expect) => request('POST', project(pid) + '/experiments/' + encodeURIComponent(eid) + '/cancel', expect ? { expect } : undefined),
  runRecord: (pid, eid, number) => request('GET', project(pid) + '/experiments/' + encodeURIComponent(eid) + '/runs/' + number),
  runLog: (pid, eid, number, offset) => request('GET', project(pid) + '/experiments/' + encodeURIComponent(eid) + '/runs/' + number + '/log' + query({ offset })),
  fileUrl: (pid, path, download) => project(pid) + '/files/' + pathOf(path) + (download ? '?download=1' : ''),

  paper: (pid) => request('GET', project(pid) + '/paper'),
  startPaper: (pid, body) => request('POST', project(pid) + '/paper', body),

  /** Create a project from the composer: a real multipart upload, with progress.
      `onProgress(fraction)` follows the bytes sent; the returned object can `abort()`. */
  createProject(form, onProgress) {
    const xhr = new XMLHttpRequest();
    const promise = new Promise((resolve, reject) => {
      xhr.open('POST', '/api/projects');
      xhr.setRequestHeader('Accept', 'application/json');
      xhr.upload.onprogress = (ev) => { if (ev.lengthComputable && onProgress) onProgress(ev.loaded / ev.total); };
      xhr.onerror = () => { connection.set(false); reject(new Unreachable('The server did not answer, or the connection broke while the files were being sent.')); };
      xhr.onabort = () => reject(Object.assign(new Error('The upload was cancelled.'), { name: 'AbortError' }));
      xhr.onload = () => {
        connection.to(xhr.status === 401 ? 'denied' : 'ok');
        let data = null;
        try { data = xhr.responseText ? JSON.parse(xhr.responseText) : null; } catch (e) { data = null; }
        if (xhr.status >= 200 && xhr.status < 300 && data) resolve(data);
        else reject(refusal(xhr.status, data));
      };
      xhr.send(form);
    });
    promise.abort = () => xhr.abort();
    return promise;
  },
};
