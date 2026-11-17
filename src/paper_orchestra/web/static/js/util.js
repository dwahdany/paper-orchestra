/* Small helpers: safe HTML building, DOM lookups, times, numbers, storage.

   Everything the views insert into the page is built with the `html` template tag, which
   escapes every interpolated value unless it is itself the result of `html` or `raw`.
   `raw` is used in exactly two kinds of places: fixed markup written in this code base
   (icons, glyphs), and the fields the server rendered (`html`, `marked_html`, `diff_html`).
   Text from the API (wiki content is model-written and copied from the web) never reaches
   `raw`. */

const ENTITIES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = (value) => String(value == null ? '' : value).replace(/[&<>"']/g, (c) => ENTITIES[c]);

class Safe {
  constructor(text) { this.text = text; }
  toString() { return this.text; }
}
const piece = (value) => {
  if (value == null || value === false || value === true) return '';
  if (value instanceof Safe) return value.text;
  if (Array.isArray(value)) return value.map(piece).join('');
  return esc(value);
};
/** Markup that is already safe: written here, or rendered by the server. */
export const raw = (text) => new Safe(String(text == null ? '' : text));
/** Template tag: every interpolated value is escaped unless it is `html` or `raw` output. */
export function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += piece(values[i]) + strings[i + 1];
  return new Safe(out);
}
export const join = (parts, separator = '') => new Safe(parts.map(piece).join(piece(separator)));
/** A sentence of the server that names a command between backticks (`paper-orchestra
    doctor`), with the command set as code instead of the marks. */
export const codeSpans = (text) => join(String(text == null ? '' : text).split('`').map((part, i) => (i % 2 ? html`<code>${part}</code>` : part)));
/** Put built markup into an element. Refuses a plain string, which would not be escaped. */
export function setHTML(element, markup) {
  if (!(markup instanceof Safe)) throw new TypeError('setHTML takes the result of html`...`');
  element.innerHTML = markup.text;
}

export const $ = (selector, root) => (root || document).querySelector(selector);
export const $$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));
export const plural = (n, one, many) => `${n} ${n === 1 ? one : many || one + 's'}`;
/** For attributes that take the words true and false (the template tag prints booleans as nothing). */
export const tf = (flag) => (flag ? 'true' : 'false');
export const cap = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '');
export const listOf = (parts) => (parts.length > 1 ? parts.slice(0, -1).join(', ') + ' and ' + parts[parts.length - 1] : parts[0] || '');

/* ---------- times: the API speaks UTC stamps, the page speaks the person's clock ---------- */
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export const parseStamp = (stamp) => {
  if (!stamp) return null;
  const time = Date.parse(stamp);
  return Number.isNaN(time) ? null : time;
};
const two = (n) => String(n).padStart(2, '0');
export const hm = (stamp) => { const t = parseStamp(stamp); if (t == null) return ''; const d = new Date(t); return two(d.getHours()) + ':' + two(d.getMinutes()); };
export const dayOf = (stamp, now = Date.now()) => {
  const t = parseStamp(stamp);
  if (t == null) return '';
  const d = new Date(t), today = new Date(now);
  const days = Math.round((new Date(today.getFullYear(), today.getMonth(), today.getDate()) - new Date(d.getFullYear(), d.getMonth(), d.getDate())) / 864e5);
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  return d.getDate() + ' ' + MONTHS[d.getMonth()] + (d.getFullYear() === today.getFullYear() ? '' : ' ' + d.getFullYear());
};
export const dateTime = (stamp) => { const day = dayOf(stamp); return day ? (day === 'Today' ? 'today' : day === 'Yesterday' ? 'yesterday' : day) + ', ' + hm(stamp) : ''; };
/** A sentence of the server as the app says things: the person is "you", a time is "2 min ago"
    or "today, 01:12", and what the command line would be told to do is what the app offers
    for it (more research is "Search the literature again", in the menu beside Continue). */
export function plain(text) {
  const stamp = '\\d{4}-\\d\\d-\\d\\dT\\d\\d:\\d\\d(?::\\d\\d(?:\\.\\d+)?)?(?:Z|[+-]\\d\\d:?\\d\\d)';
  return String(text == null ? '' : text)
    .replace(new RegExp('\\bat (' + stamp + ')', 'g'), (all, s) => (parseStamp(s) == null ? all : ago(s)))
    .replace(new RegExp(stamp, 'g'), (s) => (parseStamp(s) == null ? s : dateTime(s)))
    .replace(/\bhuman (changed|edited|wrote|saved|decided|accepted|rejected|updated|deleted|created|approved)\b/g, 'you $1')
    .replace(/\bby (?:the )?human\b(?!-)/g, 'by you')
    .replace(/\bat the person's request\b/g, 'at your request')
    .replace(/\bRaise settings\.limits\.\w+ in the project's settings\b/g, 'Raise the limit')
    .replace(/\b(?:to follow it, )?run (?:it|the research phase next) with `paper-orchestra research advance [^`]*--phase research`, or Continue with research in the app\b/g, (all) => (all.startsWith('to follow it') ? 'to follow it, ' : '') + 'choose “Search the literature again” when you continue')
    .replace(/ \(next: research\)/g, '');
}
/** Why something will not happen, as the server says it of an experiment ("it is not what
    was approved at <stamp> (main.py: changed after the approval (written by agent:designer
    at <stamp>))"): in the app's words for times and for who did it. */
export const whyNot = (reason) => plain(reason).replace(/\bagent:(\w+)/g, 'the $1').replace(/[.\s]+$/, '');
/** What the phases did before a checkpoint, as the page says it beside the live count of
    what waits and the button that says what comes next. The engine keeps that account by
    itself (`did` of the checkpoint); the heading the page already has is left out. A
    checkpoint of an earlier version has only its sentence (`summary`: "The theorist
    proposed 2 hypotheses. 2 hypotheses and 6 findings await your review. Next: designing
    experiments for the hypotheses you accepted."), whose count and whose "Next" were true
    at that moment only: there they are cut off. Takes the checkpoint, or such a sentence. */
export function checkpointDid(checkpoint) {
  const heading = /^Round \d+, [^:.]{1,40}:\s*/;
  if (checkpoint && typeof checkpoint === 'object' && typeof checkpoint.did === 'string') return plain(checkpoint.did.replace(heading, '').trim());
  const summary = checkpoint && typeof checkpoint === 'object' ? checkpoint.summary : checkpoint;
  return plain(String(summary || '')
    .replace(heading, '')
    /* what comes next is said by the button, from the state as it is now */
    .replace(/(^|[.!?]\s+)Next: [^.]*\.\s*$/, '$1')
    /* how many decisions waited, where an agent had asked for one: the page counts them as they are now */
    .replace(/(^|[.!?]\s+)\d+ decisions? waits? for your answer\.\s*$/, '$1')
    .replace(/(^|[.!?]\s+)[^.!?]*\b(?:(?:is|are) waiting for you|awaits? your review)\.\s*$/, '$1')
    .trim());
}
/** "38 min ago", "2 h ago", "3 days ago", then the date. */
export const ago = (stamp, now = Date.now()) => {
  const t = parseStamp(stamp);
  if (t == null) return '';
  const s = Math.max(0, (now - t) / 1000);
  if (s < 50) return 'just now';
  if (s < 3600) return Math.max(1, Math.round(s / 60)) + ' min ago';
  if (s < 86400) return Math.round(s / 3600) + ' h ago';
  if (s < 86400 * 14) { const d = Math.round(s / 86400); return d + (d === 1 ? ' day ago' : ' days ago'); }
  return dayOf(stamp, now);
};
/* ---------- the research cutoff: a year, and a month if the person gives one ---------- */
export const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
/** A research cutoff as the settings hold it ("2024", "2024-06", "2024-06-15") in its parts;
    `null` for none; `{raw}` when it cannot be read (the server keeps such a value as written
    and stops the research until it is corrected). */
export function cutoffParts(value) {
  if (value == null || String(value).trim() === '') return null;
  const m = /^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$/.exec(String(value).trim());
  if (!m || (m[2] && (+m[2] < 1 || +m[2] > 12)) || (m[3] && (+m[3] < 1 || +m[3] > 31))) return { raw: String(value) };
  return { year: m[1], month: m[2] ? +m[2] : 0, day: m[3] ? +m[3] : 0 };
}
/** From a year and a month as chosen: `{value}` ("2024-06", "2024", or `null` for no cutoff), or `{error}`. */
export function cutoffFrom(year, month) {
  const y = String(year == null ? '' : year).trim(), m = +month || 0;
  if (!y) return m ? { error: 'Give the year of the cutoff as well, or set the month back to “any month”.' } : { value: null };
  if (!/^\d{4}$/.test(y) || +y < 1900 || +y > 2200) return { error: 'The year of the cutoff is four digits, such as 2024.' };
  return { value: m ? y + '-' + two(m) : y };
}
/** "June 2024", "2024", "15 June 2024": what the literature has to be published before. */
export const cutoffWords = (parts) => (parts.month ? (parts.day ? parts.day + ' ' : '') + MONTH_NAMES[parts.month - 1] + ' ' + parts.year : String(parts.year));

/** A stopwatch: 1:07, 21:40, 1:02:09. */
export const clock = (seconds) => {
  const s = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return (h ? h + ':' + two(m) : String(m)) + ':' + two(s % 60);
};
/** A length of time in words: 2.1 s, 48 s, 6 m 12 s, 1 h 14 m. */
export const duration = (seconds) => {
  if (seconds == null || Number.isNaN(+seconds)) return '';
  const s = +seconds;
  if (s < 10) return (Math.round(s * 10) / 10) + ' s';
  if (s < 60) return Math.round(s) + ' s';
  if (s < 3600) return Math.floor(s / 60) + ' m ' + two(Math.round(s % 60)) + ' s';
  return Math.floor(s / 3600) + ' h ' + two(Math.round((s % 3600) / 60)) + ' m';
};
export const minutes = (m) => (m == null ? '' : m < 1 ? 'under a minute' : m >= 90 ? (Math.round(m / 6) / 10) + ' h' : Math.round(m) + ' min');
export const money = (usd) => (usd == null || Number.isNaN(+usd) ? '' : '$' + (+usd).toFixed(2));
export const bytes = (n) => (n == null ? '' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : n >= 1e3 ? Math.round(n / 1e3) + ' KB' : n + ' B');

/* ---------- storage that may be unavailable (private windows, policies) ---------- */
export const storage = {
  get(key, fallback = null) {
    try { const text = localStorage.getItem(key); return text == null ? fallback : JSON.parse(text); } catch (e) { return fallback; }
  },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; } },
  remove(key) { try { localStorage.removeItem(key); } catch (e) { /* nothing to remove */ } },
  keys(prefix) { try { return Object.keys(localStorage).filter((k) => k.startsWith(prefix)); } catch (e) { return []; } },
};

export const debounce = (fn, wait) => {
  let timer = 0;
  const run = (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
  run.cancel = () => clearTimeout(timer);
  return run;
};

/** One path of the API from its parts; every part is encoded, a page path keeps its slashes. */
export const pathOf = (path) => String(path).split('/').map(encodeURIComponent).join('/');

/* ---------- a small line diff, for showing what changed under an edit ---------- */
/** Rows `[' ' | '+' | '-', text]` turning `before` into `after`, with unchanged stretches cut
    to `context` lines around each change. Sizes are capped: a page is not a repository. */
export function lineDiff(before, after, context = 2) {
  const a = String(before || '').split('\n'), b = String(after || '').split('\n');
  if (a.length * b.length > 4e6) return [['-', '(the earlier text, ' + a.length + ' lines)'], ['+', '(the current text, ' + b.length + ' lines)']];
  const n = a.length, m = b.length;
  const lcs = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  const rows = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { rows.push([' ', a[i]]); i++; j++; }
    else if (lcs[i + 1][j] >= lcs[i][j + 1]) rows.push(['-', a[i++]]);
    else rows.push(['+', b[j++]]);
  }
  while (i < n) rows.push(['-', a[i++]]);
  while (j < m) rows.push(['+', b[j++]]);
  const keep = new Uint8Array(rows.length);
  rows.forEach((row, k) => { if (row[0] !== ' ') for (let d = -context; d <= context; d++) if (rows[k + d]) keep[k + d] = 1; });
  const out = [];
  let gap = false;
  rows.forEach((row, k) => {
    if (keep[k]) { if (gap && out.length) out.push(['…', '']); gap = false; out.push(row); }
    else gap = true;
  });
  return out;
}
