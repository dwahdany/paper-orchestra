/* The vocabulary of the interface: kinds, states, relations, and the marks that stand for
   them. Kind is the silhouette of a glyph, review its outline, state its interior, so that
   no status is told by colour alone. */

import { html, raw } from './util.js';

/** The stages of the map, left to right: the way research flows. */
export const STAGES = ['idea', 'source', 'topic', 'question', 'hypothesis', 'experiment', 'result', 'finding'];
/** Kinds listed in the wiki index after the stages. */
export const EXTRA_KINDS = ['journal', 'lesson'];
export const KIND = {
  idea: { one: 'Idea', many: 'Idea', hint: 'what you asked' },
  source: { one: 'Source', many: 'Sources', hint: 'verified papers' },
  topic: { one: 'Topic', many: 'Topics', hint: 'what is known' },
  question: { one: 'Question', many: 'Questions', hint: 'what is open' },
  hypothesis: { one: 'Hypothesis', many: 'Hypotheses', hint: 'claims to test' },
  experiment: { one: 'Experiment', many: 'Experiments', hint: 'plans and code' },
  result: { one: 'Result', many: 'Results', hint: 'run records' },
  finding: { one: 'Finding', many: 'Findings', hint: 'what it means' },
  journal: { one: 'Journal', many: 'Journal', hint: 'round summaries' },
  lesson: { one: 'Lesson', many: 'Lessons', hint: 'process notes for the agents' },
  bundle: { one: 'Sources', many: 'Sources', hint: 'folded sources' },
};
export const kindName = (kind, many) => (KIND[kind] ? (many ? KIND[kind].many : KIND[kind].one) : kind || 'Page');

/** Tone of a lifecycle state: how the mark inside the glyph is drawn. */
export const TONE = {
  open: 'open', planned: 'open', approved: 'ready', queued: 'ready',
  testing: 'live', running: 'live',
  /* a run whose worker is gone: the page says "running", nothing runs it, and it waits for the person */
  interrupted: 'halt',
  /* approved, and no longer what was approved: nothing runs it until the person has looked again */
  'needs approval again': 'halt',
  supported: 'pos', succeeded: 'pos', answered: 'pos',
  refuted: 'neg', failed: 'neg',
  inconclusive: 'void', dropped: 'void', cancelled: 'void',
};
/** An experiment whose page reads approved (or queued) and that would not run as it stands:
    its code or plan changed since the approval, or that approval was used by a run. The
    server says so wherever it lists the experiment (`will_not_run`: the reason, as the rest
    of a sentence). It waits for the person's approval again, and is shown as that. */
export const lapsed = (page) => !!page && page.kind === 'experiment' && page.review === 'accepted' && (page.state === 'approved' || page.state === 'queued') && typeof page.will_not_run === 'string' && page.will_not_run !== '';
/** The state a mark and a word show for a page: what the page says, except for an experiment
    whose run was interrupted and one whose approval no longer holds. The server marks such
    a page wherever it lists it (`interrupted`, `will_not_run`); its file still says
    "running" or "approved", and the mark must not. */
export const shownState = (page) => (page && page.kind === 'experiment' && page.state === 'running' && page.interrupted ? 'interrupted' : lapsed(page) ? 'needs approval again' : (page && page.state) || '');
/** The run of an experiment that succeeded before the one its state is about: the server
    names it (`last_success`: `{number, result_page}`) where the page reads cancelled or
    failed, so that the state is not taken to mean that there is no result. */
export const earlierSuccess = (page) => { const won = page && page.kind === 'experiment' && (page.state === 'cancelled' || page.state === 'failed') ? page.last_success : null; return won && typeof won === 'object' && won.number != null ? won : null; };
/** "run 1 succeeded", for beside the state; '' where no earlier run did. */
export const successWords = (page) => { const won = earlierSuccess(page); return won ? 'run ' + won.number + ' succeeded' : ''; };
export const STATES = {
  question: ['open', 'answered', 'dropped'],
  hypothesis: ['open', 'testing', 'supported', 'refuted', 'inconclusive', 'dropped'],
};
/** Relation -> family (line weight on the map) and the two ways of reading it. */
export const REL = {
  mentions: { fam: 'read', out: 'mentions', inn: 'mentioned by' },
  cites: { fam: 'read', out: 'cites', inn: 'cited by' },
  about: { fam: 'read', out: 'about', inn: 'asked about by' },
  motivated_by: { fam: 'reason', out: 'motivated by', inn: 'motivates' },
  follows: { fam: 'reason', out: 'follows from', inn: 'led to' },
  answers: { fam: 'reason', out: 'answers', inn: 'answered by' },
  tests: { fam: 'test', out: 'tests', inn: 'tested by' },
  run_of: { fam: 'test', out: 'run of', inn: 'has run' },
  based_on: { fam: 'test', out: 'based on', inn: 'interpreted by' },
  supports: { fam: 'pos', out: 'supports', inn: 'supported by' },
  refutes: { fam: 'neg', out: 'refutes', inn: 'refuted by' },
};
export const relOf = (type) => REL[type] || { fam: 'read', out: String(type || 'links to').replace(/_/g, ' '), inn: 'linked from' };

export const PHASES = ['intake', 'research', 'hypothesize', 'design', 'run', 'analyze'];
export const PHASE_NAME = { intake: 'intake', research: 'research', hypothesize: 'hypotheses', design: 'experiment design', run: 'runs', analyze: 'analysis', write: 'the write-up' };
export const phaseName = (phase) => PHASE_NAME[phase] || phase || '';
/** The phases after which the engine pauses for the person. */
export const CHECKPOINT_AFTER = { research: 1, hypothesize: 1, design: 1, analyze: 1 };

/** What choosing a phase by name does, as the list beside Continue says it. */
export const PHASE_DO = {
  research: ['Search the literature again', 'The scout looks for more prior work. What it finds arrives as proposals.'],
  hypothesize: ['Propose hypotheses', 'The theorist proposes hypotheses from what you accepted.'],
  design: ['Design experiments', 'The designer plans experiments, with their code, for the hypotheses you accepted.'],
  run: ['Run what is approved', 'The experiments whose run you approved, and nothing else.'],
  analyze: ['Analyse the results', 'The analyst reads the result records and proposes findings.'],
};
/** Whom an instruction can be for. The roles come from the server (`request_for` of
    /api/defaults); these are the words for them: `pick` in the list of choices, `name` in a
    sentence, `phase` the phase in which that agent works. */
export const ROLE = {
  all: { pick: 'Everyone', name: 'everyone' },
  scout: { pick: 'The agent that searches the literature (scout)', name: 'the scout', phase: 'research' },
  theorist: { pick: 'The agent that proposes hypotheses (theorist)', name: 'the theorist', phase: 'hypothesize' },
  designer: { pick: 'The agent that designs experiments (designer)', name: 'the designer', phase: 'design' },
  analyst: { pick: 'The agent that analyses the results (analyst)', name: 'the analyst', phase: 'analyze' },
};
export const roleName = (role) => (ROLE[role] ? ROLE[role].name : role ? 'the ' + role : 'everyone');
/** The values two settings accept come from the server (`efforts` and `compute_backends` of
    /api/defaults); these are the words for the ones this interface knows. A value it does
    not know is offered under its own name. */
export const EFFORT_NAME = { xhigh: 'extra high' };
export const effortName = (effort) => EFFORT_NAME[effort] || String(effort || '');
export const COMPUTE_NAME = { local: ['This machine (local CPU)', 'this machine'], modal: ['Modal sandboxes', 'Modal sandboxes'] };
export const computeName = (backend, short) => (COMPUTE_NAME[backend] ? COMPUTE_NAME[backend][short ? 1 : 0] : String(backend || ''));

/** How the project's agents are run (`settings.agents`). The runtimes and the credentials a
    form offers come from the server (`agent_runtimes` and `agent_credentials` of
    /api/defaults); these are the words for the ones this interface knows, each with the one
    sentence that says what it means. A value it does not know is offered under its own name. */
export const RUNTIME = {
  classic: ['The tool loop', 'The agents work through the project\'s own tools over a model API: they read and write the wiki and search the literature, and have no shell.'],
  sdk: ['Claude agents with a shell', 'Each agent is a Claude Agent SDK agent with a shell in a folder of its own and the web; what counts still reaches the wiki as a proposal for you, and an experiment you approve for this machine is carried out by another agent.'],
};
export const runtimeName = (runtime) => (RUNTIME[runtime] ? RUNTIME[runtime][0] : String(runtime || ''));
export const CREDENTIAL = {
  login: ['Your Claude Code login', 'The agents sign in with the Claude Code login of this machine: your Claude account pays, and fast mode works where your account has it.'],
  federation: ['The server\'s federation settings', 'The agents sign in with the server\'s Workload Identity Federation settings: the organization of the federation rule pays, and fast mode is refused where that organization has not enabled it.'],
};
export const credentialName = (credential) => (CREDENTIAL[credential] ? CREDENTIAL[credential][0] : String(credential || ''));
/** What the other settings of the agents mean, one sentence each; the sandbox has one for on and one for off. */
export const AGENT_SAYS = {
  model: 'One model for every agent; the models of the tool loop are not used.',
  fast: 'Asks for fast mode, which answers sooner and costs considerably more per token; each agent says in the activity whether it is on.',
  ultracode: 'An agent may start helper agents and workflows of its own, under the same guardrails; every session then starts with a larger prompt.',
  sandbox: 'An agent\'s shell writes only in the agent\'s own folder and a directory for temporary files that is made for it alone, and neither it nor the file tools read your home directory, the project\'s own files, the shared directory for temporary files or a credential. Where the sandbox cannot start, no agent starts.',
  noSandbox: 'Without the sandbox an agent\'s shell and file tools reach whatever you can reach on this machine: only the refusal of the project\'s command line, the rules about credential files and an environment without credentials remain, and a shell can get around them.',
};
/** What an agent with a shell did (`data.did` of an `agent.activity` event), as the feed
    says it before what the agent chose: "ran `ls -la`", "searched the web for …". */
export const DID = { shell: 'ran', search: 'searched the web for', fetch: 'opened', write: 'wrote', helper: 'started a helper agent:', workflow: 'started a workflow of helper agents' };

export const agentName = (author) => (author === 'human' ? 'you' : author === 'system' ? 'code' : String(author || '').replace(/^agent:/, '') || 'engine');

/* ---------- glyphs ---------- */
const SHAPE = {
  idea: '<circle class="g-s" r="8"/><circle class="g-core" r="3"/>',
  source: '<rect class="g-s" x="-3.6" y="-3.6" width="7.2" height="7.2"/>',
  topic: '<rect class="g-s" x="-8" y="-4.6" width="16" height="9.2" rx="4.6"/>',
  question: '<path class="g-s" d="M0-7.6 7.6 0 0 7.6-7.6 0Z"/>',
  hypothesis: '<circle class="g-s" r="7"/>',
  experiment: '<path class="g-s" d="M-8.2 0-4.1-7h8.2L8.2 0 4.1 7h-8.2Z"/>',
  result: '<rect class="g-s" x="-4.6" y="-4.6" width="9.2" height="9.2" rx="1"/>',
  finding: '<path class="g-s" d="M0-7.6 8 6.2H-8Z"/>',
  journal: '<path class="g-s" d="M-5.5-7h11v14h-11Z"/><path class="g-m" d="M-2.5-3h5M-2.5 0h5M-2.5 3h3"/>',
  lesson: '<path class="g-s" d="M-6.5-6h13v9.5h-6L-3 7V3.5h-3.5Z"/>',
  bundle: '<rect class="g-s g-back" x="-1.2" y="-6" width="7.2" height="7.2"/><rect class="g-s" x="-6" y="-1.2" width="7.2" height="7.2"/>',
};
const HALF = {
  hypothesis: 'M0-7A7 7 0 0 0 0 7Z',
  experiment: 'M0-7h-4.1L-8.2 0-4.1 7H0Z',
  question: 'M0-7.6-7.6 0 0 7.6Z',
};
/** Inner SVG of a glyph, centred on 0,0, about 18 units across. */
export function glyphInner(kind, review, state, options) {
  const tone = TONE[state] || 'none';
  const small = kind === 'result' || kind === 'source';
  const dy = kind === 'finding' ? 1.4 : 0;
  let shape = SHAPE[kind] || SHAPE.topic;
  let mark = '';
  if (review !== 'rejected') {
    if (tone === 'live' && HALF[kind]) mark = `<path class="g-fill" d="${HALF[kind]}"/>`;
    else if (tone === 'live') mark = '<circle class="g-dot" r="1.8"/>';
    else if (tone === 'pos') {
      shape = shape.replace('class="g-s"', 'class="g-s g-solid"');
      if (!small) mark = `<path class="g-tick" d="M-3.3 ${0.2 + dy} -1 ${2.5 + dy} 3.5 ${-2.5 + dy}"/>`;
    } else if (tone === 'neg') {
      const k = small ? 2.1 : 2.7;
      mark = `<path class="g-m" d="M${-k} ${-k + dy} ${k} ${k + dy}M${k} ${-k + dy} ${-k} ${k + dy}"/>`;
    } else if (tone === 'ready') mark = '<circle class="g-dot" r="2"/>';
    else if (tone === 'halt') mark = `<path class="g-m" d="M-1.9 ${-2.7 + dy}v5.4M1.9 ${-2.7 + dy}v5.4"/>`;
    else if (tone === 'void') mark = '<path class="g-m" d="M-3 0h6"/>';
  }
  if (review === 'rejected') mark += '<path class="g-strike" d="M-9.5 8 9.5-8"/>';
  if (options && options.change) mark += '<circle class="g-change" r="11.2"/>';
  const k = SHAPE[kind] ? kind : 'topic';
  const r = review === 'proposed' || review === 'rejected' ? review : 'accepted';
  return raw(`<g class="gl k-${k} r-${r} t-${tone}">${shape}${mark}</g>`);
}
export function glyph(kind, review, state, size) {
  const z = +size || 18;
  return raw(`<svg class="glyph" width="${z}" height="${z}" viewBox="-10 -10 20 20" aria-hidden="true">${glyphInner(kind, review, state)}</svg>`);
}
export const pageGlyph = (page, size) => glyph(page.kind, page.review, shownState(page), size);

/* ---------- icons (16px, 1.5 stroke) ---------- */
const ICONS = {
  check: 'M3.5 8.5l3 3 6-7',
  x: 'M4 4l8 8M12 4l-8 8',
  pencil: 'M10.8 2.7l2.5 2.5L6 12.5l-3.2.7.7-3.2zM9.4 4.1l2.5 2.5',
  clock: 'M8 4.5V8l2.3 1.6M14 8A6 6 0 1 1 2 8a6 6 0 0 1 12 0z',
  search: 'M11 11l3 3M12 7.3A4.7 4.7 0 1 1 2.6 7.3a4.7 4.7 0 0 1 9.4 0z',
  sun: 'M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1M10.8 8a2.8 2.8 0 1 1-5.6 0 2.8 2.8 0 0 1 5.6 0z',
  moon: 'M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8z',
  right: 'M6 3.5L10.5 8 6 12.5',
  down: 'M3.5 6L8 10.5 12.5 6',
  plus: 'M8 3v10M3 8h10',
  minus: 'M3 8h10',
  fit: 'M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10',
  open: 'M9 2.5h4.5V7M13.5 2.5L7.5 8.5M6 4H2.5v9.5H12V10',
  file: 'M4 1.8h5l3 3v9.4H4zM9 1.8v3h3',
  stop: 'M4.5 4.5h7v7h-7z',
  play: 'M5 3.2l7.5 4.8L5 12.8z',
  copy: 'M5.5 5.5v-3h8v8h-3M2.5 5.5h8v8h-8z',
  map: 'M2 4l4-1.5 4 1.5 4-1.5v10L10 14l-4-1.5L2 14zM6 2.5v10M10 4v10',
  undo: 'M5.5 3L2.5 6l3 3M2.5 6h6.7a4.3 4.3 0 0 1 0 8.6H6',
  warn: 'M8 2l6.3 11.5H1.7zM8 6.5v3.2M8 11.6v.2',
  help: 'M6.2 6.2a1.9 1.9 0 1 1 2.7 1.7c-.6.3-.9.8-.9 1.5M8 11.6v.2M14 8A6 6 0 1 1 2 8a6 6 0 0 1 12 0z',
  link: 'M6.8 9.2a2.6 2.6 0 0 0 3.7 0l2-2a2.6 2.6 0 0 0-3.7-3.7l-.6.6M9.2 6.8a2.6 2.6 0 0 0-3.7 0l-2 2a2.6 2.6 0 0 0 3.7 3.7l.6-.6',
  trash: 'M3 4.5h10M6.2 4.5V2.8h3.6v1.7M4.3 4.5l.6 8.7h6.2l.6-8.7M6.7 7v4M9.3 7v4',
  pause: 'M5.5 3.5v9M10.5 3.5v9',
  dot: 'M8 8h.01',
  ring: 'M13.2 8A5.2 5.2 0 1 1 2.8 8a5.2 5.2 0 0 1 10.4 0z',
  upload: 'M8 10.5V3M5 5.8L8 2.8l3 3M3 10.5v3h10v-3',
  diff: 'M4.5 2.5v5M2 5h5M9.5 5H14M2.5 11.5h5M9.5 11.5H14M11.7 9.2v4.6',
  cloud: 'M4.6 12.5h6.9a2.9 2.9 0 0 0 .5-5.76A4.1 4.1 0 0 0 4.1 7.6a2.5 2.5 0 0 0 .5 4.9z',
  flask: 'M6.2 2.5h3.6M6.8 2.5v4L3.3 12.4a.9.9 0 0 0 .8 1.3h7.8a.9.9 0 0 0 .8-1.3L9.2 6.5v-4M5 10h6',
  layers: 'M8 2.5l6 3-6 3-6-3zM2.6 8.5L8 11.2l5.4-2.7M2.6 11L8 13.7l5.4-2.7',
  book: 'M3 3h4.2A1.8 1.8 0 0 1 9 4.8V13a1.5 1.5 0 0 0-1.5-1.5H3zM13 3H9.8A1.8 1.8 0 0 0 8 4.8',
  again: 'M13 8a5 5 0 1 1-1.6-3.7M13 2.6v2.9h-2.9',
};
export const icon = (name, cls) => raw(`<svg class="ic ${cls || ''}" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="${ICONS[name] || ICONS.dot}"/></svg>`);

export const kbd = (...keys) => html`${keys.map((k) => html`<kbd>${k}</kbd>`)}`;
