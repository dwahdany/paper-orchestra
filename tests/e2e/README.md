# Browser checks of the web app

`web_check.js` drives the real frontend in a real browser against a real server: every
screen at four widths in both themes (no console error, nothing blocked by the
Content-Security-Policy, no horizontal overflow), the inbox on a phone that is touched
(three widths, and once held on its side: nothing wider than the screen, the list and then
one proposal, every answer on the screen and a finger wide, the toast on no row, no field
under 16 px), and the flows a person goes through:
deciding proposals from the map, the inbox and the page (with the short undo, a reason for a
rejection, a stale decision, a double click that must decide once), editing a page into a
conflict and out of it both ways, adding and deleting a page, approving an experiment and
watching a real local run stream its log and end with metrics and a figure, editing its code
and cancelling a run, a run that succeeds only after the runner repaired the code, leaving
an instruction, starting a phase on a scripted model, watching its activity arrive, stopping
and resuming it, a worker that dies, a phase that fails, a budget stop, the server going
away and coming back, the keyboard keeping its place, the command palette, a project of
150 pages, a new project from the composer with two uploaded files, titles and text
with markup, scripts, odd Unicode and words longer than any column, and a project whose
agents have a shell (below).

It also checks that the interface follows the rules of the engine and the store, each
against what the server answers:

- an instruction has an addressee (everyone, or one agent, from `GET /api/defaults`), an
  open one lists the phases that were given it so far, and it can be withdrawn;
- more research is the person's choice: the list beside the main button offers "Search
  the literature again" (an `advance` that names the phase), and the analyst's
  recommendation from the checkpoint of the analysis stands next to that choice;
- editing is not deciding: Save keeps a proposal proposed and a rejected page rejected,
  "Save and accept" sends `accept: true`, and the app says which it did;
- the number check is shown in the server's summary sentence with the numbers it counts
  (not found, found only outside the finding's records, not checked), on a finding, on a
  suggested edit to a finding (inbox and page) and in the paper view;
- rejecting an experiment that has run lists, with links, the findings and verdicts the
  server says rest on it (`dependents` of the experiment's view: the same after a reload
  and in a browser that never saw the rejection), and the paper view names the accepted
  findings a paper would then leave out, each with the server's reason;
- a result record that was edited by hand says so;
- a worker the server cannot verify is shown with the server's note, without Stop, and
  can be forgotten; a run whose worker is gone is shown as interrupted, by its words, its
  pill and its mark in every view, and Run starts it again; the banner for a server that
  is gone says what that server said becomes of agents and experiments when it ends
  (`stops_workers_on_exit` of `GET /api/health`);
- the library lists a project that cannot be opened with the reason, and its rows count
  what waits, suggested edits included;
- the composer's numbers are those of `GET /api/defaults`, and it takes a research cutoff
  as a month and a year;
- a project's problems are listed on the overview and counted by a quiet mark in the top
  bar, in the server's words; a research cutoff that cannot be read is corrected there;
- the main button and the strip say what a press does as the server says it (`next_step`
  of the project view: "Run the approved experiment, then analyse it"), the phase that was
  next is offered again afterwards, and in the moment between the state arriving and the
  project view the button promises neither a run nor an analysis that has just been done;
- cancelling is what the server says it is (`cancel_does`): an approved experiment that
  has not started offers "Take the approval back" and is a plan again afterwards, one in
  the queue is cancelled, the press tells the server which of the two the person was told,
  and what happened is said in the server's sentence; a press that another window overtook
  (it started the run) changes nothing and says so;
- a project whose `state.json` is lost stands where its events say, with the engine's
  sentence about it beside the main button, among the problems and on its row in the
  library; with the events gone as well the app says that nothing tells where it stands,
  and the one button asks for the phase to go on with; with a link in the file's place
  the app shows the server's sentence and offers no press at all, and a press sent past
  the app is refused by the server with the same sentence;
- the analyst's recommendation of more research is the checkpoint's field (`recommends`,
  with the journal page): it goes when that page is rejected and returns when it is
  accepted again;
- a suggested edit to a finding is marked in the inbox list by what its number check found;
  the quick accept of a marked row says what the edit brings, the map opens such an edit
  instead of accepting it, and a row the listing could not check opens first as well;
- how the last write-up went, one that was refused and one whose worker was killed in the
  middle included, is the server's account (`last` of `GET .../paper`), the same after a
  reload;
- the composer offers the model efforts and compute backends `GET /api/defaults` lists;
- `[redacted]` and `***` in a log are plain text, not an error;
- a project whose models cannot be used (where everybody is who tries the demonstration
  without credentials) is something to set up, not an accident. The server says beforehand
  where a press stops for want of a model (`next_step.blocked`): the main button promises
  the run of what is approved and no analysis, the engine's one sentence stands beside it
  once, with the way to Setup and the word that the app is started again afterwards, and no
  phase that needs a model is offered. The press runs the approved experiment and pauses
  at a checkpoint after the runs; from then on the main button leads to Setup. A press
  that is made all the same (a button drawn while a model could still be used) ends its
  worker with `last_exit.reason: "needs_model"`: the app says once that nothing was
  started, and the strip, the top bar, the overview and the library go on saying
  checkpoint, also after a reload; no log is offered and nothing is "stopped
  unexpectedly". Such a worker ends within half a second and writes no event, so the page
  does not wait for the event stream to report it: while it holds a worker as busy and
  hears nothing, it asks the project view. Setup says what works without credentials
  (`without_credentials` of `GET /api/doctor`). On a third copy of the demo project, which
  no other check touches, with the scripted model switched off (below);
- an approved experiment whose code changed after the approval will not run
  (`will_not_run` of its rows, `not_running` of the forecast): its page, the list beside
  it, the table and the map show it as needing approval again, the main button does not
  promise its run, and the sentence beside the button says that it will not run and why,
  in the app's words, with a link to the experiment that the keyboard reaches;
- a run the person cancelled is no result: an experiment whose first run succeeded and
  whose second was cancelled says "cancelled; run 1 succeeded" wherever its state is said
  (`last_success` of its rows), the words lead to the record of that run, and the main
  button counts one new result to analyse;
- a suggested edit that was decided can be found again: on the page it was about
  (`decided_proposals` of its view) and in the inbox's list of what was decided, where a
  rejection stands with its reason; a proposal the person edited is still said to be
  proposed when the agent proposed it; the New dialog offers the kinds the server lists
  (`creatable_kinds` of `GET /api/defaults`), a lesson among them; a tooltip shows a key
  as a key, and the one of the main button does not lie over the top bar; the paper view
  stops saying "Writing…" in the frame in which the top bar does;
- a server that was restarted has a new access token and refuses the cookie the browser
  holds: the app says "This server was restarted; open the new link printed in its
  terminal.", in the banner, at the foot of the project and where a page could not be read;
  it does not call the server unreachable, promises no retry and asks the server nothing
  while it is refused; once the new link was opened in the same browser, the page works
  again and the event stream is open again. (The check does not restart the server: it
  takes the browser offline, puts another value into its cookie and brings it back, which
  is what a restart is to a browser.)

- agents with a shell (the `sdk` runtime), on a fourth copy of the demo project which no
  other check touches. The composer offers the runtimes and credentials the server lists
  (`agent_runtimes`, `agent_credentials` and `settings.agents` of `GET /api/defaults`), names
  the server's default as the default, says in one sentence what each choice means (the
  login and federation: who pays and what becomes of fast mode; the sandbox switched off:
  what the agents can then reach), and sends only what the person chose, so that a composer
  left alone leaves the runtime to the server's environment. The overview says how the
  project's agents work and changes it (`PATCH` with `settings.agents`); a model without a
  name is not sent, and a change the server refuses is shown in the server's words with
  the form unchanged. Where the agents cannot start, the engine's sentence stands beside
  the main button with the way to those settings, not to Setup. An approval for this
  machine says that an agent carries the run out; one for Modal says what it always said.
  The feed sets what an agent chose apart from the words around it (a shell command, a web
  search, an address, which is text and never a link, a file, a helper and what the helper
  did, a call that was refused), tells a board tool in the project's sentence, folds what
  one agent did in a row into one line that opens to its steps and grows while it is open,
  and tells a run an agent carried out as one line from the agent with the agent's steps in
  it; a run the runner made reads as before. The experiment shows such a run with its log,
  its metrics and the word that an agent carried it out (`carried_out_by` of the run's
  record), also where it failed. At 320 and 390 px nothing of this is wider than the
  phone, a long command wraps under when and who, and every field is a finger high and
  16 px. No agent can be started on the scripted model, so the check writes what agents
  leave behind (events with `data.did`, a run's record) into the project's files;

And a few things a reload, another window or the keyboard must not break:

- whom an instruction is for is kept with the text that is being typed, across a reload; an
  instruction that was withdrawn in another window leaves the list by itself (the server
  records a `request.withdrawn` event), and a press of Withdraw that comes a moment after
  the other window's says that it was withdrawn already;
- E opens the editor with the keyboard in the text and nothing typed into it;
- the list beside the main button closes when Tab leaves it, and Escape from the question
  of a rejection gives the keyboard back to what asked it;
- with a phase to resume and a run approved meanwhile, the line of the round draws the run,
  its analysis and then the resumed phase;
- the run of a result record that was edited by hand says that the record was edited since.

`pytest tests/test_web_e2e.py` does everything below by itself, and skips with the reason
when Node, Playwright or a browser is missing. It takes about five minutes.

## The access token

The server lets nobody in without its access token, on loopback too (the link it prints at
start carries it). The checks reach it as a person and a script do:

- every browser context opens `/?token=...` once, before anything else: the server sets its
  cookie and sends the browser on to the address without the token;
- the requests the checks make themselves (from Node, not from the browser) send
  `Authorization: Bearer <token>`.

The token comes from the environment variable `PAPER_ORCHESTRA_UI_TOKEN`, the same variable
that fixes the token of a server. `pytest tests/test_web_e2e.py` makes a token, starts its
server with it and hands it to the checks that way; nothing else has to be set. Against a
server somebody else started, `--base` may be the link that server printed
(`--base 'http://127.0.0.1:8600/?token=...'`); the token is then an argument of the `node`
process, which other users of the machine can read while the checks run, so prefer the
variable on a shared machine. Without the token the scripts stop at once and say so.

## By hand

Node with Playwright (`require('playwright')` must resolve, a Chromium must be installed)
and the package's `ui` extra are needed. Nothing leaves the machine: the phases run on
`scripted_model.py`, which the server is told to use for every model.

```bash
WS=$(mktemp -d)
uv run paper-orchestra research demo --dir "$WS/demo"                # about 6 s: its runs are real
uv run paper-orchestra research demo --dir "$WS/large" --pages 150   # for the `large` check
uv run paper-orchestra research demo --dir "$WS/nomodel"             # for the `nomodel` check
uv run paper-orchestra research demo --dir "$WS/shell"               # for the `shell` check

# The server's access token, for the server and for the checks (16 characters or more).
export PAPER_ORCHESTRA_UI_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')
# A file that does not exist yet: while it does, the scripted model is switched off. The
# `nomodel` check writes and removes it; the server and the checks are given the same name.
export PAPER_ORCHESTRA_E2E_NO_MODEL="$WS/no-model"

PYTHONPATH=tests/e2e \
PAPER_ORCHESTRA_SCRIPTED_CLIENT=scripted_model:client \
PAPER_ORCHESTRA_E2E_PAUSE=0.5 \
  uv run paper-orchestra ui --workspace "$WS" --port 8600 --no-browser &

node tests/e2e/web_check.js --base http://127.0.0.1:8600        # exit status 0 = all passed
```

To look at the app yourself meanwhile, open the link the server printed.

The flows change the project, so every run needs a fresh workspace. Options:

- `--only map,inbox` / `--skip composer`: the checks are `screens`, `phone`, `shell`, `nomodel`, `map`,
  `inbox`, `wiki`, `experiments`, `agents`, `states`, `rules`, `keys`, `paper`, `large`,
  `composer`, `hostile`. They are written to run in that order on one fresh demo project; a
  single one also runs alone.
- `nomodel` looks at the project in the folder `nomodel` (a third copy of the demo
  project, told from the other two by the name of its folder) while the scripted model is
  switched off. It says in a `note` line that it was skipped when the workspace has no
  such project, when `PAPER_ORCHESTRA_E2E_NO_MODEL` is not set, or when the server is on
  another machine (the file is the switch). It comes before every check that starts
  agents: the switch is the server's, not one project's.
- `shell` looks at the project in the folder `shell` (a fourth copy of the demo project),
  whose agents it gives a shell and into whose files it writes; it says in a `note` line
  that it was skipped when the workspace has no such project, and which parts it left out
  when the server is on another machine or when such agents can start here. It passes
  with `PAPER_ORCHESTRA_AGENT_RUNTIME=sdk` set for the server too, and then sees the
  composer name such agents as the default; the other checks need that variable unset,
  because the projects they make run on the scripted model, which is no such agent.
- `--shots DIR`: photographs of the moving states (a running phase, a live log, a failure)
  as the checks pass them.
- `--scheme dark`: the flows in the dark theme. The checks are meant to pass in both
  themes: run them once without the option and once with it, each on a fresh workspace.
- `--slow 2`: doubles every wait, for a slow machine.
- `large` also checks the keyboard on the map (Tab reaches the selected page, the arrow
  keys move from there with the map following, Tab leaves a map of 150 pages within a few
  presses) and the question about network access: for a plan that asks for the network
  the approval form offers two answers with neither chosen, an approval pressed before
  the answer sends nothing, and the answer goes out with the approval.
- `large` looks at the project of 150 pages when the workspace holds it (the two demo
  projects have the same title; they are told apart by the number of their pages);
  otherwise it says that it was skipped. It also checks that what is being typed in one
  project (an instruction, an edit of a code file) does not turn up in the other.
- A few states cannot be reached by using the app: a worker on another machine, a run
  whose worker is gone, a `state.json` that is lost (and the events with it), a
  `project.json` with a cutoff nobody can read, a project that cannot be opened. For those
  the checks write, move or remove the project's files the way a person with an editor
  would, and take the workspace's path from `GET /api/health`. With a server on another
  machine these parts are skipped, and the checks say so in a `note` line. What only an
  agent brings about (a journal page that recommends more research, a suggested edit to a
  finding) is not written by hand: the scripted analyst does it when an instruction asks
  for it (below).
- `hostile` writes pages, an instruction and a project title full of markup, script tags,
  right-to-left overrides, combining marks and a word of 225 letters, looks at every view
  at two widths, and removes them again.

`web_shots.js --base URL --out DIR` photographs every static screen in light and dark at
1440 px, and the main ones at 820, 1100 and 1900 px, without changing the project. It gets
the token as `web_check.js` does.

## The scripted model

`scripted_model.py` answers as each role (intake, scout, theorist, designer, runner,
analyst) through the same tools a model gets, with a pause between steps so that a browser
can watch pages arrive and stop a phase in the middle. It never looks anything up: the
scout writes a topic and a question instead of verifying papers. Asked to repair the code
of a failed run, the scripted runner replaces it by a script that works, which is how the
checks get a run that succeeded on repaired code. To the stages of the write-up it has
nothing useful to say, and says so slowly: a write-up fails at its outline after a second
or two, which is how the checks get a failed write-up and, by ending its worker before
that, one that was cut short.

An instruction steers it where a check needs a state that only an agent can bring about.
The marks are words in the text of the instruction:

- `[scripted-failure]` makes the phase that reads it fail, which is how the checks get a
  failed phase to look at;
- `[scripted-more-research]` makes the analyst write `next: research` into the round's
  journal page: the recommendation the checkpoint of the analysis repeats;
- `[scripted-edit]` makes the analyst suggest an edit to the first accepted finding of its
  index, adding a sentence with a number that is on no record (73.25).

The last two concern the analyst alone: the `rules` check leaves them in an instruction
for the analyst, through the box on the overview, and the next analysis is given it.

It can also be switched off, to show a project without a usable model. While the file that
`PAPER_ORCHESTRA_E2E_NO_MODEL` names exists, the scripted client offers no tools and no
memory tool, which no agent can work with: the server's forecast says that a model is
needed (`next_step.blocked`), a worker whose phase needs one starts nothing and ends for
that reason, and the run of an approved experiment still happens. Remove the file and the
models can be used again; the switch takes effect at once. Without the variable nothing
changes.

# Live checks of the sdk runtime

Two scripts start a real Claude Agent SDK agent. Neither is collected by pytest, and both
need the `agents` extra and a credential; each costs a few cents on `claude-haiku-5-5`.

`agent_guardrails.py` shows what the unit tests with a scripted agent cannot: that the
sandbox of a real agent's shell, and the agent's own process, hold. It makes a scratch
project, plants marker files where an agent must not read, tells a real scout agent to try
each thing once, and then looks at what happened (not at what the agent says): no marker in
anything it was answered, no file where none may be, credential places closed,
`paper-orchestra` refused, and the web, its own folder and the board working. With
`--helpers` a sub-agent tries too.

Before that agent starts, on Linux, the script shows without a model call that no agent
starts where `bwrap` and `socat` are missing from the path the agent's process gets: once
through this package's own look for them, and once past that look, where the agent's
process must refuse by itself. It also checks the agent's temporary files (a directory of
its own where `mktemp` works; the shared one and another agent's closed to its shell and
to the Read tool; gone when the agent ended), that `python` in its shell is the worker's,
that uv's credentials are closed, and that WebFetch is refused the loopback under every
spelling while the test server counts no connection at all.

```bash
uv run --extra agents python tests/e2e/agent_guardrails.py --credential federation --model claude-haiku-5-5 --helpers
```

`agent_hello.py` starts one agent, has it say one word, and prints what its process
reports: the model, whether fast mode is on (and why not), and its tools.

```bash
uv run --extra agents python tests/e2e/agent_hello.py --credential login --model claude-fable-5-2 --fast --ultracode
```
