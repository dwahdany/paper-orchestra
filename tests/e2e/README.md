# Browser checks of the web app

`web_check.js` drives the real frontend in a real browser against a real server: every
screen at four widths in both themes (no console error, nothing blocked by the
Content-Security-Policy, no horizontal overflow), and the flows a person goes through:
deciding proposals from the map, the inbox and the page (with the short undo, a reason for a
rejection, a stale decision, a double click that must decide once), editing a page into a
conflict and out of it both ways, adding and deleting a page, approving an experiment and
watching a real local run stream its log and end with metrics and a figure, editing its code
and cancelling a run, a run that succeeds only after the runner repaired the code, leaving
an instruction, starting a phase on a scripted model, watching its activity arrive, stopping
and resuming it, a worker that dies, a phase that fails, a budget stop, the server going
away and coming back, the keyboard keeping its place, the command palette, a project of
150 pages, a new project from the composer with two uploaded files, and titles and text
with markup, scripts, odd Unicode and words longer than any column.

`pytest tests/test_web_e2e.py` does everything below by itself, and skips with the reason
when Node, Playwright or a browser is missing. It takes about five minutes.

## By hand

Node with Playwright (`require('playwright')` must resolve, a Chromium must be installed)
and the package's `ui` extra are needed. Nothing leaves the machine: the phases run on
`scripted_model.py`, which the server is told to use for every model.

```bash
WS=$(mktemp -d)
uv run paper-orchestra research demo --dir "$WS/demo"                # about 6 s: its runs are real
uv run paper-orchestra research demo --dir "$WS/large" --pages 150   # for the `large` check

PYTHONPATH=tests/e2e \
PAPER_ORCHESTRA_SCRIPTED_CLIENT=scripted_model:client \
PAPER_ORCHESTRA_E2E_PAUSE=0.5 \
  uv run paper-orchestra ui --workspace "$WS" --port 8600 --no-browser &

node tests/e2e/web_check.js --base http://127.0.0.1:8600        # exit status 0 = all passed
```

The flows change the project, so every run needs a fresh workspace. Options:

- `--only map,inbox` / `--skip composer`: the checks are `screens`, `map`, `inbox`, `wiki`,
  `experiments`, `agents`, `states`, `keys`, `paper`, `large`, `composer`, `hostile`. They
  are written to run in that order on one fresh demo project; a single one also runs alone.
- `--shots DIR`: photographs of the moving states (a running phase, a live log, a failure)
  as the checks pass them.
- `--scheme dark`: the flows in the dark theme (for the photographs).
- `--slow 2`: doubles every wait, for a slow machine.
- `large` looks at the project of 150 pages when the workspace holds it (the two demo
  projects have the same title; they are told apart by the number of their pages);
  otherwise it says that it was skipped. It also checks that what is being typed in one
  project (an instruction, an edit of a code file) does not turn up in the other.
- `hostile` writes pages, an instruction and a project title full of markup, script tags,
  right-to-left overrides, combining marks and a word of 225 letters, looks at every view
  at two widths, and removes them again.

`web_shots.js --base URL --out DIR` photographs every static screen in light and dark at
1440 px, and the main ones at 820, 1100 and 1900 px, without changing the project.

## The scripted model

`scripted_model.py` answers as each role (intake, scout, theorist, designer, runner,
analyst) through the same tools a model gets, with a pause between steps so that a browser
can watch pages arrive and stop a phase in the middle. It never looks anything up: the
scout writes a topic and a question instead of verifying papers. An instruction that
contains `[scripted-failure]` makes the phase that reads it fail, which is how the checks
get a failed phase to look at. Asked to repair the code of a failed run, the scripted runner
replaces it by a script that works, which is how the checks get a run that succeeded on
repaired code.
