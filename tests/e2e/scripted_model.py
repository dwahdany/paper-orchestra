"""A scripted model for the browser test of the web app.

The server under test is started with ``PAPER_ORCHESTRA_SCRIPTED_CLIENT=scripted_model:client``
(and this directory on ``PYTHONPATH``), so the worker processes it launches get this client
for every model: the whole application runs, the engine and the access layer included, and
nothing leaves the machine.  Each role does what a capable model would do with the tools it
is given, slowly enough (``PAPER_ORCHESTRA_E2E_PAUSE`` seconds between steps) for a browser
to watch the pages arrive and to stop a phase in the middle.

It never calls ``verify_paper`` or the web search: those would reach Semantic Scholar and
the web.  The scout writes a topic and a question from the brief instead.  Intake names a
project the person left untitled (``name_project`` is offered only then), so that a
browser can watch the title change.  To the stages of the write-up it has nothing useful
to say, and says so after two pauses: a write-up fails at its outline, late enough for a
browser to see it under way and to end its worker before it has failed.

An instruction of the person steers it where the browser test needs a state that only an
agent can bring about.  The marks are words in the text of a request (or of the
instruction given with a press of Continue):

* ``[scripted-failure]`` makes the phase that reads it fail;
* ``[scripted-more-research]`` makes the analyst write ``next: research`` into the round's
  journal page: its recommendation of another look at the literature;
* ``[scripted-edit]`` makes the analyst suggest an edit to an accepted finding (the first
  one in its index) that adds a sentence with a number that is on no record.

The last two concern the analyst alone: leave them in a request for the analyst, which
its next phase is given and closes.  Every other role ignores them.
"""

from __future__ import annotations

import json
import os
import re
import time

from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory.wiki import split_front_matter

PAUSE = float(os.environ.get("PAPER_ORCHESTRA_E2E_PAUSE", "0.5"))
FAILURE_MARK = "[scripted-failure]"
#: In an instruction the analyst reads: its journal page recommends more research.
RESEARCH_MARK = "[scripted-more-research]"
#: In an instruction the analyst reads: it suggests an edit to an accepted finding.
EDIT_MARK = "[scripted-edit]"
#: What that edit adds to the finding, and the number in it that no record holds.
EDIT_NUMBER = "73.25"
EDIT_SENTENCE = f"With the warmup the spike is {EDIT_NUMBER} times smaller."
EDIT_SUMMARY = "Say how much smaller the spike is with the warmup"

HYPOTHESIS = """---
title: {title}
summary: {summary}
motivated_by: [{motivated_by}]
---
## Statement

{title}.

## Rationale

Proposed by the scripted model of the browser test; see [[idea]].

## Predictions

The effect is measurable in a run of a few seconds.

## How to test

Run the comparison twice with the same seed.

## Risks

It is a scripted hypothesis.
"""

EXPERIMENT = """---
title: {title}
summary: A scripted comparison that takes a second
tests: [{tests}]
mode: script
entrypoint: main.py
timeout_seconds: 120
est_minutes: 1
est_usd: 0
---
## Objective

Test the prediction of {tests}.

## Method

Count to five and write one metric.

## Metrics and success criteria

`steps` equals 5.

## Resources

CPU, a few seconds.
"""

CODE = """\
import json
import time

for step in range(1, 6):
    print(f"step {step} of 5", flush=True)
    time.sleep(0.4)
with open("metrics.json", "w") as handle:
    json.dump({"steps": 5, "loss": 0.25}, handle)
"""


def _pause() -> None:
    time.sleep(PAUSE)


def _has(request: FakeRequest, slug: str) -> bool:
    return slug in (request.system or "")


def _round(request: FakeRequest) -> str:
    session = request.memory.session_id if request.memory is not None else ""
    found = re.search(r"r(\d+)-", session or "")
    return found.group(1) if found else "1"


def _created_id(result: str) -> str | None:
    found = re.search(r"It is ([QHEF]\d+);", result)
    return found.group(1) if found else None


def _title(request: FakeRequest) -> str:
    """A title for an untitled project: not the idea's first line, which it shows already."""
    idea = re.search(r"<idea>\n(.*?)\n</idea>", request.text or "", re.S)
    words = re.findall(r"[\w'-]+", idea.group(1) if idea else "")[:6]
    return ("Scripted study: " + " ".join(words or ["an untitled idea"]))[:80].rstrip()


def intake(request: FakeRequest) -> str:
    _pause()
    if "name_project" in request.tools:
        request.call_tool("name_project", {"title": _title(request)})
        _pause()
    if not _has(request, "what-the-files-say"):
        request.memory.create(
            "/memories/wiki/topics/what-the-files-say.md",
            "---\ntitle: What the attached files say\nsummary: A first reading of the "
            "attachments\n---\nThe attached files were read by the scripted model of the "
            "browser test. They are described in [[idea]].\n",
        )
    _pause()
    if not _has(request, "what-is-still-open"):
        request.memory.create(
            "/memories/wiki/questions/what-is-still-open.md",
            "---\ntitle: What do the attached files leave open?\nabout: [idea]\n---\n"
            "The files do not say how large the effect is.\n",
        )
    _pause()
    return "Wrote one topic and one question."


def scout(request: FakeRequest) -> str:
    _pause()
    if not _has(request, "what-is-known"):
        request.memory.create(
            "/memories/wiki/topics/what-is-known.md",
            "---\ntitle: What is known\nsummary: A scripted summary; no paper was looked up\n"
            "---\nThe scripted model of the browser test looked nothing up: this page "
            "stands in for a synthesis of the literature. See [[idea]].\n",
        )
    _pause()
    if not _has(request, "does-it-hold-at-small-scale"):
        request.memory.create(
            "/memories/wiki/questions/does-it-hold-at-small-scale.md",
            "---\ntitle: Does it hold at small scale?\nabout: [idea]\n---\n"
            "Nothing on record says so.\n",
        )
    _pause()
    return "Wrote one topic and one question."


def theorist(request: FakeRequest) -> str:
    # What the hypotheses rest on: an accepted question, else a topic.  The person may have
    # rejected every question, and the idea itself is not something a hypothesis can be
    # motivated by, so the last resort is to name nothing.
    question = re.search(r"questions/(Q\d+)-", request.system or "")
    topic = re.search(r"(topics/[\w-]+)\.md", request.system or "")
    motivated_by = question.group(1) if question else topic.group(1) if topic else ""
    number = _round(request)
    for slug, title in (
        (f"scripted-effect-is-there-r{number}", "The scripted effect is there"),
        (f"scripted-effect-grows-r{number}", "The scripted effect grows with the batch size"),
    ):
        _pause()
        if not _has(request, slug):
            request.memory.create(
                f"/memories/wiki/hypotheses/{slug}.md",
                HYPOTHESIS.format(
                    title=title,
                    summary=f"Proposed in round {number} by the scripted theorist",
                    motivated_by=motivated_by,
                ),
            )
    # Thinks it over for a while: long enough for the browser test to stop the phase.
    for _ in range(4):
        _pause()
    return "Proposed two hypotheses."


def designer(request: FakeRequest) -> str:
    if request.history:
        for experiment in re.findall(r"^(E\d+):$", request.text, re.M):
            request.call_tool(
                "write_code", {"experiment": experiment, "name": "main.py", "text": CODE}
            )
        return "Repaired the code."
    wanted = re.findall(r'^- (H\d+) "', request.text, re.M)
    for hypothesis in wanted:
        _pause()
        slug = f"scripted-check-of-{hypothesis.lower()}"
        known = re.search(rf"experiments/(E\d+)-{slug}", request.system or "")
        if known:
            experiment = known.group(1)
        else:
            experiment = _created_id(
                request.memory.create(
                    f"/memories/wiki/experiments/{slug}.md",
                    EXPERIMENT.format(title=f"A scripted check of {hypothesis}", tests=hypothesis),
                )
            )
        if experiment:
            request.call_tool(
                "write_code", {"experiment": experiment, "name": "main.py", "text": CODE}
            )
    _pause()
    return f"Designed {len(wanted)} experiment(s)."


def runner(request: FakeRequest) -> str:
    return json.dumps(
        {
            "explanation": "The scripted runner replaces the script by one that works.",
            "files": [{"name": "main.py", "text": CODE}],
        }
    )


def _asked(request: FakeRequest, mark: str) -> bool:
    """Whether what the role was told (its message, its system prompt) carries ``mark``."""
    return mark in (request.text or "") or mark in (request.system or "")


def _page_text(request: FakeRequest, path: str) -> str:
    """The text of a page as the memory tool shows it, without the line numbers."""
    shown = request.memory.view(path).split("\n")[1:]
    return "\n".join(line.split("\t", 1)[1] if "\t" in line else line for line in shown)


def _suggest_edit(request: FakeRequest) -> str | None:
    """Suggest an edit to the first accepted finding of the index: its text with one
    sentence more, which names a number that is on no record.  Returns the finding's
    id, or ``None`` when the index lists no finding."""
    # As the index lists a page: "- findings/F1-what-it-shows.md [F1, accepted] ...".
    found = re.search(
        r"^- (findings/(F\d+)-[\w-]+\.md) \[F\d+, accepted", request.system or "", re.M
    )
    if not found:
        return None
    _, body = split_front_matter(_page_text(request, f"/memories/wiki/{found.group(1)}"))
    request.call_tool(
        "propose_edit",
        {
            "target": found.group(2),
            "summary": EDIT_SUMMARY,
            "body": body.rstrip() + "\n\n" + EDIT_SENTENCE + "\n",
        },
    )
    return found.group(2)


def analyst(request: FakeRequest) -> str:
    if request.history:
        return "The numbers stay as they are."
    results = re.findall(
        r"^- /memories/(wiki/results/(E\d+)-run\d+)\.md: .*tests (H\d+)", request.text, re.M
    )
    for path, experiment, hypothesis in results:
        _pause()
        name = path.rsplit("/", 1)[1]
        slug = f"what-{name.lower()}-shows"
        if not _has(request, slug):
            request.memory.create(
                f"/memories/wiki/findings/{slug}.md",
                f"---\ntitle: What {experiment} shows\nsummary: The scripted run finished\n"
                f"based_on: [{name}]\nsupports: [{hypothesis}]\n---\n## What was observed\n\n"
                f"The run of {experiment} finished.\n\n## What it means\n\nThe prediction "
                "held.\n\n## Limits\n\nOne scripted run.\n",
            )
    _pause()
    edited = _suggest_edit(request) if _asked(request, EDIT_MARK) else None
    journal = re.search(r"goes to (/memories/wiki/journal/\S+\.md)", request.text)
    if journal:
        # What the engine repeats at the checkpoint of the analysis, when asked for.
        recommends = "next: research\n" if _asked(request, RESEARCH_MARK) else ""
        request.memory.create(
            journal.group(1),
            f"---\ntitle: What this round did\nsummary: {len(results)} result(s) read\n"
            f"{recommends}---\n"
            "The runs were read and findings proposed by the scripted analyst.\n",
        )
    said = "Wrote the findings and the journal."
    return f"{said} Suggested an edit to {edited}." if edited else said


ROLES = {
    "intake": intake,
    "scout": scout,
    "theorist": theorist,
    "designer": designer,
    "runner": runner,
    "analyst": analyst,
}


def respond(request: FakeRequest) -> str:
    """Answer as the role the request's label names (``hypothesize.theorist``)."""
    role = request.label.rsplit(".", 1)[-1]
    if _asked(request, FAILURE_MARK):
        # An instruction the person leaves makes the phase that reads it fail: the
        # browser test looks at a failed phase and retries it once the instruction is gone.
        raise RuntimeError("scripted failure: the instruction asked the phase to fail")
    handler = ROLES.get(role)
    if handler is None:
        # The stages of the write-up come here.  The answer is of no use to them (the
        # write-up fails at its outline), but it comes slowly enough for a browser to see
        # a write-up under way, and to end its worker in the middle of it.
        _pause()
        _pause()
        return "The scripted model of the browser test has nothing to say here."
    return handler(request)


def client() -> FakeLLM:
    """The client ``PAPER_ORCHESTRA_SCRIPTED_CLIENT=scripted_model:client`` names."""
    return FakeLLM(respond)
