"""Have one real SDK agent raise a decision through the board, and look at the record.

Not a test pytest collects.  The unit tests (``tests/test_research_sdk_decisions.py``)
show with a scripted agent that the board has ``ask_decision`` and what it refuses.
This script shows that a real agent, started through the code the engine uses and
given the message its role is given, finds the tool and uses it: one short session
of the theorist on a scratch project (a few cents on ``claude-haiku-5-5``)::

    uv run --extra agents python tests/e2e/agent_decision.py \\
        --credential federation --model claude-haiku-5-5 [--workspace DIR] [--keep]

The scratch project has an accepted question page whose answer is "the researcher's
decision" and one decision the person has already answered.  Afterwards the script
looks at what is on file, not at what the agent says: a new record under
``decisions/`` raised by the theorist's session, open and unanswered, its event in
the feed, the answered decision in the message the agent was opened with and
untouched on file.  Nothing is answered for the person.

Exit code 0 when every check passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

QUESTION_PAGE = """\
The idea says the model is "re-tested after the discussion". Two readings are possible,
and they are different studies:

1. The re-test happens in the same conversation, with the discussion still in context.
2. The re-test happens in a fresh conversation that is given only a written summary.

No source and no run can settle which one is meant: it is a choice about what is studied.

## What would answer it

The researcher's decision on which re-test channel is the object of study.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--credential", choices=("login", "federation"), default="federation")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--keep", action="store_true", help="keep the scratch project")
    parser.add_argument(
        "--workspace",
        default=None,
        help="where the scratch project is made (default: a new directory in your home)",
    )
    args = parser.parse_args()

    from paper_orchestra.memory.harness import ContextHarness
    from paper_orchestra.memory.transcript import TranscriptLog
    from paper_orchestra.research import agents
    from paper_orchestra.research.agents import AgentContext
    from paper_orchestra.research.project import Project
    from paper_orchestra.research.sdk.session import AgentRuntime

    workspace = Path(
        args.workspace or tempfile.mkdtemp(prefix="paper-orchestra-decision-", dir=Path.home())
    )
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        project = Project.create(
            workspace / "project",
            title="Decision check",
            brief="Does a model change its answer when it is re-tested after the discussion?",
            settings={
                "agents": {
                    "runtime": "sdk",
                    "model": args.model,
                    "credential": args.credential,
                    "fast": False,
                    "ultracode": False,
                },
                # A session of a few cents: the budget ends it long before it could cost more.
                "limits": {"model_usd": 2.0, "max_hypotheses": 1},
            },
        )
        page = project.create_page(
            "question", "Which re-test channel is meant?", QUESTION_PAGE, author="human"
        )
        earlier = project.ask_decision(
            "Is the study about one model or about several?",
            options=[{"label": "One model"}, {"label": "Several models"}],
            default="One model.",
            role="scout",
        )
        project.answer_decision(earlier.id, option=1, text="One is enough for now.")
        earlier_file = project.root / "decisions" / f"{earlier.id}.json"
        earlier_before = earlier_file.read_bytes()

        runtime = AgentRuntime(project)
        need = runtime.need()
        if need is not None:
            print(f"An agent cannot start: {need['why']}: {need['fix']}.")
            return 1
        harness = ContextHarness(
            transcripts=TranscriptLog(project.root / "transcripts"),
            client_for=lambda name: None,
            dream_mode="off",
        )
        ctx = AgentContext(
            project=project,
            harness=harness,
            round=1,
            phase="hypothesize",
            session="r1-hypothesize",
            runtime=runtime,
            instruction=(
                "This is a short check. Read the question page about the re-test channel, "
                "raise what it leaves to me as a decision, propose one hypothesis under "
                "your default, and finish."
            ),
        )
        try:
            report = agents.run_theorist(ctx)
        finally:
            for session in ctx.sessions:
                session.close()

        transcript = project.root / "transcripts" / "agents" / "r1-hypothesize-theorist.jsonl"
        lines = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
        started = next((line for line in lines if line["type"] == "started"), {})
        opened = next((line["text"] for line in lines if line["type"] == "prompt"), "")
        calls = [
            line
            for line in lines
            if line["type"] == "tool" and line.get("name") == "mcp__board__ask_decision"
        ]
        raised = [d for d in project.decisions() if d.id != earlier.id]
        events = [e for e in project.events() if e["type"] == "decision.asked"]
        new = raised[0] if raised else None
        checks = [
            (
                "the agent's process was offered mcp__board__ask_decision",
                "mcp__board__ask_decision" in (started.get("tools") or []),
            ),
            (
                "its opening message lists what was answered, with the answer",
                "<decided>" in opened
                and earlier.id in opened
                and "One is enough for now." in opened,
            ),
            ("the agent called the tool on the board", bool(calls)),
            ("a record of a new decision is on file", new is not None),
            (
                "it was raised by the theorist's session, in its phase",
                new is not None
                and new.raised_by.get("role") == "theorist"
                and new.raised_by.get("session") == "r1-hypothesize-theorist"
                and new.raised_by.get("phase") == "hypothesize",
            ),
            (
                "it waits: open, no answer, two to five options and a default",
                new is not None
                and new.state == "open"
                and not new.answers
                and 2 <= len(new.options) <= 5
                and bool(new.default),
            ),
            (
                "its file is in the project's decisions folder",
                new is not None and (project.root / "decisions" / f"{new.id}.json").is_file(),
            ),
            (
                "the feed says that the theorist asks",
                any(
                    e["agent"] == "theorist" and new is not None and e["data"]["decision"] == new.id
                    for e in events
                ),
            ),
            (
                "the decision the person had answered is as it was",
                earlier_file.read_bytes() == earlier_before,
            ),
            (
                "no agent answered or dismissed anything",
                # (The one answer on file is the person's, given before the agent started.)
                [e["type"] for e in project.events() if e["type"].startswith("decision.a")].count(
                    "decision.answered"
                )
                == 1
                and not [e for e in project.events() if e["type"] == "decision.dismissed"],
            ),
        ]
        print(f"agent: {started.get('model')}, credential {args.credential}")
        print("report:", report.summary)
        for decision in raised:
            print(
                "decision:",
                json.dumps(
                    {
                        "id": decision.id,
                        "question": decision.question,
                        "options": [option.label for option in decision.options],
                        "recommended": decision.recommended,
                        "default": decision.default,
                        "page": decision.page,
                        "about_the_question_page": decision.page == page.path,
                        "raised_by": dict(decision.raised_by),
                        "state": decision.state,
                    }
                ),
            )
        failed = 0
        for name, passed in checks:
            failed += not passed
            print(f"{'PASS' if passed else 'FAIL'}  {name}")
        usage = harness.usage.to_dict()["by_label"]
        print("spend:", [(row["label"], row["model"], row.get("cost_usd")) for row in usage])
        if failed:
            args.keep = True
            print(f"kept: {workspace} (transcript: {transcript})")
        return 1 if failed else 0
    finally:
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
