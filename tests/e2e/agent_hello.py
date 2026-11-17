"""Start one real SDK agent, have it say one word, and print what its process reports.

Not a test pytest collects.  It answers, for the machine it runs on: does an agent
start with this credential, on which model, is fast mode on, and which tools does it
have (with ``--ultracode``: the helper tools)?  One short turn, so a cent or two::

    uv run --extra agents python tests/e2e/agent_hello.py \\
        --credential login --model claude-fable-5-2 --fast --ultracode
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--credential", choices=("login", "federation"), default="login")
    parser.add_argument("--model", default="claude-fable-5-2")
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ultracode", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from paper_orchestra.memory.harness import ContextHarness
    from paper_orchestra.memory.transcript import TranscriptLog
    from paper_orchestra.research.agents import AgentContext
    from paper_orchestra.research.project import Project
    from paper_orchestra.research.sdk.session import AgentRuntime, AgentSession

    workspace = Path(tempfile.mkdtemp(prefix="paper-orchestra-hello-", dir=Path.home()))
    try:
        project = Project.create(
            workspace / "project",
            title="Hello",
            brief="Nothing to research: a check that an agent starts.",
            settings={
                "agents": {
                    "runtime": "sdk",
                    "model": args.model,
                    "credential": args.credential,
                    "fast": args.fast,
                    "ultracode": args.ultracode,
                }
            },
        )
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
            phase="research",
            session="r1-research",
            runtime=runtime,
        )
        session = AgentSession(ctx, "scout", max_turns=2)
        try:
            said = session.send("Reply with the single word ok. Use no tool.")
        finally:
            session.close()
        path = project.root / "transcripts" / "agents" / "r1-research-scout.jsonl"
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        started = next(line for line in lines if line["type"] == "started")
        feed = [e["summary"] for e in project.events() if e["type"] == "agent.activity"]
        print("answer:", said.strip())
        print("model:", started["model"])
        print("fast_mode_state:", started["fast_mode_state"])
        print("fast_mode_disabled_reason:", started["fast_mode_disabled_reason"])
        print("tools:", ", ".join(started.get("tools") or []))
        print("feed:", feed[0])
        rows = harness.usage.to_dict()["by_label"]
        print("spend:", [(row["label"], row["model"], row.get("cost_usd")) for row in rows])
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
