"""The demonstration project: complete, consistent, and honest about what really ran.

The project is built once for the module (it runs two small experiments on this
machine, which takes a few seconds).  Tests that change it work on a copy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from paper_orchestra.compute import get_backend
from paper_orchestra.llm.base import Usage
from paper_orchestra.llm.pricing import estimate_cost_usd
from paper_orchestra.research import Project, ProjectError, experiments
from paper_orchestra.research.demo import DEMO_TITLE, ILLUSTRATIVE_MODEL_USD, build_demo_project
from paper_orchestra.research.model import RELATIONS, parse_stamp
from paper_orchestra.utils.common import read_json


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("demo") / "demo-warmup"
    build_demo_project(root)
    return root


@pytest.fixture
def demo(built: Path) -> Project:
    """The demonstration project, for reading."""
    return Project.open(built)


@pytest.fixture
def copy(built: Path, tmp_path: Path) -> Project:
    """A copy of the demonstration project that a test may change."""
    shutil.copytree(built, tmp_path / "copy")
    return Project.open(tmp_path / "copy")


def run_dir(project: Project, experiment: str, number: int) -> Path:
    return project.experiment_dir(experiment) / "runs" / str(number)


# -- the project as a whole ----------------------------------------------------------


def test_the_project_opens_cleanly_and_says_that_it_is_a_demonstration(demo):
    assert demo.problems() == []
    assert demo.title == DEMO_TITLE and demo.title.startswith("Demonstration")
    idea = demo.page("wiki/idea.md")
    assert idea.author == "human" and idea.review == "accepted"
    assert "This is a demonstration project" in idea.body
    assert "No model was called" in idea.body
    # The idea the workspace was designed around is the person's own text.
    assert "it is small batches that need it" in idea.body


def test_a_directory_that_is_in_use_is_refused_and_left_as_it_was(tmp_path):
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file").write_text("x")

    with pytest.raises(ProjectError, match="not an empty directory"):
        build_demo_project(tmp_path / "taken")

    assert sorted(p.name for p in (tmp_path / "taken").iterdir()) == ["file"]


def test_the_wiki_holds_every_kind_of_page_in_the_numbers_the_story_tells(demo):
    counts = demo.counts()
    assert counts["by_kind"] == {
        "idea": 1,
        "question": 6,
        "source": 14,
        "topic": 4,
        "hypothesis": 5,
        "experiment": 4,
        "result": 3,
        "finding": 2,
        "journal": 2,
        "lesson": 1,
    }
    authors = {page.author for page in demo.pages()}
    assert {"human", "system", "agent:intake", "agent:scout", "agent:theorist",
            "agent:designer", "agent:analyst"} <= authors  # fmt: skip
    assert {page.round for page in demo.pages()} == {0, 1, 2}
    for page in demo.pages():
        assert page.summary, page.path
        assert len(page.body.strip()) > 80, page.path


def test_hypotheses_are_in_every_review_and_lifecycle_state(demo):
    stands = {page.id: (page.review, page.state) for page in demo.pages(kind="hypothesis")}
    assert stands == {
        "H1": ("accepted", "supported"),
        "H2": ("accepted", "testing"),
        "H3": ("accepted", "refuted"),
        "H4": ("rejected", "open"),
        "H5": ("proposed", "open"),
    }
    rejected = demo.page("H4")
    assert rejected.decided_by == "human"
    assert "Ma and Yarats (2021) already compare RAdam" in rejected.decision_reason
    for page in demo.pages(kind="hypothesis"):
        for heading in ("Statement", "Rationale", "Predictions", "How to test", "Risks"):
            assert f"## {heading}" in page.body, (page.id, heading)


def test_experiments_are_one_succeeded_one_failed_one_waiting_and_one_approved(demo):
    stands = {page.id: (page.review, page.state) for page in demo.pages(kind="experiment")}
    assert stands == {
        "E1": ("accepted", "succeeded"),
        "E2": ("accepted", "failed"),
        "E3": ("proposed", "planned"),
        "E4": ("accepted", "approved"),
    }
    waiting = experiments.load_plan(demo, "E3")
    assert (waiting.mode, waiting.gpu, waiting.est_minutes, waiting.est_usd) == (
        "gpu",
        "T4",
        20,
        0.2,
    )
    assert experiments.load_plan(demo, "E4").backend == "local"
    for page in demo.pages(kind="experiment"):
        for heading in ("Objective", "Method", "Metrics and success criteria", "Resources"):
            assert f"## {heading}" in page.body, (page.id, heading)
        files = experiments.code_files(demo, str(page.id))
        assert "main.py" in files
        errors, _ = experiments.split_problems(experiments.check_code(demo, str(page.id)))
        assert errors == [], page.id


def test_the_graph_has_no_dangling_edges_and_every_relation_the_views_draw(demo):
    graph = demo.graph(include_rejected=True)
    assert graph["dangling"] == []
    nodes = {node["id"] for node in graph["nodes"]}
    assert len(nodes) == demo.counts()["pages"] == 42
    for edge in graph["edges"]:
        assert edge["source"] in nodes and edge["target"] in nodes
    drawn = {edge["type"] for edge in graph["edges"]}
    assert drawn == {*RELATIONS, "mentions"}
    # Every page but the idea is joined to the rest.
    joined = {edge["source"] for edge in graph["edges"]} | {e["target"] for e in graph["edges"]}
    assert nodes - joined == set()
    # What the person rejected is not drawn unless asked for.
    assert "wiki/hypotheses/" + Path(demo.page("H4").path).name not in {
        node["id"] for node in demo.graph()["nodes"]
    }


def test_every_wiki_link_is_written_on_one_line_and_leads_somewhere(demo):
    for page in demo.pages():
        for line in page.body.splitlines():
            assert line.count("[[") == line.count("]]"), (page.path, line)
    proposed = [proposal.body for proposal in demo.proposals(review=None) if proposal.body]
    for line in "\n".join(proposed).splitlines():
        assert line.count("[[") == line.count("]]"), line


def test_the_attachments_are_the_four_files_of_the_brief(demo):
    names = {entry["name"]: entry["media_type"] for entry in demo.attachments()}
    assert names == {
        "notes.md": "text/markdown",
        "prior_run.csv": "text/csv",
        "loss_curves.png": "image/png",
        "kingma2015.pdf": "application/pdf",
    }
    with Image.open(demo.inputs_dir / "loss_curves.png") as image:
        assert image.format == "PNG" and image.size == (640, 360)
    assert (demo.inputs_dir / "kingma2015.pdf").read_bytes().startswith(b"%PDF")
    assert "0.82" in (demo.inputs_dir / "prior_run.csv").read_text()


# -- what really ran -------------------------------------------------------------------


def test_the_succeeded_run_really_ran_and_its_record_is_what_it_produced(demo, tmp_path):
    (record,) = experiments.runs(demo, "E1")
    directory = run_dir(demo, "E1", 1)
    assert record["status"] == "succeeded" and record["backend"] == "local"
    assert record["seconds"] > 0 and record["error"] is None
    assert record["metrics"] == json.loads((directory / "metrics.json").read_text())
    assert record["artifacts"] == ["artifacts/loss_curve.png"]
    with Image.open(directory / "artifacts" / "loss_curve.png") as image:
        assert image.format == "PNG" and image.width > 400
    log = (directory / "log.txt").read_text()
    assert "batch 16, warmup 0: loss 0.693 at the start" in log
    assert record["code_digest"] == experiments.code_digest(demo, "E1")

    # Run the same code again, by hand: the record holds what that code computes.
    again = tmp_path / "again"
    shutil.copytree(experiments.code_dir(demo, "E1"), again)
    subprocess.run([sys.executable, "main.py"], cwd=again, check=True, capture_output=True)
    assert json.loads((again / "metrics.json").read_text()) == record["metrics"]


def test_the_result_record_shows_the_runs_values_as_written_and_its_figure(demo):
    (record,) = experiments.runs(demo, "E1")
    result = demo.page("results/E1-run1")
    assert (result.review, result.state, result.author) == ("accepted", "succeeded", "system")
    assert result.relations == {"run_of": ["E1"]}
    for name, value in record["metrics"].items():
        assert f"| {name} | {value} |" in result.body
    assert "artifacts/loss_curve.png" in result.body and "![" in result.body


def test_the_failed_experiment_has_a_repair_on_record_and_an_error_to_act_on(demo):
    first, second = experiments.runs(demo, "E2")
    assert (first["status"], second["status"]) == ("failed", "failed")
    assert "NameError" in first["error"] and first["repairs"] == []
    (repair,) = second["repairs"]
    assert "-" in repair["diff"] and "WARMUP_STEPS][-1]" in repair["diff"]
    assert "Cannot take a larger sample than population" in second["error"]
    # The second run got as far as the plan allows, and its log shows it.
    log = (run_dir(demo, "E2", 2) / "log.txt").read_text()
    assert "batch 2048: spike" in log and "batch 4096" not in log
    for name in ("results/E2-run1", "results/E2-run2"):
        assert demo.page(name).state == "failed"
    assert "WARMUP_STEPS][-1]" in experiments.code_files(demo, "E2")["main.py"]
    # The journal says what the person has to decide.
    journal = " ".join(demo.page("journal/round-1").body.split())
    assert "the plan asks for a batch of 4096 drawn without replacement from 4000" in journal


def test_the_approved_experiment_runs_when_the_person_continues(copy):
    backend = get_backend("local")
    try:
        record = experiments.run(copy, "E4", compute=backend)
    finally:
        backend.close()

    assert record["status"] == "succeeded", record["error"]
    assert record["metrics"]["spike_warmup0"] > record["metrics"]["spike_warmup100"]
    assert record["artifacts"] == ["artifacts/warmup_length.png"]
    assert copy.page("E4").state == "succeeded"


# -- the project as the engine finds it ------------------------------------------------


def test_the_model_cost_is_what_the_usage_ledger_adds_up_to(demo):
    """The engine prices the model spend from usage.json; state.json only repeats it."""
    ledger = read_json(demo.root / "usage.json")
    assert ledger["estimated_cost_usd"] == ILLUSTRATIVE_MODEL_USD
    assert ledger["unpriced_models"] == []
    assert {row["label"] for row in ledger["by_label"]} == {
        "intake.intake", "research.scout", "hypothesize.theorist", "design.designer",
        "run.runner", "analyze.analyst",
    }  # fmt: skip
    priced = sum(
        estimate_cost_usd(row["model"], Usage(**{name: row[name] for name in Usage().to_dict()}))
        for row in ledger["by_label"]
    )
    assert round(priced, 4) == ILLUSTRATIVE_MODEL_USD
    for name, count in ledger["total"].items():
        assert count == sum(row[name] for row in ledger["by_label"]), name


def test_the_engine_keeps_what_the_story_spent_when_the_person_continues(copy):
    """Continuing the demonstration is what it is for.  The engine takes the project as
    its own and prices the model spend from the ledger: the story's cost must survive
    that, not drop to zero at the first run."""
    from paper_orchestra.research.engine import Engine

    record = Engine(copy).run_experiment("E4")

    assert record["status"] == "succeeded", record["error"]
    state = read_json(copy.root / "state.json")
    assert state["spent"]["model_usd"] == ILLUSTRATIVE_MODEL_USD
    assert state["spent"]["experiment_seconds"] > record["seconds"]  # the earlier runs count
    assert read_json(copy.root / "usage.json")["estimated_cost_usd"] == ILLUSTRATIVE_MODEL_USD
    assert (state["status"], state["round"], state["next_phase"]) == ("checkpoint", 2, "run")


def test_the_results_of_round_one_are_marked_as_analysed(demo):
    """The engine gives an analysis every result record without the mark it leaves on
    those it has read (``analysed``, the round).  The story's analyst has read the runs
    of E1 and E2: unmarked, they would be analysed again when the person continues."""
    results = demo.pages(kind="result")
    assert [page.path.rsplit("/", 1)[-1] for page in results] == [
        "E1-run1.md",
        "E2-run1.md",
        "E2-run2.md",
    ]
    assert [page.meta.get("analysed") for page in results] == [1, 1, 1]
    # Marking a record is the pipeline's doing and leaves it a record.
    assert {page.author for page in results} == {"system"}


def test_phase_and_checkpoint_events_carry_the_data_the_engine_gives_them(demo):
    """The app is built and tested against this project: its events have to read the
    way the engine's do."""
    events = demo.events()
    finished = [event for event in events if event["type"] == "phase.finished"]
    assert len(finished) == 8
    for event in finished:
        assert set(event["data"]) == {"produced", "seconds"}
        assert event["data"]["seconds"] > 0 and event["agent"] and event["phase"]
        assert event["refs"] == event["data"]["produced"][:50]
    checkpoints = [event for event in events if event["type"] == "checkpoint"]
    assert [(e["round"], e["phase"], e["agent"], e["data"]["next"]) for e in checkpoints] == [
        (1, "research", "scout", "hypothesize"),
        (1, "hypothesize", "theorist", "design"),
        (1, "design", "designer", "run"),
        (1, "analyze", "analyst", "hypothesize"),
        (2, "hypothesize", "theorist", "design"),
        (2, "design", "designer", "run"),
    ]
    last = checkpoints[-1]
    assert set(last["data"]) == {"pending", "edits", "next"}
    assert last["data"]["edits"] == 2
    assert last["data"]["pending"] == {
        "question": 1,
        "hypothesis": 1,
        "experiment": 2,
        "finding": 1,
    }
    assert len(last["refs"]) == 5 and all(demo.resolve(ref) for ref in last["refs"])


# -- findings and the number check -----------------------------------------------------


def test_the_accepted_finding_quotes_only_numbers_of_the_result_record(demo):
    finding = demo.page("F1")
    metrics = experiments.runs(demo, "E1")[0]["metrics"]
    assert finding.review == "accepted"
    assert finding.relations == {
        "answers": ["Q1"],
        "supports": ["H1", "H2"],
        "refutes": ["H3"],
        "based_on": ["results/E1-run1"],
    }
    for name in ("spike_b16_no_warmup", "initial_loss", "final_loss_b16_warmup"):
        assert str(metrics[name]) in finding.body
    checks = experiments.number_check(demo, finding)
    assert checks, "the finding states numbers"
    by_number = {check["number"]: check["found_in"] for check in checks}
    # Every number it states is on record somewhere; what the run measured is in the
    # run's record, and a setting such as the batch size is in what the person wrote.
    assert all(by_number.values()), by_number
    for name in ("spike_b16_no_warmup", "initial_loss", "final_loss_b16_warmup"):
        assert "wiki/results/E1-run1.md" in by_number[str(metrics[name])], name
    assert "256" in by_number and "wiki/results/E1-run1.md" not in by_number["256"]
    # What the finding says is what the run shows.
    assert metrics["spike_b16_no_warmup"] > 0.05
    assert metrics["spike_b16_warmup"] == metrics["spike_b256_no_warmup"] == 0.0
    assert metrics["final_loss_b16_warmup"] >= metrics["final_loss_b16_no_warmup"]


def test_the_finding_that_waits_has_one_number_no_record_supports(demo):
    finding = demo.page("F2")
    assert finding.review == "proposed"
    assert experiments.unverified_numbers(demo, finding) == ["1840"]
    traced = [check for check in experiments.number_check(demo, finding) if check["found_in"]]
    assert len(traced) >= 4
    # It is flagged where the person decides: on the page and at the checkpoint.
    assert finding.meta["unverified"] == ["1840"]
    told = [e["summary"] for e in demo.events() if e["type"] == "checkpoint" and e["round"] == 1]
    assert "F2 states a number that no record supports: 1840." in told[-1]
    assert "unverified" not in demo.page("F1").meta


def test_a_build_that_breaks_off_does_not_read_as_running(tmp_path, monkeypatch):
    """The runs are real, so they can fail on a machine; what was built stays to be read."""

    def no_run(project, ref, **options):
        raise ProjectError("E1 cannot run here.")

    monkeypatch.setattr(experiments, "run", no_run)

    with pytest.raises(ProjectError, match="E1 cannot run here"):
        build_demo_project(tmp_path / "broken")

    state = read_json(tmp_path / "broken" / "state.json")
    assert (state["status"], state["phase"]) == ("failed", "run")
    assert state["error"] == "The demonstration could not be built: E1 cannot run here."
    assert Project.open(tmp_path / "broken").page("E1").state == "approved"


# -- what waits for the person, and what was decided -------------------------------------


def test_a_checkpoint_of_the_second_round_is_waiting_for_the_person(demo):
    state = read_json(demo.root / "state.json")
    assert (state["status"], state["round"], state["phase"], state["next_phase"]) == (
        "checkpoint",
        2,
        "design",
        "run",
    )
    assert state["checkpoint"]["round"] == 2 and state["checkpoint"]["phase"] == "design"
    assert "waiting for you" in state["checkpoint"]["summary"]
    assert state["worker"] is None and state["error"] is None
    phases = [(record["round"], record["phase"]) for record in state["phases"]]
    assert phases == [
        (1, "intake"), (1, "research"), (1, "hypothesize"), (1, "design"), (1, "run"),
        (1, "analyze"), (2, "hypothesize"), (2, "design"),
    ]  # fmt: skip
    for record in state["phases"]:
        assert record["status"] == "completed" and record["seconds"] > 0
        assert all(demo.resolve(path) for path in record["produced"])
    runs = [*experiments.runs(demo, "E1"), *experiments.runs(demo, "E2")]
    assert state["spent"] == {
        "model_usd": ILLUSTRATIVE_MODEL_USD,
        "experiment_seconds": round(sum(run["seconds"] for run in runs), 1),
        "experiment_usd": 0.0,
    }

    pending = demo.counts()["pending"]
    assert pending["by_kind"] == {"question": 1, "hypothesis": 1, "experiment": 1, "finding": 1}
    assert pending["proposals"] == 2


def test_a_pending_edit_changes_one_paragraph_of_a_page_the_person_edited(demo, copy):
    by_target = {proposal.target: proposal for proposal in demo.proposals()}
    topic = demo.page("topics/why-warmup-helps")
    proposal = by_target[topic.path]
    assert proposal.author == "agent:analyst" and proposal.body is not None
    # The person has edited the page, and the edit was proposed against that version.
    assert topic.meta["updated_by"] == "human" and "## My reading" in topic.body
    assert [version.author for version in demo.history(topic.path)][-1] == "agent:scout"
    assert proposal.base_updated == topic.updated
    before, after = topic.body.split("\n\n"), proposal.body.split("\n\n")
    changed = [(old, new) for old, new in zip(before, after, strict=True) if old != new]
    assert len(changed) == 1
    assert "[[F1]] now gives one data point" in changed[0][1]

    # Accepting it applies cleanly and keeps what the person wrote.
    copy.review(proposal.id, "accept")
    accepted = copy.page(topic.path)
    assert accepted.body == proposal.body and "## My reading" in accepted.body


def test_the_analysts_verdict_on_a_hypothesis_waits_and_earlier_ones_were_decided(demo):
    waiting = {proposal.target: proposal for proposal in demo.proposals()}
    verdict = waiting[demo.page("H2").path]
    assert verdict.fields == {"state": "supported"} and verdict.body is None
    decided = {p.summary: p for p in demo.proposals(review=None) if p.review != "proposed"}
    assert {summary: p.review for summary, p in decided.items()} == {
        "Mark H1 as supported by F1": "accepted",
        "Mark H3 as refuted by F1": "accepted",
        "Narrow the idea to problems with sparse gradients": "rejected",
    }
    rejected = decided["Narrow the idea to problems with sparse gradients"]
    assert "The idea stays as I wrote it" in rejected.decision_reason


def test_requests_are_open_and_handled(demo):
    waiting = demo.requests()
    assert [request.text for request in waiting] == [
        "Also look at schedule-free optimisers.",
        "Can E3 finish on a T4 in under 20 minutes? If not, shrink the model first.",
    ]
    assert waiting[1].target == demo.page("E3").path
    handled = [request for request in demo.requests(state=None) if request.state == "done"]
    assert [request.handled_in for request in handled] == [
        {"round": 1, "phase": "design"},
        {"round": 2, "phase": "design"},
        {"round": 2, "phase": "design"},
    ]


def test_the_activity_tells_two_rounds_in_order_and_ends_just_now(demo):
    events = demo.events()
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert len(events) > 120
    types = {event["type"] for event in events}
    assert {
        "project.created", "phase.started", "phase.finished", "checkpoint", "page.created",
        "page.updated", "review.decided", "proposal.created", "request.created",
        "agent.activity", "experiment.state", "run.started", "run.log", "run.finished",
    } <= types  # fmt: skip
    assert {event["round"] for event in events} == {0, 1, 2}
    assert {event["agent"] for event in events} >= {
        "intake", "scout", "theorist", "designer", "runner", "analyst",
    }  # fmt: skip
    for event in events:
        assert event["summary"].strip() and "\n" not in event["summary"]
    moments = [parse_stamp(event["ts"]) for event in events]
    assert moments == sorted(moments)
    now = datetime.now(UTC)
    assert now - timedelta(hours=4) < moments[0] < now - timedelta(hours=2)
    assert now - timedelta(minutes=30) < moments[-1] <= now + timedelta(seconds=2)
    # What an agent did carries the phase it happened in; what the person did does not.
    created = [event for event in events if event["type"] == "page.created" and event["agent"]]
    assert all(event["phase"] for event in created)
    decided = [event for event in events if event["type"] == "review.decided"]
    assert len(decided) > 15 and not any(event["phase"] for event in decided)


# -- a large project for layout tests ----------------------------------------------------


def test_pages_fills_the_wiki_with_generated_pages_for_layout_tests(tmp_path):
    project = build_demo_project(tmp_path / "large", pages=150)

    counts = project.counts()
    assert counts["pages"] == 150
    assert project.problems() == [] and project.graph()["dangling"] == []
    assert counts["by_review"]["proposed"] > 20 and counts["by_review"]["rejected"] > 10
    assert all(counts["by_kind"][kind] > 6 for kind in ("source", "topic", "question",
                                                        "hypothesis", "experiment", "finding"))  # fmt: skip
    generated = [page for page in project.pages() if page.title.startswith("Generated")]
    assert len(generated) == 150 - 42
    assert all("layout tests" in page.body for page in generated)
    # The story itself is untouched.
    assert project.page("F1").review == "accepted"
    assert read_json(project.root / "state.json")["status"] == "checkpoint"
