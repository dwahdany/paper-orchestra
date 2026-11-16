from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

import pytest

from paper_orchestra.memory.wiki import split_front_matter
from paper_orchestra.research import model
from paper_orchestra.research.model import EditProposal, ProjectSettings
from paper_orchestra.research.project import (
    Conflict,
    NotAllowed,
    NotFound,
    Project,
    ProjectError,
)

BRIEF = "Does learning-rate warmup reduce the variance of early updates?\n\nI suspect it does."
NOON = datetime(2026, 11, 15, 12, 0, 0, tzinfo=UTC)


def frozen_clock() -> datetime:
    """A clock that never moves: stamps must still tell versions apart."""
    return NOON


@pytest.fixture
def project(tmp_path) -> Project:
    return Project.create(tmp_path / "p", title="Warmup and variance", brief=BRIEF)


@pytest.fixture
def still(tmp_path) -> Project:
    return Project.create(
        tmp_path / "still", title="Warmup and variance", brief=BRIEF, clock=frozen_clock
    )


def hypothesis(project: Project, title: str = "Warmup reduces variance", **kwargs):
    kwargs.setdefault("author", "agent:theorist")
    return project.create_page("hypothesis", title, "## Statement\n\nIt does.", **kwargs)


def experiment(project: Project, title: str = "Sweep warmup length", **kwargs):
    kwargs.setdefault("author", "agent:designer")
    return project.create_page("experiment", title, "## Objective\n\nMeasure it.", **kwargs)


def raw(project: Project, path: str) -> str:
    return (project.root / path).read_text(encoding="utf-8")


def event_types(project: Project) -> list[str]:
    return [event["type"] for event in project.events()]


# -- creating and opening -------------------------------------------------------


def test_create_lays_out_the_project(project):
    data = json.loads(raw(project, "project.json"))
    assert data["schema"] == 1 and data["title"] == "Warmup and variance"
    assert data["id"] == project.id and data["id"].startswith("p-") and len(data["id"]) == 14
    assert model.parse_stamp(data["created"]) is not None
    assert data["settings"] == ProjectSettings().to_dict()
    for sub in ("inputs", "wiki", "notes", "transcripts", "proposals", "history", "requests"):
        assert (project.root / sub).is_dir()
    assert raw(project, "inputs/brief.md") == BRIEF
    assert json.loads(raw(project, "inputs/manifest.json")) == {
        "brief": "brief.md",
        "files": [],
        "skipped": [],
    }

    idea = project.page("wiki/idea.md")
    assert (idea.kind, idea.author, idea.review, idea.id) == ("idea", "human", "accepted", None)
    assert idea.title == "Warmup and variance" and idea.body.strip() == BRIEF
    assert idea.round == 0 and idea.created == idea.updated == data["created"]
    # The index agents are given exists from the start and names the idea.
    assert "idea.md" in raw(project, "MEMORY.md")

    (event,) = project.events()
    assert event["seq"] == 1 and event["type"] == "project.created"
    assert event["summary"] == 'Project "Warmup and variance" was created.'
    assert "wiki/idea.md" in event["refs"]


def test_create_copies_files_and_directories_but_not_hidden_files_or_links(tmp_path):
    material = tmp_path / "material"
    (material / "data" / ".cache").mkdir(parents=True)
    (material / "data" / "runs.csv").write_text("seed,loss\n1,0.5\n")
    (material / "data" / ".env").write_text("SECRET=1")
    (material / "data" / ".cache" / "blob").write_text("x")
    (material / "notes.md").write_text("# Notes\n")
    (material / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    outside = tmp_path / "outside.txt"
    outside.write_text("not attached")
    os.symlink(outside, material / "data" / "link.txt")
    os.symlink(outside, material / "linked.txt")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "notes.md").write_text("same name, other file\n")

    project = Project.create(
        tmp_path / "p",
        title="T",
        brief="b",
        files=[
            material / "data",
            material / "notes.md",
            str(material / "figure.png"),
            material / "linked.txt",
            tmp_path / "other" / "notes.md",
        ],
    )
    manifest = json.loads(raw(project, "inputs/manifest.json"))
    files = {entry["name"]: entry for entry in manifest["files"]}
    assert sorted(files) == ["data/runs.csv", "figure.png", "notes-2.md", "notes.md"]
    assert files["data/runs.csv"] == {
        "name": "data/runs.csv",
        "size": 16,
        "media_type": "text/csv",
        "sha256": hashlib.sha256(b"seed,loss\n1,0.5\n").hexdigest(),
    }
    assert files["notes.md"]["media_type"] == "text/markdown"
    assert files["figure.png"]["media_type"] == "image/png"
    assert raw(project, "inputs/notes-2.md") == "same name, other file\n"
    assert not (project.inputs_dir / "data" / ".env").exists()
    assert not (project.inputs_dir / "data" / ".cache").exists()
    assert not (project.inputs_dir / "data" / "link.txt").exists()
    assert not (project.inputs_dir / "linked.txt").exists()
    assert {entry["reason"] for entry in manifest["skipped"]} == {"symbolic link"}
    assert len(manifest["skipped"]) == 2
    assert project.attachments() == manifest["files"]
    assert project.events()[0]["summary"] == 'Project "T" was created with 4 attached files.'


def test_create_refuses_what_it_cannot_do_and_leaves_nothing_behind(tmp_path, project):
    with pytest.raises(ProjectError, match="already holds a research project"):
        Project.create(project.root, title="again")
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file.txt").write_text("x")
    with pytest.raises(ProjectError, match="not an empty directory"):
        Project.create(tmp_path / "busy", title="T")
    with pytest.raises(ProjectError, match="does not exist"):
        Project.create(tmp_path / "new", title="T", files=[tmp_path / "missing.csv"])
    assert not (tmp_path / "new").exists()
    with pytest.raises(ProjectError, match=r"settings.compute"):
        Project.create(tmp_path / "new", title="T", settings={"compute": "mainframe"})
    assert not (tmp_path / "new").exists()
    # A place where no directory can be made is said in words, not as a bare OSError.
    with pytest.raises(ProjectError, match=r"could not be created.*Choose a place"):
        Project.create(tmp_path / "busy" / "file.txt" / "p", title="T")


def test_a_create_that_fails_part_way_can_simply_be_tried_again(tmp_path, monkeypatch):
    from paper_orchestra.research import project as store

    attached = tmp_path / "data.csv"
    attached.write_text("seed,loss\n")
    real_copy = store._copy_with_digest

    def disk_full(source, target):
        real_copy(source, target)
        raise OSError(28, "No space left on device")

    for name, exists in (("made", False), ("empty", True)):
        root = tmp_path / name
        if exists:
            root.mkdir()
        monkeypatch.setattr(store, "_copy_with_digest", disk_full)
        with pytest.raises(ProjectError) as caught:
            Project.create(root, title="T", brief="b", files=[attached])
        # Said in words a person can act on, not as a bare OSError.
        assert str(attached) in str(caught.value) and "No space left" in str(caught.value)
        assert "No project was created" in str(caught.value)
        # Nothing is left: not half a project, not a lock file, not a directory made for it.
        assert root.exists() == exists and (not exists or list(root.iterdir()) == [])
        monkeypatch.setattr(store, "_copy_with_digest", real_copy)
        project = Project.create(root, title="T", brief="b", files=[attached])
        assert [entry["name"] for entry in project.attachments()] == ["data.csv"]
        assert Project.open(root).page("idea").body == "b\n"


def test_two_requests_to_create_the_same_project_do_not_write_into_each_other(tmp_path):
    root = tmp_path / "ws" / "p"
    outcomes: dict[str, Project | BaseException] = {}
    together = threading.Barrier(2)

    def create(name: str) -> None:
        together.wait()
        try:
            outcomes[name] = Project.create(root, title=name, brief=f"{name} idea")
        except BaseException as exc:
            outcomes[name] = exc

    threads = [threading.Thread(target=create, args=(name,)) for name in ("First", "Second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    # Each is built on its own and one is moved into place: the other is told so.
    (winner,) = [name for name, outcome in outcomes.items() if isinstance(outcome, Project)]
    (loser,) = [name for name in outcomes if name != winner]
    assert isinstance(outcomes[loser], ProjectError)
    assert "already holds a research project" in str(outcomes[loser])
    opened = Project.open(root)
    assert opened.title == winner and opened.page("idea").body == f"{winner} idea\n"
    assert opened.history("idea") == [] and loser not in raw(opened, "events.jsonl")
    assert [event["type"] for event in opened.events()] == ["project.created"]
    assert [entry.name for entry in root.parent.iterdir()] == ["p"]


def test_a_folder_that_cannot_be_listed_is_reported_as_left_out(tmp_path, monkeypatch):
    from paper_orchestra.research import project as store

    material = tmp_path / "material"
    (material / "locked").mkdir(parents=True)
    (material / "open.txt").write_text("x")
    real_walk = os.walk

    def walk(top, onerror=None):
        # What os.walk does with a directory it may not list (not reproducible as root).
        onerror(PermissionError(13, "Permission denied", str(material / "locked")))
        return real_walk(top, onerror=onerror)

    monkeypatch.setattr(store.os, "walk", walk)
    project = Project.create(tmp_path / "p", title="T", brief="b", files=[material])
    manifest = json.loads(raw(project, "inputs/manifest.json"))
    assert [entry["name"] for entry in manifest["files"]] == ["material/open.txt"]
    assert manifest["skipped"] == [{"path": "material/locked", "reason": "cannot be read"}]


def test_a_title_taken_from_the_brief_is_provisional_until_someone_names_the_project(tmp_path):
    brief = "Warmup is said to matter most for large batches. I suspect the opposite under Adam."
    project = Project.create(tmp_path / "p", title="", brief=brief)
    assert project.title_is_provisional
    assert Project.open(project.root).title_is_provisional

    # An agent may name a project that has no name of its own, once.
    project.rename("Warmup at small batch sizes under Adam", by="agent:intake")
    reopened = Project.open(project.root)
    assert reopened.title == "Warmup at small batch sizes under Adam"
    assert not reopened.title_is_provisional
    # The idea page showed the provisional title; it follows.  Its text is the person's.
    idea = reopened.page("wiki/idea.md")
    assert idea.title == "Warmup at small batch sizes under Adam" and idea.body.strip() == brief
    assert idea.author == "human"
    with pytest.raises(NotAllowed, match="already has a title"):
        reopened.rename("Something else", by="agent:scout")
    assert Project.open(project.root).title == "Warmup at small batch sizes under Adam"

    # The person may always rename it, and their title is never provisional.
    reopened.rename("  My   own title ", by="human")
    assert Project.open(project.root).title == "My own title"
    for bad in ("", "   ", "x" * 300):
        with pytest.raises(ProjectError):
            reopened.rename(bad, by="human")


def test_a_title_the_person_gave_is_not_an_agents_to_change(tmp_path):
    project = Project.create(tmp_path / "p", title="Mine", brief="An idea.")
    assert not project.title_is_provisional
    with pytest.raises(NotAllowed):
        project.rename("Theirs", by="agent:intake")
    # Renaming through the settings (the web app's form) also makes the title the person's.
    other = Project.create(tmp_path / "q", title="", brief="Another idea entirely.")
    other.title = "Chosen in the app"
    other.save_settings()
    assert not Project.open(other.root).title_is_provisional
    # Saving settings without touching the title leaves a provisional title provisional.
    third = Project.create(tmp_path / "r", title="", brief="A third idea.")
    third.settings.update({"limits": {"model_usd": 3}})
    third.save_settings()
    assert Project.open(third.root).title_is_provisional


def test_a_title_is_taken_from_the_brief_when_none_is_given(tmp_path):
    project = Project.create(tmp_path / "p", title="", brief="\nWhy do seeds matter?\nMore text.")
    assert project.title == "Why do seeds matter?"
    assert Project.create(tmp_path / "q", title=" ").title == "Untitled project"


def test_open_reads_what_create_wrote_and_settings_round_trip(tmp_path):
    created = Project.create(
        tmp_path / "p",
        title="T",
        brief="b",
        settings={"compute": "modal", "limits": {"model_usd": 3}, "review": {"topic": "auto"}},
    )
    opened = Project.open(created.root)
    assert (opened.id, opened.title, opened.created) == (created.id, "T", created.created)
    assert opened.settings == created.settings
    assert opened.settings.compute == "modal" and opened.settings.limits.model_usd == 3
    assert opened.settings.limits.max_sources == 40 and opened.settings.policy("topic") == "auto"
    assert opened.problems() == []

    opened.settings.limits.max_hypotheses = 2
    opened.settings.update({"research_cutoff": "2024-06"})
    opened.title = "A better title"
    opened.save_settings()
    again = Project.open(created.root)
    assert again.title == "A better title" and again.id == created.id
    assert again.settings.limits.max_hypotheses == 2 and again.settings.research_cutoff == "2024-06"
    # The other process sees it once it looks again.
    created.reload()
    assert created.title == "A better title"

    again.settings.compute = "mainframe"
    with pytest.raises(ProjectError, match=r"settings.compute"):
        again.save_settings()
    assert json.loads(raw(again, "project.json"))["settings"]["compute"] == "modal"


def test_the_repository_setting_is_kept_in_the_project_file(tmp_path):
    created = Project.create(
        tmp_path / "p", title="T", brief="b", settings={"repo": {"path": "/home/me/code"}}
    )
    assert created.settings.repo == {"path": "/home/me/code", "ref": None}
    stored = json.loads(raw(created, "project.json"))["settings"]
    assert stored["repo"] == {"path": "/home/me/code", "ref": None}
    opened = Project.open(created.root)
    assert opened.settings.repo.path == "/home/me/code" and opened.settings.repo.ref is None
    opened.settings.update({"repo": {"ref": "v2"}})
    opened.save_settings()
    assert Project.open(created.root).settings.repo == {"path": "/home/me/code", "ref": "v2"}
    opened.settings.update({"repo": None})
    opened.save_settings()
    assert json.loads(raw(created, "project.json"))["settings"]["repo"] is None
    # A project without one, as every project before this setting existed.
    plain = Project.create(tmp_path / "q", title="T")
    assert plain.settings.repo is None
    assert json.loads(raw(plain, "project.json"))["settings"]["repo"] is None


def test_an_older_or_hand_edited_project_file_still_opens(project):
    path = project.root / "project.json"
    path.write_text(
        json.dumps(
            {
                "title": "Old",
                "settings": {"compute": "mainframe", "limits": {"model_usd": 9}},
                "starred": True,
            }
        )
    )
    opened = Project.open(project.root)
    assert opened.title == "Old" and opened.id.startswith("p-")
    assert opened.id == Project.open(project.root).id
    assert opened.settings.compute == "local" and opened.settings.limits.model_usd == 9
    messages = [problem["message"] for problem in opened.problems()]
    assert any("settings.compute" in message for message in messages)
    assert any("no id" in message for message in messages)
    # Saving keeps what this version does not know about.
    opened.save_settings()
    assert json.loads(path.read_text())["starred"] is True


def test_opening_something_that_is_not_a_project_says_what_to_do(tmp_path, project):
    with pytest.raises(ProjectError, match=r"project\.json is missing.*research new"):
        Project.open(tmp_path / "nothing")
    (project.root / "project.json").write_text("{broken")
    with pytest.raises(ProjectError, match="cannot be read as JSON"):
        Project.open(project.root)


def test_the_record_sections_are_read_only_at_the_wiki_layer(project):
    assert project.wiki.root == project.root
    assert project.wiki.is_read_only("wiki/sources/Smith2020Deep.md")
    assert project.wiki.is_read_only("wiki/results/E1-run1.md")
    assert not project.wiki.is_read_only("wiki/hypotheses/H1-x.md")


def test_places(project):
    assert project.inputs_dir == project.root / "inputs"
    assert project.paper_dir == project.root / "paper"
    assert project.experiment_dir("e3") == project.root / "experiments" / "E3"
    page = experiment(project)
    assert project.experiment_dir(page.path) == project.root / "experiments" / "E1"
    with pytest.raises(ProjectError, match="not an experiment id"):
        project.experiment_dir("../../etc")
    with pytest.raises(ProjectError, match="not an experiment id"):
        project.experiment_dir("H1")


# -- creating pages -------------------------------------------------------------


def test_an_agents_page_is_a_proposal_with_an_id_and_a_first_state(project):
    project.create_page("topic", "Background", "What is known.", author="human")
    page = hypothesis(
        project,
        summary="Warmup lowers early variance",
        relations={"motivated_by": ["topics/background"]},
        round=2,
        fields={"confidence": 0.6, "state": "supported", "review": "accepted", "id": "H99"},
    )
    assert page.path == "wiki/hypotheses/H1-warmup-reduces-variance.md"
    assert (page.id, page.kind, page.review, page.state) == ("H1", "hypothesis", "proposed", "open")
    assert (page.author, page.round) == ("agent:theorist", 2)
    assert page.relations == {"motivated_by": ["topics/background"]}
    assert page.meta["confidence"] == 0.6
    assert page.created == page.updated and model.parse_stamp(page.created)
    assert project.page("H1") == page

    event = project.events()[-1]
    assert event["type"] == "page.created" and event["agent"] == "theorist"
    assert event["summary"] == 'The theorist proposed hypothesis H1 "Warmup reduces variance".'
    assert event["refs"] == [page.path] and event["round"] == 2
    assert event["data"]["review"] == "proposed"


def test_who_writes_decides_the_first_review(project):
    assert hypothesis(project, author="human").review == "accepted"
    assert project.events()[-1]["summary"].startswith("You added hypothesis H1")
    # Journal pages are accepted on creation by the default policy; findings are not.
    journal = project.create_page("journal", "Round 1", "What happened.", author="agent:analyst")
    assert (journal.review, journal.path, journal.id) == (
        "accepted",
        "wiki/journal/round-1.md",
        None,
    )
    assert project.create_page("finding", "Gap", "x", author="analyst").review == "proposed"
    assert project.create_page("journal", "Round 2", "x", author="system").review == "accepted"
    project.settings.review["finding"] = "auto"
    assert project.create_page("finding", "Gap 2", "x", author="agent:analyst").review == "accepted"


def test_a_person_may_set_the_state_of_their_own_new_page(project):
    assert hypothesis(project, author="human", fields={"state": "testing"}).state == "testing"
    with pytest.raises(ProjectError, match="'done' is not a state"):
        hypothesis(project, author="human", fields={"state": "done"})


def test_pages_that_are_not_created_this_way(project):
    with pytest.raises(ProjectError, match="write_record"):
        project.create_page("source", "A paper", "x", author="system")
    with pytest.raises(NotAllowed, match="idea page is the person's"):
        project.create_page("idea", "Another idea", "x", author="agent:scout")
    with pytest.raises(ProjectError, match="already has its idea page"):
        project.create_page("idea", "Another idea", "x", author="human")
    with pytest.raises(ProjectError, match="'essay' is not a kind"):
        project.create_page("essay", "T", "x", author="human")
    with pytest.raises(ProjectError, match="needs a title"):
        project.create_page("topic", "  ", "x", author="human")
    with pytest.raises(ProjectError, match="'tests' is not a field of a hypothesis"):
        hypothesis(project, relations={"tests": ["H1"]})
    with pytest.raises(ProjectError, match="not an author"):
        hypothesis(project, author="somebody")
    assert project.pages(kind="hypothesis") == []


def test_ids_count_up_and_are_never_used_twice(project):
    first, second = hypothesis(project, "One"), hypothesis(project, "Two")
    assert (first.id, second.id) == ("H1", "H2")
    assert experiment(project).id == "E1"
    project.delete_page("H2", author="human")
    assert hypothesis(project, "Three").id == "H3"
    # A page put there by hand counts too, by its name or by its front matter.
    (project.root / "wiki/hypotheses/H7-by-hand.md").write_text("---\ntitle: By hand\n---\nx\n")
    (project.root / "wiki/hypotheses/loose.md").write_text("---\ntitle: Loose\nid: H11\n---\nx\n")
    assert hypothesis(project, "Four").id == "H12"
    # Working files of an experiment whose page is gone still hold their number.
    (project.root / "experiments" / "E5").mkdir(parents=True)
    assert experiment(project, "Another").id == "E6"


def test_pages_without_numbers_get_distinct_names(project):
    first = project.create_page("topic", "Warmup", "a", author="agent:scout")
    second = project.create_page("topic", "Warmup", "b", author="agent:scout")
    named = project.create_page("topic", "Anything", "c", author="agent:scout", slug="Warm Up!")
    assert [first.path, second.path, named.path] == [
        "wiki/topics/warmup.md",
        "wiki/topics/warmup-2.md",
        "wiki/topics/warm-up.md",
    ]


def test_relation_targets_are_written_the_way_the_wiki_refers_to_them(project):
    first = hypothesis(project, "One")
    topic = project.create_page("topic", "Warmup", "a", author="human")
    exp = experiment(
        project, relations={"tests": [f"[[{first.link}]]", "h1"]}, fields={"cites": "Nobody2020"}
    )
    # Known pages by id, unknown targets as they were given (problems() reports them).
    assert exp.relations == {"tests": ["H1"], "cites": ["Nobody2020"]}
    second = hypothesis(
        project, "Two", relations={"motivated_by": "/memories/wiki/topics/warmup.md"}
    )
    assert second.relations == {"motivated_by": [topic.link]} == {"motivated_by": ["topics/warmup"]}


# -- finding pages --------------------------------------------------------------


def test_a_page_is_found_by_path_id_link_or_name(project):
    page = hypothesis(project)
    for ref in (
        page.path,
        "H1",
        "h1",
        " H1 ",
        "hypotheses/H1-warmup-reduces-variance",
        "[[hypotheses/H1-warmup-reduces-variance|the first one]]",
        "wiki/hypotheses/H1-warmup-reduces-variance.md",
        "/memories/wiki/hypotheses/H1-warmup-reduces-variance.md",
        "H1-warmup-reduces-variance",
        "hypotheses/H1",
        "H1-some-older-name",
    ):
        assert project.resolve(ref) == page.path, ref
    assert project.resolve("idea") == "wiki/idea.md"
    for ref in ("H2", "", "MEMORY", "../project.json", "notes/x", "inputs/brief.md", None):
        assert project.resolve(ref) is None, ref
    with pytest.raises(NotFound, match="No page matches 'H2'"):
        project.page("H2")


def test_a_stale_path_never_becomes_a_page_of_another_section(project):
    exp = experiment(project, author="human")
    topic = project.create_page("topic", "E1 setup notes", "x", author="human")
    assert topic.path == "wiki/topics/e1-setup-notes.md"
    project.delete_page(topic.path, author="human")
    # The path of the deleted topic starts with what looks like the experiment's id.  A
    # second delete from a stale browser tab must not remove the experiment.
    for ref in (topic.path, "topics/e1-setup-notes", "topics/E1", "/memories/wiki/topics/e1.md"):
        assert project.resolve(ref) is None, ref
    with pytest.raises(NotFound):
        project.delete_page(topic.path, author="human")
    assert project.page("E1") == exp
    # Within its own section an older name still finds the page, and a bare name does too.
    assert project.resolve("experiments/E1-an-older-name") == exp.path
    assert project.resolve("Experiments/e1") == exp.path
    assert project.resolve("E1-an-older-name") == exp.path
    # A link to the deleted topic is reported as dangling, not drawn to the experiment.
    project.update_page("idea", author="human", body="See [[topics/e1-setup-notes]].")
    graph = project.graph()
    assert graph["edges"] == []
    assert graph["dangling"] == [
        {"source": "wiki/idea.md", "target": "topics/e1-setup-notes", "type": "mentions"}
    ]


def test_an_id_means_the_numbered_page_even_if_a_source_has_that_key(project):
    page = hypothesis(project)
    project.write_record("source", "H1", "A paper keyed like an id", "Abstract.")
    assert project.resolve("H1") == page.path and project.resolve("h1") == page.path
    assert project.resolve("sources/H1") == "wiki/sources/H1.md"
    assert project.set_state("H1", "testing", by="system").kind == "hypothesis"


def test_pages_handed_out_share_nothing_with_later_readers(project):
    question = project.create_page("question", "Why?", "x", author="human")
    hypothesis(project, relations={"motivated_by": [question.id]}, fields={"tags": ["a"]})
    (page,) = project.pages(kind="hypothesis")
    page.relations["motivated_by"].append("Q9")
    page.relations["cites"] = ["Nobody1900"]
    page.meta["tags"].append("b")
    page.meta["motivated_by"].append("Q9")
    (again,) = project.pages(kind="hypothesis")
    assert again.relations == {"motivated_by": ["Q1"]}
    assert again.meta["tags"] == ["a"] and again.meta["motivated_by"] == ["Q1"]
    assert project.graph()["dangling"] == []
    # An empty filter, as an unset query parameter arrives, is no filter.
    assert len(project.pages(kind="", review="", state="")) == 3


def test_pages_can_be_filtered_and_come_in_reading_order(project):
    for number in range(1, 11):
        hypothesis(project, f"Hypothesis {number}")
    project.review("H2", "accept")
    project.review("H3", "reject", reason="not testable")
    experiment(project)
    labels = [page.label for page in project.pages()]
    assert labels[0] == "Warmup and variance"
    assert labels[1:11] == [f"H{n}" for n in range(1, 11)]  # H2 before H10
    assert labels[11] == "E1"
    assert [page.id for page in project.pages(kind="hypothesis", review="accepted")] == ["H2"]
    assert [
        page.id for page in project.pages(review=["accepted", "rejected"], kind="hypothesis")
    ] == [
        "H2",
        "H3",
    ]
    assert [page.id for page in project.pages(state="planned")] == ["E1"]


# -- changing pages -------------------------------------------------------------


def test_an_agent_revises_its_own_proposal_and_the_old_version_is_kept(project):
    page = hypothesis(project)
    before = raw(project, page.path)
    revised = project.update_page(
        "H1",
        author="agent:theorist",
        body="## Statement\n\nIt does, for small batches.",
        fields={"title": "Warmup reduces variance at small batch size", "confidence": 0.4},
        relations={"cites": ["Smith2020Deep"]},
    )
    assert revised.path == page.path and revised.id == "H1"
    assert revised.relations == {"cites": ["Smith2020Deep"]}
    assert revised.title == "Warmup reduces variance at small batch size"
    assert revised.body == "## Statement\n\nIt does, for small batches.\n"
    assert revised.review == "proposed" and revised.created == page.created
    assert revised.updated > page.updated
    assert revised.meta["updated_by"] == "agent:theorist"

    (version,) = project.history("H1")
    assert (version.stamp, version.author) == (page.updated, "agent:theorist")
    assert version.size == len(before.encode())
    assert version.path.startswith(f"history/{page.path}/")
    assert project.version("H1", version.stamp) == before
    assert project.version("H1", model.compact_stamp(version.stamp)) == before
    assert project.events()[-1]["type"] == "page.updated"
    assert project.events()[-1]["summary"].startswith("The theorist revised hypothesis H1")

    # Saying nothing new writes nothing.
    events = len(project.events())
    assert project.update_page("H1", author="agent:theorist", fields={"confidence": 0.4}) == revised
    assert len(project.history("H1")) == 1 and len(project.events()) == events
    with pytest.raises(NotFound):
        project.version("H1", "2001-01-01T00:00:00Z")


def test_updated_stamps_grow_even_when_the_clock_does_not(still):
    page = hypothesis(still)
    stamps = [page.updated]
    for number in range(5):
        stamps.append(
            still.update_page("H1", author="agent:theorist", body=f"Version {number}").updated
        )
    assert stamps == sorted(set(stamps)) and len(stamps) == 6
    assert stamps[0] == "2026-11-15T12:00:00Z" and stamps[-1] == "2026-11-15T12:00:05Z"
    # History is complete: every earlier version, each under its own stamp.
    history = still.history("H1")
    assert [version.stamp for version in history] == stamps[-2::-1]
    assert "Version 3" in still.version("H1", stamps[4])
    assert "It does." in still.version("H1", stamps[0])
    # A page deleted and created again at the same place does not go back in time either.
    topic = still.create_page("topic", "Warmup", "a", author="human")
    still.update_page(topic.path, author="human", body="b")
    still.delete_page(topic.path, author="human")
    again = still.create_page("topic", "Warmup", "c", author="human")
    assert again.path == topic.path
    assert again.updated > max(version.stamp for version in still.history(topic.path))


def test_what_an_agent_may_not_change_directly(project):
    own = hypothesis(project)
    theirs = hypothesis(project, "Another", author="agent:analyst")
    persons = project.create_page("topic", "Mine", "My notes.", author="human")
    record = project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    accepted = project.review(hypothesis(project, "Accepted").path, "accept")
    rejected = project.review(hypothesis(project, "Rejected").path, "reject", reason="vague")

    cases = [
        (persons.path, "was written by the person; propose an edit with propose_edit"),
        ("wiki/idea.md", "was written by the person; propose an edit with propose_edit"),
        (record.path, "is a record"),
        (theirs.path, "was written by agent:analyst; propose an edit with propose_edit"),
        (accepted.path, "H3 was accepted by the person; propose an edit with propose_edit"),
        (rejected.path, r"H4 was rejected by the person \(vague\)"),
    ]
    for ref, message in cases:
        before = raw(project, project.resolve(ref))
        with pytest.raises(NotAllowed, match=message):
            project.update_page(ref, author="agent:theorist", body="Mine now.")
        with pytest.raises(NotAllowed):
            project.update_page(ref, author="agent:theorist", fields={"summary": "Mine now."})
        with pytest.raises(NotAllowed):
            project.delete_page(ref, author="agent:theorist")
        assert raw(project, project.resolve(ref)) == before
        assert project.history(ref) == [] or ref in (accepted.path, rejected.path)

    # Its own undecided proposal it may change and withdraw.
    project.update_page(own.path, author="agent:theorist", body="Better.")
    project.delete_page(own.path, author="agent:theorist")
    assert project.resolve("H1") is None


def test_an_agent_cannot_rewrite_a_plan_the_person_approved(tmp_path):
    # With experiments accepted on creation there is no review to lock the page: approving
    # the run is the person's decision, and what they approved has to stay what they read.
    project = Project.create(
        tmp_path / "auto", title="T", brief=BRIEF, settings={"review": {"experiment": "auto"}}
    )
    page = experiment(project, fields={"network": False})
    assert (page.review, page.state, page.decided_by) == ("accepted", "planned", None)
    revised = project.update_page("E1", author="agent:designer", body="A better plan.")
    approved = project.set_state("E1", "approved", by="human", base_updated=revised.updated)
    for change in ({"fields": {"network": True}}, {"body": "Something else entirely."}):
        with pytest.raises(NotAllowed, match=r"E1 is approved.*propose_edit"):
            project.update_page("E1", author="agent:designer", **change)
    with pytest.raises(NotAllowed, match="E1 is approved"):
        project.delete_page("E1", author="agent:designer")
    assert project.page("E1") == approved
    # Nor while it runs or once it has run; the pipeline's own bookkeeping goes on.
    for state in ("queued", "running", "failed"):
        project.set_state("E1", state, by="agent:runner")
        with pytest.raises(NotAllowed, match=f"E1 is {state}"):
            project.update_page("E1", author="agent:designer", fields={"network": True})
    assert project.page("E1").meta["network"] is False and project.page("E1").body == revised.body
    # Back to a plan by the person's own hand, it is the designer's to revise again.
    project.set_state("E1", "planned", by="human")
    assert project.update_page("E1", author="agent:designer", body="Third plan.").body == (
        "Third plan.\n"
    )


def test_a_hypothesis_under_test_is_not_rewritten_by_its_author(tmp_path):
    project = Project.create(
        tmp_path / "auto", title="T", brief=BRIEF, settings={"review": {"hypothesis": "auto"}}
    )
    hypothesis(project)
    project.set_state("H1", "testing", by="system")
    with pytest.raises(NotAllowed, match="H1 is testing"):
        project.update_page("H1", author="agent:theorist", body="It does the opposite.")
    project.set_state("H1", "open", by="system")
    project.update_page("H1", author="agent:theorist", body="Sharper.")


def test_an_agent_cannot_decide_or_approve_through_fields(project):
    page = experiment(project)
    for fields in ({"state": "approved"}, {"state": "running"}):
        with pytest.raises(NotAllowed):
            project.update_page("E1", author="agent:designer", fields=fields)
    # Managed fields are not the caller's to set: they are left as they are.
    same = project.update_page(
        "E1",
        author="agent:designer",
        fields={"review": "accepted", "author": "human", "decided_by": "human", "id": "E9"},
    )
    assert same == page
    assert (same.review, same.author, same.id, same.state) == (
        "proposed",
        "agent:designer",
        "E1",
        "planned",
    )


def test_pipeline_code_sets_fields_but_does_not_rewrite_what_the_person_decided(project):
    finding = project.create_page("finding", "Gap", "Seen: 0.9.", author="agent:analyst")
    flagged = project.update_page(finding.path, author="system", fields={"checked": ["0.9"]})
    assert flagged.meta["checked"] == ["0.9"] and flagged.review == "proposed"
    project.update_page(finding.path, author="system", body="Rewritten while undecided.")
    project.review(finding.path, "accept")
    project.update_page(finding.path, author="system", fields={"checked": None})
    assert "checked" not in project.page(finding.path).meta
    with pytest.raises(NotAllowed, match="does not rewrite its text"):
        project.update_page(finding.path, author="system", body="Rewritten after the decision.")
    with pytest.raises(NotAllowed, match="does not rewrite its text"):
        project.update_page("wiki/idea.md", author="system", body="A better idea.")


def test_editing_is_not_deciding(project):
    page = hypothesis(project)
    edited = project.update_page("H1", author="human", body="## Statement\n\nMy wording.")
    # The correction is in, and the decision is still open.
    assert edited.review == "proposed" and edited.author == "agent:theorist"
    assert "decided_by" not in edited.meta and edited.meta["updated_by"] == "human"
    assert [item.path for item in project.pending()] == [page.path]
    assert project.events()[-1]["type"] == "page.updated"
    assert project.history("H1")[0].author == "agent:theorist"
    # What the person wrote is not its author's to rewrite any more.
    with pytest.raises(NotAllowed, match="edited by the person"):
        project.update_page("H1", author="agent:theorist", body="Back to mine.")
    # Nor after pipeline code set a field on it since.
    project.update_page("H1", author="system", fields={"note": "seen"})
    with pytest.raises(NotAllowed, match="edited by the person"):
        project.update_page("H1", author="agent:theorist", body="Back to mine.")
    # A page that was turned down stays turned down, with its reason.
    other = project.review(hypothesis(project, "Other").path, "reject", reason="vague")
    reworded = project.update_page(other.path, author="human", body="Sharper now.")
    assert reworded.review == "rejected" and reworded.decision_reason == "vague"
    assert page.created == edited.created


def test_save_and_accept_is_the_explicit_decision(project):
    page = hypothesis(project)
    saved = project.update_page(
        "H1", author="human", body="## Statement\n\nMy wording.", accept=True
    )
    assert saved.review == "accepted" and saved.body.endswith("My wording.\n")
    assert saved.meta["decided_by"] == "human" and saved.meta["decided_at"] == saved.updated
    event = project.events()[-1]
    assert (event["type"], event["data"]["decision"], event["data"]["edited"]) == (
        "review.decided",
        "accept",
        True,
    )
    assert project.pending() == []
    with pytest.raises(NotAllowed, match="accepted by the person"):
        project.update_page("H1", author="agent:theorist", body="Back to mine.")
    # The person changed their mind about one they had turned down.
    other = project.review(hypothesis(project, "Other").path, "reject", reason="vague")
    revived = project.update_page(other.path, author="human", body="Sharper now.", accept=True)
    assert revived.review == "accepted" and "decision_reason" not in revived.meta
    # Saving an accepted page with it is an edit and nothing more.
    again = project.update_page("H1", author="human", fields={"summary": "Mine"}, accept=True)
    assert again.meta["decided_at"] == saved.meta["decided_at"]
    assert project.events()[-1]["type"] == "page.updated"
    # Only the person decides, however the call is made.
    third = hypothesis(project, "Third")
    with pytest.raises(NotAllowed, match="Only the person"):
        project.update_page(third.path, author="agent:theorist", body="x", accept=True)
    assert project.page(third.path).review == "proposed" and page.created == saved.created


def test_a_page_accepted_without_review_is_the_persons_once_they_edit_it(project):
    journal = project.create_page("journal", "Round 1", "What happened.", author="agent:analyst")
    assert journal.review == "accepted" and "decided_by" not in journal.meta
    edited = project.update_page(journal.path, author="human", body="What really happened.")
    assert edited.review == "accepted" and edited.meta["decided_by"] == "human"
    with pytest.raises(NotAllowed, match="accepted by the person"):
        project.update_page(journal.path, author="agent:analyst", body="Mine again.")


def test_a_page_of_the_persons_marked_proposed_by_hand_can_be_accepted(project):
    for name in ("one", "two", "three"):
        (project.root / "wiki" / "topics").mkdir(exist_ok=True)
        (project.root / "wiki" / "topics" / f"{name}.md").write_text(
            f"---\ntitle: {name}\nreview: proposed\n---\nWritten in an editor.\n"
        )
    assert [page.path for page in project.pending()] == [
        f"wiki/topics/{name}.md" for name in ("one", "three", "two")
    ]
    assert all(page.author == "human" for page in project.pending())
    # Accepting it clears it from the inbox ...
    accepted = project.review("topics/one", "accept")
    assert (accepted.review, accepted.decided_by) == ("accepted", "human")
    # ... so does accepting with a correction, which is applied ...
    edited = project.review("topics/three", "accept", edits={"body": "Corrected."})
    assert (edited.review, edited.body) == ("accepted", "Corrected.\n")
    # ... and so does saving it with the decision; a plain edit leaves it waiting.
    saved = project.update_page("topics/two", author="human", body="Rewritten.")
    assert saved.review == "proposed" and len(project.pending()) == 1
    saved = project.update_page("topics/two", author="human", accept=True)
    assert saved.review == "accepted"
    assert project.pending() == []
    # (The second one was saved twice: once as an edit, once with the decision.)
    assert [len(project.history(f"topics/{name}")) for name in ("one", "two", "three")] == [1, 2, 1]
    # Their own accepted page has nothing to decide, and is never "rejected".
    before = raw(project, "wiki/idea.md")
    assert project.review("idea", "accept").decided_by is None
    assert raw(project, "wiki/idea.md") == before
    with pytest.raises(ProjectError, match="own page"):
        project.review("topics/one", "reject")


def test_a_form_that_sends_every_field_changes_only_what_differs(project):
    page = project.review(hypothesis(project).path, "accept")
    same = project.update_page(
        "H1",
        author="human",
        body=page.body,
        fields={"title": page.title, "kind": "hypothesis", "state": None, "round": 7},
        relations={name: page.relations.get(name, []) for name in model.RELATIONS},
        base_updated=page.updated,
    )
    assert same == page and project.history("H1") == project.history("H1")[:1]
    topic = project.update_page("wiki/idea.md", author="human", fields={"state": None})
    assert topic.state is None


def test_an_edit_based_on_a_stale_version_is_a_conflict(project):
    page = hypothesis(project)
    newer = project.update_page("H1", author="agent:theorist", body="Second version.")
    with pytest.raises(Conflict, match="H1 changed after the version being edited") as caught:
        project.update_page("H1", author="human", body="Mine.", base_updated=page.updated)
    assert caught.value.page == newer
    assert project.page("H1").body == "Second version.\n"
    done = project.update_page("H1", author="human", body="Mine.", base_updated=newer.updated)
    assert done.body == "Mine.\n"


def test_deleting_moves_the_page_to_history(project):
    topic = project.create_page("topic", "Warmup", "Kept.", author="human")
    text = raw(project, topic.path)
    project.delete_page(topic.path, author="human")
    assert not (project.root / topic.path).exists()
    assert project.resolve(topic.path) is None
    (version,) = project.history(topic.path)
    assert project.version(topic.path, version.stamp) == text
    assert project.events()[-1]["type"] == "page.deleted"
    assert project.events()[-1]["summary"] == 'You deleted topic "Warmup".'
    with pytest.raises(ProjectError, match=r"idea page.*cannot be deleted"):
        project.delete_page("wiki/idea.md", author="human")
    with pytest.raises(NotFound):
        project.history("wiki/topics/never-there.md")


# -- review ---------------------------------------------------------------------


def test_accepting_and_rejecting_a_proposed_page(project):
    first, second = hypothesis(project, "One"), hypothesis(project, "Two")
    accepted = project.review("H1", "accept")
    assert accepted.review == "accepted" and accepted.state == "open"
    assert (
        accepted.meta["decided_by"] == "human" and accepted.meta["decided_at"] == accepted.updated
    )
    assert "decision_reason" not in accepted.meta
    assert accepted.body == first.body

    rejected = project.review(second.path, "reject", reason="Not falsifiable as stated")
    assert rejected.review == "rejected"
    assert rejected.decision_reason == "Not falsifiable as stated"
    assert rejected.body == second.body and (project.root / second.path).is_file()

    decided = [event for event in project.events() if event["type"] == "review.decided"]
    assert [event["summary"] for event in decided] == [
        'You accepted hypothesis H1 "One".',
        'You rejected hypothesis H2 "Two": Not falsifiable as stated.',
    ]
    assert decided[1]["data"]["decision"] == "reject" and decided[1]["refs"] == [second.path]
    assert project.pending() == []
    # Deciding the same thing twice changes nothing; the person may change their mind.
    assert project.review("H1", "accept") == accepted
    assert project.review("H2", "accepted").review == "accepted"


def test_the_persons_edits_are_applied_on_accepting(project):
    page = hypothesis(project)
    accepted = project.review(
        "H1",
        "accept",
        edits={
            "body": "## Statement\n\nSharper.",
            "fields": {"title": "Warmup halves variance", "summary": "A sharper claim"},
            "relations": {"cites": ["Smith2020Deep"]},
        },
    )
    assert accepted.title == "Warmup halves variance" and accepted.summary == "A sharper claim"
    assert accepted.body == "## Statement\n\nSharper.\n" and accepted.review == "accepted"
    assert accepted.relations == {"cites": ["Smith2020Deep"]}
    assert "It does." in project.version("H1", page.updated)
    assert len(project.history("H1")) == 1
    assert project.events()[-1]["data"]["edited"] is True
    with pytest.raises(ProjectError, match="'body', 'fields' and/or 'relations'"):
        project.review(hypothesis(project, "Other").path, "accept", edits={"text": "x"})


def test_only_the_person_decides(project):
    page = hypothesis(project)
    for by in ("agent:theorist", "system", "agent:analyst"):
        with pytest.raises(NotAllowed, match="Only the person accepts or rejects"):
            project.review("H1", "accept", by=by)
    proposal = project.propose_edit("wiki/idea.md", author="agent:analyst", summary="s", body="b")
    with pytest.raises(NotAllowed):
        project.review(proposal.id, "accept", by="agent:analyst")
    with pytest.raises(ProjectError, match="'accept' or 'reject'"):
        project.review("H1", "maybe")
    assert project.page("H1") == page and project.proposal(proposal.id).review == "proposed"
    with pytest.raises(ProjectError, match="person's own page"):
        project.review("wiki/idea.md", "reject")


def test_accepting_an_experiment_does_not_approve_it(project):
    experiment(project)
    accepted = project.review("E1", "accept")
    assert (accepted.review, accepted.state) == ("accepted", "planned")
    project.set_state("E1", "approved", by="human")
    # Turning the plan down afterwards takes the approval with it.
    rejected = project.review("E1", "reject", reason="too costly")
    assert (rejected.review, rejected.state) == ("rejected", "planned")


def test_the_person_decides_on_the_version_they_read(project):
    seen = experiment(project)
    project.update_page("E1", author="agent:designer", fields={"network": True})
    with pytest.raises(Conflict, match="E1 changed after the version"):
        project.review("E1", "accept", base_updated=seen.updated)
    assert project.page("E1").review == "proposed"
    current = project.page("E1")
    accepted = project.review("E1", "accept", base_updated=current.updated)
    with pytest.raises(Conflict, match="E1 changed after the version"):
        project.set_state("E1", "approved", by="human", base_updated=current.updated)
    assert project.set_state("E1", "approved", by="human", base_updated=accepted.updated).state == (
        "approved"
    )


# -- edit proposals -------------------------------------------------------------


def test_an_edit_proposal_is_stored_in_the_documented_shape(project):
    page = project.review(hypothesis(project).path, "accept")
    source = project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    proposal = project.propose_edit(
        "H1",
        author="agent:analyst",
        summary="Mark H1 as supported by F1",
        fields={"state": "supported"},
        relations_add={"cites": ["smith2020deep"]},
        round=2,
    )
    stored = json.loads(raw(project, f"proposals/{proposal.id}.json"))
    assert stored == {
        "id": proposal.id,
        "target": page.path,
        "author": "agent:analyst",
        "round": 2,
        "created": proposal.created,
        "summary": "Mark H1 as supported by F1",
        "body": None,
        "fields": {"state": "supported"},
        "relations_add": {"cites": [source.id]},
        "base_updated": page.updated,
        "review": "proposed",
        "decided_by": None,
        "decided_at": None,
        "decision_reason": None,
    }
    assert proposal.id.startswith("ep-") and len(proposal.id) == 11
    assert project.proposals() == [proposal] == [EditProposal.from_dict(stored)]
    assert project.pending() == [proposal]
    event = project.events()[-1]
    assert event["type"] == "proposal.created" and event["agent"] == "analyst"
    assert event["summary"] == (
        'The analyst proposed an edit to hypothesis H1 "Warmup reduces variance": '
        '"Mark H1 as supported by F1".'
    )
    assert event["refs"] == [page.path, f"proposals/{proposal.id}.json"]
    # Nothing changed yet.
    assert project.page("H1") == page


def test_accepting_an_edit_applies_it_and_keeps_the_previous_version(project):
    page = project.review(hypothesis(project).path, "accept")
    project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    proposal = project.propose_edit(
        "H1",
        author="agent:analyst",
        summary="Support and sharpen",
        body="## Statement\n\nIt does, by half.",
        fields={"state": "supported", "confidence": 0.9},
        relations_add={"cites": ["Smith2020Deep"]},
    )
    decided = project.review(proposal.id, "accept")
    assert decided.review == "accepted" and decided.decided_by == "human" and decided.decided_at
    assert project.proposal(proposal.id) == decided
    new = project.page("H1")
    assert new.state == "supported" and new.meta["confidence"] == 0.9
    assert new.body == "## Statement\n\nIt does, by half.\n"
    assert new.relations == {"cites": ["Smith2020Deep"]}
    # The page stays what the person accepted, written this time by the analyst.
    assert (new.review, new.author, new.meta["updated_by"]) == (
        "accepted",
        "agent:theorist",
        "agent:analyst",
    )
    assert project.version("H1", page.updated).endswith(page.body)
    assert project.proposals() == [] and project.proposals(review=None) == [decided]
    assert project.events()[-1]["summary"] == (
        'You accepted the analyst\'s edit to hypothesis H1 "Warmup reduces variance": '
        '"Support and sharpen".'
    )
    with pytest.raises(ProjectError, match="already accepted"):
        project.review(proposal.id, "reject")


def test_an_accepted_edit_makes_an_unreviewed_page_the_persons_decision(project):
    journal = project.create_page("journal", "Round 1", "What happened.", author="agent:analyst")
    assert journal.review == "accepted" and journal.decided_by is None
    project.update_page(journal.path, author="agent:analyst", body="Still mine to change.")
    proposal = project.propose_edit(
        journal.path, author="agent:scout", summary="Add the count", body="12 papers verified."
    )
    project.review(proposal.id, "accept")
    page = project.page(journal.path)
    assert page.decided_by == "human" and page.body == "12 papers verified.\n"
    with pytest.raises(NotAllowed, match="accepted by the person; propose an edit"):
        project.update_page(journal.path, author="agent:analyst", body="Back to my version.")
    # A page still waiting for review stays that: the edit was accepted, not the page.
    pending = hypothesis(project)
    proposal = project.propose_edit(pending.path, author="agent:analyst", summary="s", body="b")
    project.review(proposal.id, "accept")
    assert project.page(pending.path).review == "proposed"
    assert project.page(pending.path).decided_by is None


def test_rejecting_an_edit_leaves_the_page_alone(project):
    page = project.review(hypothesis(project).path, "accept")
    proposal = project.propose_edit("H1", author="agent:analyst", summary="Drop it", body="Gone.")
    decided = project.review(proposal.id, "reject", reason="It still stands")
    assert decided.review == "rejected" and decided.decision_reason == "It still stands"
    assert project.page("H1") == page and project.history("H1") != []
    assert len(project.history("H1")) == 1  # only the accept wrote a new version


def test_a_person_editing_meanwhile_makes_the_proposed_text_a_conflict(project):
    project.review(hypothesis(project).path, "accept")
    with_fields = project.propose_edit(
        "H1",
        author="agent:analyst",
        summary="Support and rewrite",
        body="The analyst's text.",
        fields={"state": "supported"},
    )
    text_only = project.propose_edit(
        "H1", author="agent:analyst", summary="Rewrite", body="The analyst's other text."
    )
    edited = project.update_page("H1", author="human", body="The person's text.")

    with pytest.raises(Conflict, match="replacement text was not applied") as caught:
        project.review(with_fields.id, "accept")
    assert caught.value.applied is True and "field changes were applied" in str(caught.value)
    page = project.page("H1")
    assert page.state == "supported" and page.body == "The person's text.\n"
    assert caught.value.page == page
    left = project.proposal(with_fields.id)
    assert left.review == "accepted" and "after this edit was proposed" in left.conflict
    assert caught.value.proposal == left
    assert json.loads(raw(project, f"proposals/{left.id}.json"))["conflict"] == left.conflict

    # With nothing but text to apply, nothing happens and the proposal stays open.
    events = len(project.events())
    with pytest.raises(Conflict, match="Nothing was changed") as caught:
        project.review(text_only.id, "accept")
    assert caught.value.applied is False
    assert project.page("H1") == page and len(project.events()) == events
    assert project.proposal(text_only.id).review == "proposed"
    assert project.proposal(text_only.id).conflict
    # The person can still take it, with text of their own.
    done = project.review(text_only.id, "accept", edits={"body": "Merged by hand."})
    assert done.review == "accepted" and project.page("H1").body == "Merged by hand.\n"
    assert edited.updated < page.updated


def test_an_accept_cut_off_before_the_proposal_was_closed_can_be_repeated(project):
    project.review(hypothesis(project).path, "accept")
    proposal = project.propose_edit(
        "H1", author="agent:analyst", summary="Sharpen", body="Sharper.", fields={"summary": "S"}
    )
    file = project.root / "proposals" / f"{proposal.id}.json"
    open_text = file.read_text()
    project.review(proposal.id, "accept")
    applied = project.page("H1")
    # The process died after writing the page: the proposal file still says "proposed".
    file.write_text(open_text)
    assert [p.id for p in project.proposals()] == [proposal.id]
    again = project.review(proposal.id, "accept")
    assert again.review == "accepted" and again.conflict is None
    assert project.page("H1") == applied and project.proposals() == []


def test_a_change_that_left_the_text_alone_is_no_conflict(project):
    page = project.review(hypothesis(project).path, "accept")
    rewrite = project.propose_edit("H1", author="agent:analyst", summary="Rewrite", body="New.")
    verdict = project.propose_edit(
        "H1", author="agent:analyst", summary="Supported", fields={"state": "supported"}
    )
    # The other edit, a state set by code and a decision all change the page, not its text.
    project.review(verdict.id, "accept")
    project.set_state("H1", "open", by="human")
    assert project.page("H1").updated > page.updated
    assert project.review(rewrite.id, "accept").conflict is None
    assert project.page("H1").body == "New.\n"


def test_what_an_edit_may_not_propose(project):
    exp = project.review(experiment(project).path, "accept")
    hyp = project.review(hypothesis(project).path, "accept")
    source = project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    with pytest.raises(NotAllowed, match="state of an experiment changes only when the person"):
        project.propose_edit(
            exp.path, author="agent:analyst", summary="Go", fields={"state": "approved"}
        )
    for field in ("review", "author", "decided_by", "id", "kind", "session"):
        with pytest.raises(NotAllowed, match=f"'{field}' is set by the project"):
            project.propose_edit(hyp.path, author="agent:analyst", summary="s", fields={field: "x"})
    with pytest.raises(ProjectError, match="'done' is not a state of a hypothesis"):
        project.propose_edit(
            hyp.path, author="agent:analyst", summary="s", fields={"state": "done"}
        )
    with pytest.raises(NotAllowed, match="is a record"):
        project.propose_edit(
            source.path, author="agent:scout", summary="s", body="Better abstract."
        )
    with pytest.raises(ProjectError, match="needs a one-line summary"):
        project.propose_edit(hyp.path, author="agent:analyst", summary=" ", body="x")
    with pytest.raises(ProjectError, match="changes something"):
        project.propose_edit(hyp.path, author="agent:analyst", summary="Nothing")
    with pytest.raises(ProjectError, match="'tests' is not a field of a hypothesis"):
        project.propose_edit(
            hyp.path, author="agent:analyst", summary="s", relations_add={"tests": ["H1"]}
        )
    with pytest.raises(NotFound):
        project.propose_edit("H9", author="agent:analyst", summary="s", body="x")
    assert project.proposals() == []


def test_a_proposal_file_cannot_approve_an_experiment(project):
    exp = project.review(experiment(project).path, "accept")
    # Written around the store, as a tampered or hand-made file would be.
    forged = EditProposal(
        id="ep-0badc0de",
        target=exp.path,
        author="agent:designer",
        created="2026-11-15T12:00:00Z",
        summary="Fix a typo",
        fields={"state": "approved"},
        base_updated=exp.updated,
    )
    (project.root / "proposals" / "ep-0badc0de.json").write_text(json.dumps(forged.to_dict()))
    with pytest.raises(NotAllowed, match="Only the person does that"):
        project.review("ep-0badc0de", "accept")
    assert project.page("E1").state == "planned"


def test_an_edit_whose_page_is_gone_can_only_be_rejected(project):
    topic = project.create_page("topic", "Warmup", "a", author="human")
    proposal = project.propose_edit(topic.path, author="agent:scout", summary="Add", body="b")
    project.delete_page(topic.path, author="human")
    with pytest.raises(NotFound, match=r"no longer exists. Reject the proposal"):
        project.review(proposal.id, "accept")
    assert project.review(proposal.id, "reject").review == "rejected"


# -- lifecycle ------------------------------------------------------------------


def test_an_experiment_runs_only_after_the_person_approved_it(project):
    experiment(project)
    with pytest.raises(ProjectError, match="E1 is proposed, not accepted"):
        project.set_state("E1", "approved", by="human")
    project.review("E1", "accept")
    for by in ("system", "agent:runner", "agent:designer"):
        with pytest.raises(NotAllowed, match="E1: Only the person"):
            project.set_state("E1", "approved", by=by)
    with pytest.raises(ProjectError, match="E1: An experiment that is planned cannot go straight"):
        project.set_state("E1", "running", by="system")
    assert project.page("E1").state == "planned"

    approved = project.set_state("E1", "approved", by="human", reason="cheap enough")
    assert approved.state == "approved" and approved.meta["state_reason"] == "cheap enough"
    event = project.events()[-1]
    assert event["type"] == "experiment.state"
    assert event["summary"] == 'Experiment E1 "Sweep warmup length" is now approved.'
    assert event["data"]["previous"] == "planned" and event["data"]["by"] == "human"

    events = len(project.events())
    assert project.set_state("E1", "approved", by="human") == approved
    assert len(project.events()) == events

    for state in ("queued", "running", "succeeded"):
        assert project.set_state("E1", state, by="system").state == state
    assert "state_reason" not in project.page("E1").meta
    assert [version.author for version in project.history("E1")][:4] == [
        "system",
        "system",
        "human",
        "human",
    ]
    # Running it again is a new decision.
    with pytest.raises(NotAllowed):
        project.set_state("E1", "approved", by="system")
    assert project.set_state("E1", "approved", by="human").state == "approved"


def test_setting_the_state_a_page_is_in_writes_nothing(project):
    (project.root / "wiki" / "hypotheses").mkdir(exist_ok=True)
    path = project.root / "wiki" / "hypotheses" / "H7-by-hand.md"
    path.write_text("---\ntitle: By hand\n---\nThe person's own, with no state written.\n")
    before, events = path.read_text(), len(project.events())
    for by in ("agent:theorist", "system", "human"):
        assert project.set_state("H7", "open", by=by).state == "open"
    # In particular an agent, who could make no real move, has not touched the person's page.
    assert path.read_text() == before and project.history("H7") == []
    assert len(project.events()) == events
    with pytest.raises(NotAllowed):
        project.set_state("H7", "supported", by="agent:theorist")
    assert project.set_state("H7", "supported", by="human").meta["state"] == "supported"


def test_verdicts_on_hypotheses_are_set_by_the_person(project):
    project.review(hypothesis(project).path, "accept")
    assert project.set_state("H1", "testing", by="system").state == "testing"
    with pytest.raises(NotAllowed, match="H1: Whether a hypothesis is supported"):
        project.set_state("H1", "supported", by="agent:analyst")
    done = project.set_state("H1", "supported", by="human")
    assert done.state == "supported"
    assert project.events()[-1]["type"] == "page.updated"
    assert (
        project.events()[-1]["summary"]
        == 'Hypothesis H1 "Warmup reduces variance" is now supported.'
    )
    with pytest.raises(ProjectError, match="no lifecycle"):
        project.set_state("wiki/idea.md", "open", by="human")
    with pytest.raises(ProjectError, match="'done' is not a state"):
        project.set_state("H1", "done", by="human")


# -- records --------------------------------------------------------------------


def test_records_are_written_by_code_and_accepted(project):
    source = project.write_record(
        "source",
        "Smith2020Deep",
        "Deep things",
        "> Abstract, quoted.",
        summary="Smith et al. 2020",
        fields={"year": 2020, "url": "https://example.org/p", "review": "rejected"},
        round=1,
    )
    assert source.path == "wiki/sources/Smith2020Deep.md"
    assert (source.kind, source.id, source.author, source.review) == (
        "source",
        "Smith2020Deep",
        "system",
        "accepted",
    )
    assert source.meta["year"] == 2020 and source.round == 1 and source.state is None
    assert project.resolve("smith2020deep") == source.path
    assert project.events()[-1]["summary"] == (
        'The pipeline recorded source Smith2020Deep "Deep things".'
    )

    experiment(project)
    result = project.write_record(
        "result",
        "E1-run1",
        "E1 run 1",
        "| metric | value |\n|---|---|\n| accuracy | 0.91 |",
        fields={"state": "succeeded"},
        relations={"run_of": ["E1"]},
    )
    assert (result.path, result.id, result.state) == ("wiki/results/E1-run1.md", None, "succeeded")
    assert result.relations == {"run_of": ["E1"]} and result.review == "accepted"
    assert project.resolve("E1-run1") == project.resolve("results/E1-run1") == result.path

    # Writing it again replaces it and keeps the old version.
    again = project.write_record(
        "result", "E1-run1", "E1 run 1", "Corrected.", fields={"state": "failed"}
    )
    assert again.state == "failed" and again.created == result.created
    assert len(project.history(result.path)) == 1
    assert project.events()[-1]["type"] == "page.updated"

    with pytest.raises(ProjectError, match="write_record writes records"):
        project.write_record("topic", "x", "T", "b")
    with pytest.raises(ProjectError, match="cannot name a source"):
        project.write_record("source", "../escape", "T", "b")
    with pytest.raises(ProjectError, match="'done' is not a state of a result"):
        project.write_record("result", "E1-run2", "T", "b", fields={"state": "done"})
    with pytest.raises(ProjectError, match="record"):
        project.set_state(result.path, "succeeded", by="system")


def test_the_person_may_turn_down_or_delete_a_source_and_code_respects_it(project):
    source = project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    rejected = project.review(source.path, "reject", reason="Not about warmup")
    assert rejected.review == "rejected"
    rewritten = project.write_record("source", "Smith2020Deep", "Deep things", "Abstract, again.")
    assert rewritten.review == "rejected" and rewritten.decision_reason == "Not about warmup"
    project.delete_page(source.path, author="human")
    assert project.resolve("Smith2020Deep") is None
    assert len(project.history(source.path)) == 3


def test_sources_wait_for_the_person_where_the_policy_says_so(project):
    project.settings.review["source"] = "review"
    assert project.write_record("source", "Lee2021Wide", "Wide", "A.").review == "proposed"
    # A result is what a run produced: never held back.
    assert project.write_record("result", "E1-run1", "R", "x").review == "accepted"


# -- requests -------------------------------------------------------------------


def test_requests_steer_the_agents(project):
    page = hypothesis(project)
    first = project.add_request("Look at small batch sizes too")
    second = project.add_request("  Is this already known?  ", target="h1")
    assert first.id.startswith("rq-") and len(first.id) == 11
    assert (second.text, second.target, second.state) == (
        "Is this already known?",
        page.path,
        "open",
    )
    assert json.loads(raw(project, f"requests/{first.id}.json")) == {
        "id": first.id,
        "text": "Look at small batch sizes too",
        "target": None,
        "created": first.created,
        "state": "open",
        "for": "all",
        "handled_in": [],
    }
    assert project.requests() == [first, second]
    event = project.events()[-1]
    assert event["type"] == "request.created"
    assert event["summary"] == "You asked the agents: Is this already known?"
    assert event["refs"] == [f"requests/{second.id}.json", page.path]

    project.close_request(first.id, round=2, phase="hypothesize")
    assert project.requests() == [second]
    (done,) = project.requests(state="done")
    assert done.handled_in == [{"round": 2, "phase": "hypothesize"}]
    assert len(project.requests(state=None)) == 2
    project.delete_request(second.id)
    assert project.requests() == []

    with pytest.raises(ProjectError, match="needs some text"):
        project.add_request("   ")
    with pytest.raises(NotFound, match="No page matches 'H9'"):
        project.add_request("About this", target="H9")
    for missing in ("rq-ffffffff", "../project", None):
        with pytest.raises(NotFound):
            project.close_request(missing, round=1, phase="research")


# -- events ---------------------------------------------------------------------


def test_events_are_numbered_without_gaps_and_can_be_tailed(project):
    hypothesis(project)
    emitted = project.emit(
        "agent.activity",
        "Verified 12 of 15\ncandidate papers",
        phase="research",
        agent="scout",
        refs=("wiki/sources/Smith2020Deep.md",),
        data={"verified": 12},
        round=1,
    )
    assert emitted == {
        "seq": 3,
        "ts": emitted["ts"],
        "type": "agent.activity",
        "phase": "research",
        "round": 1,
        "agent": "scout",
        "summary": "Verified 12 of 15 candidate papers",
        "refs": ["wiki/sources/Smith2020Deep.md"],
        "data": {"verified": 12},
    }
    assert model.parse_stamp(emitted["ts"]) is not None
    events = project.events()
    assert [event["seq"] for event in events] == [1, 2, 3] and events[-1] == emitted
    assert project.events(after=2) == [emitted] and project.events(after=3) == []
    lines = raw(project, "events.jsonl").splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [1, 2, 3]
    # Another opening of the same project continues the count.
    assert Project.open(project.root).emit("checkpoint", "Waiting for you.")["seq"] == 4
    assert project.emit("error", "Something broke.")["seq"] == 5
    with pytest.raises(ProjectError, match=r"'page.renamed' is not an event type"):
        project.emit("page.renamed", "No such thing.")


def test_a_line_cut_off_by_a_crash_does_not_swallow_the_next_event(project):
    with (project.root / "events.jsonl").open("a") as handle:
        handle.write('{"seq": 2, "ts": "2026-11-15T12:00:0')
    event = Project.open(project.root).emit("error", "The worker died.")
    assert event["seq"] == 2
    assert [e["seq"] for e in project.events()] == [1, 2]
    assert project.events()[-1]["summary"] == "The worker died."


def test_events_carry_the_round_and_phase_the_engine_is_in(project):
    assert project.current_round() == 0
    (project.root / "state.json").write_text(
        json.dumps({"schema": 1, "status": "running", "round": 2, "phase": "hypothesize"})
    )
    page = hypothesis(project)
    assert page.round == 2
    event = project.events()[-1]
    assert (event["round"], event["phase"]) == (2, "hypothesize")
    (project.root / "state.json").write_text(
        json.dumps({"schema": 1, "status": "checkpoint", "round": 2, "phase": "hypothesize"})
    )
    decided = Project.open(project.root).review("H1", "accept")
    assert decided.round == 2
    event = project.events()[-1]
    assert (event["round"], event["phase"]) == (2, "")


# -- graph ----------------------------------------------------------------------


def test_text_shown_as_code_is_not_a_link(project):
    # A result record quotes the job's log.  numpy prints matrices as [[0.1 0.2]], and a job
    # may print anything: neither is a statement by the page about another page.
    hyp = hypothesis(project)
    record = project.write_record(
        "result",
        "E9-run1",
        "A run",
        "Mentions [[H1]] in prose.\n\n"
        "```\nweights [[0.1 0.2]]\nsee [[idea]]\n```\n\n"
        "~~~text\n[[topics/nowhere]]\n~~~\n\n"
        "Inline `[[idea]]` and ``[[topics/nowhere]]`` are code too.\n",
    )
    graph = project.graph()
    assert [(e["target"], e["type"]) for e in graph["edges"] if e["source"] == record.path] == [
        (hyp.path, "mentions")
    ]
    assert graph["dangling"] == []


def test_the_graph_has_typed_edges_mentions_and_reports_what_dangles(project):
    source = project.write_record("source", "Smith2020Deep", "Deep things", "Abstract.")
    topic = project.create_page(
        "topic",
        "Warmup",
        "Background from [[sources/Smith2020Deep]] and [[idea]]; see [[topics/missing]].",
        author="human",
        relations={"cites": ["Smith2020Deep"]},
    )
    hyp = hypothesis(
        project,
        relations={"motivated_by": [topic.link, "Smith2020Deep"], "cites": ["Ghost1999"]},
    )
    project.update_page(
        hyp.path, author="agent:theorist", body="As in [[H1]] and [[topics/warmup]]."
    )
    exp = experiment(project, relations={"tests": ["H1"]})
    turned_down = project.review(
        hypothesis(project, "Bad", relations={"motivated_by": ["topics/warmup"]}).path, "reject"
    )

    graph = project.graph()
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert set(nodes) == {"wiki/idea.md", source.path, topic.path, hyp.path, exp.path}
    assert nodes[hyp.path] == {
        "id": hyp.path,
        "kind": "hypothesis",
        "label": "H1",
        "title": "Warmup reduces variance",
        # Taken from the text, where the links are markup: here they are words.
        "summary": "As in H1 and warmup.",
        "review": "proposed",
        "state": "open",
        "author": "agent:theorist",
        "round": 0,
    }
    assert nodes[topic.path]["label"] == "Warmup" and nodes[source.path]["label"] == "Smith2020Deep"
    edges = {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]}
    assert edges == {
        (topic.path, source.path, "cites"),
        (topic.path, "wiki/idea.md", "mentions"),
        (hyp.path, topic.path, "motivated_by"),
        (hyp.path, source.path, "motivated_by"),
        (exp.path, hyp.path, "tests"),
    }
    assert len(graph["edges"]) == len(edges)
    assert graph["dangling"] == [
        {"source": topic.path, "target": "topics/missing", "type": "mentions"},
        {"source": hyp.path, "target": "Ghost1999", "type": "cites"},
    ]

    everything = project.graph(include_rejected=True)
    assert turned_down.path in {node["id"] for node in everything["nodes"]}
    assert {"source": turned_down.path, "target": topic.path, "type": "motivated_by"} in everything[
        "edges"
    ]


# -- what agents are told -------------------------------------------------------


def test_the_index_lists_what_agents_may_build_on(project):
    accepted = project.review(
        hypothesis(project, summary="Warmup lowers early variance").path, "accept"
    )
    hypothesis(project, "A secret hunch", summary="Nobody has seen this yet")
    project.review(
        hypothesis(project, "A vague hunch", summary="It is all connected").path,
        "reject",
        reason="Not falsifiable",
    )
    project.create_page("finding", "The analyst's own", "x", author="agent:analyst")
    project.write_record(
        "source", "Smith2020Deep", "Deep things", "Abstract.", summary="Smith 2020"
    )

    index = project.index_text()
    assert index.startswith("# Research wiki: Warmup and variance\n")
    assert "## idea (1)" in index and "- idea.md [accepted] Warmup and variance" in index
    assert (
        "- hypotheses/H1-warmup-reduces-variance.md [H1, accepted, open] Warmup reduces variance: "
        "Warmup lowers early variance" in index
    )
    assert "- sources/Smith2020Deep.md [Smith2020Deep, accepted] Deep things: Smith 2020" in index
    # Someone's unreviewed proposal is nobody else's context.
    assert "secret" not in index and "analyst's own" not in index
    # What was turned down: title and reason, not the content.
    assert '- hypothesis H3 "A vague hunch": Not falsifiable' in index
    assert "It is all connected" not in index and "H3-a-vague-hunch" not in index
    assert raw(project, "MEMORY.md") == index

    for_theorist = project.index_text(for_role="theorist")
    assert project.index_text(for_role="agent:theorist") == for_theorist
    assert "As the theorist you can create pages under: hypotheses/, questions/." in for_theorist
    # A role is not a session: what the theorist proposed in another session stays out.
    assert "secret" not in for_theorist and "analyst's own" not in for_theorist
    assert "you can create" not in project.index_text(for_role="runner")
    # A session is shown what it proposed itself.
    mine = project.create_page(
        "hypothesis", "Written just now", "x", author="agent:theorist", fields={"session": "s9"}
    )
    in_session = project.index_text(for_role="theorist", session="s9")
    assert f"[{mine.id}, proposed in this session, open] Written just now" in in_session
    assert "secret" not in in_session
    assert "Written just now" not in project.index_text(for_role="theorist", session="other")
    assert accepted.path.removeprefix("wiki/") in for_theorist


def test_the_index_stays_within_its_budget(project):
    for number in range(120):
        project.write_record(
            "source",
            f"Author{number}Paper",
            f"A long title about topic number {number}",
            "x",
            summary="A summary that goes on for a while about what the paper found " * 2,
        )
    hypothesis(project, author="human")
    index = project.index_text(max_chars=6000)
    assert len(index) <= 6000
    assert "## hypotheses (1)" in index and "H1" in index
    assert "## sources (120)" in index
    listed = [line for line in index.splitlines() if line.startswith("- sources/")]
    assert 0 < len(listed) <= 40
    assert f"- ... {120 - len(listed)} earlier pages are not listed" in index
    assert "wiki_search" in index
    assert len(project.index_text()) <= 20_000
    assert len(project.index_text(max_chars=300)) <= 300 + 80


def test_counts(project):
    project.review(hypothesis(project, "One").path, "accept")
    hypothesis(project, "Two")
    project.review(hypothesis(project, "Three").path, "reject")
    project.review(experiment(project).path, "accept")
    experiment(project, "Other")
    project.review(experiment(project, "Bad").path, "reject")
    project.set_state("E1", "approved", by="human")
    project.propose_edit("H1", author="agent:analyst", summary="s", body="b")
    project.add_request("Do more")
    counts = project.counts()
    assert counts["pages"] == 7
    assert counts["by_kind"]["hypothesis"] == 3 and counts["by_kind"]["idea"] == 1
    assert counts["by_kind"]["source"] == 0 and set(counts["by_kind"]) == set(model.KINDS)
    assert counts["by_review"] == {"proposed": 2, "accepted": 3, "rejected": 2}
    assert counts["pending"] == {
        "total": 3,
        "pages": 2,
        "proposals": 1,
        "by_kind": {"hypothesis": 1, "experiment": 1},
    }
    assert counts["experiments"] == {
        "planned": 1,
        "approved": 1,
        "queued": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "cancelled": 0,
    }
    assert counts["requests"] == {"open": 1}
    assert len(project.pending()) == 3


def test_search_covers_accepted_pages_unless_asked_otherwise(project):
    project.review(hypothesis(project, "Warmup reduces gradient variance").path, "accept")
    hypothesis(project, "Cosine decay reduces gradient variance")
    assert [hit.path for hit in project.search("gradient")] == [
        "wiki/hypotheses/H1-warmup-reduces-gradient-variance.md"
    ]
    everything = project.search("gradient", review=None)
    assert len(everything) == 2
    assert project.search("gradient", review="proposed")[0].title.startswith("Cosine")
    assert project.search("gradient", kind="topic", review=None) == []
    assert [hit.path for hit in project.search("variance", kind="idea")] == ["wiki/idea.md"]
    assert project.search("") == []


# -- pages written by hand ------------------------------------------------------


def test_a_wiki_edited_by_hand_still_opens_and_says_what_is_wrong(project):
    wiki = project.root / "wiki"
    (wiki / "hypotheses").mkdir(exist_ok=True)
    (wiki / "hypotheses" / "my-hunch.md").write_text(
        "---\nkind: essay\nstate: maybe\ntests: H1\n---\n# A hunch\n\nIt might work.\n"
    )
    (wiki / "hypotheses" / "H4-first.md").write_text(
        "---\ntitle: First\nid: H4\ncites: Ghost1999\n---\nx\n"
    )
    (wiki / "hypotheses" / "H4-second.md").write_text("---\ntitle: Second\nid: H4\n---\nx\n")
    (wiki / "essays").mkdir()
    (wiki / "essays" / "loose.md").write_text("No front matter at all.\n")
    (wiki / "questions").mkdir(exist_ok=True)
    (wiki / "questions" / "Q1-open.md").write_text("---\ntitle: Why?\nabout: H4\n---\nx\n")
    (wiki / "topics").mkdir(exist_ok=True)
    (wiki / "topics" / "binary.md").write_bytes(b"\xff\xfe\x00not text")
    (project.root / "proposals" / "ep-12345678.json").write_text("{broken")
    (project.root / "requests" / "rq-12345678.json").write_text("[1, 2]")

    opened = Project.open(project.root)
    pages = {page.path: page for page in opened.pages()}
    hunch = pages["wiki/hypotheses/my-hunch.md"]
    assert (hunch.kind, hunch.title, hunch.review, hunch.author) == (
        "hypothesis",
        "A hunch",
        "accepted",
        "human",
    )
    assert hunch.id is None and hunch.state == "open" and model.parse_stamp(hunch.updated)
    assert pages["wiki/essays/loose.md"].kind == "topic"
    assert "wiki/topics/binary.md" not in pages
    assert opened.proposals() == [] and opened.requests() == []

    report = {(problem["path"], problem["message"]) for problem in opened.problems()}
    said = "\n".join(f"{path}: {message}" for path, message in sorted(report))
    for needle in (
        "wiki/hypotheses/my-hunch.md: kind 'essay' is not a kind of page",
        "wiki/hypotheses/my-hunch.md: no id",
        "wiki/hypotheses/my-hunch.md: 'maybe' is not a state",
        "wiki/hypotheses/my-hunch.md: 'tests' is not a relation of a hypothesis",
        "wiki/hypotheses/H4-second.md: id H4 is also used by wiki/hypotheses/H4-first.md",
        "wiki/hypotheses/H4-first.md: cites: 'Ghost1999' does not match any page",
        "wiki/questions/Q1-open.md: about: 'H4' is a hypothesis; about points at: idea, topic",
        "wiki/essays/loose.md: no kind in the front matter",
        "proposals/ep-12345678.json: not a JSON object",
        "requests/rq-12345678.json: not a JSON object",
    ):
        assert needle in said, needle

    # The store keeps working on such a wiki, and a person's edit repairs a page.
    assert hypothesis(opened, "Next").id == "H5"
    fixed = opened.update_page(
        hunch.path, author="human", fields={"title": "A hunch", "state": "open"}
    )
    assert fixed.state == "open" and fixed.created == hunch.created
    assert opened.graph()["nodes"] and opened.index_text() and opened.counts()["pages"] == 7
    (project.root / "wiki" / "idea.md").unlink()
    assert {"path": "wiki/idea.md", "message": "the idea page is missing"} in opened.problems()


def test_a_page_saved_with_a_byte_order_mark_keeps_its_front_matter(project):
    path = project.root / "wiki" / "experiments" / "E1-by-hand.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\ufeff---\r\ntitle: Saved by an editor\r\nstate: approved\r\n---\r\nThe plan.\r\n"
    path.write_text(text, encoding="utf-8", newline="")
    page = project.page("E1")
    assert (page.title, page.state, page.body) == ("Saved by an editor", "approved", "The plan.\n")
    assert [p.title for p in project.pages(kind="experiment")] == ["Saved by an editor"]
    # The first change through the store writes it back clean, front matter intact.
    edited = project.update_page("E1", author="human", fields={"summary": "A plan"})
    assert (edited.state, edited.body) == ("approved", "The plan.\n")
    assert raw(project, "wiki/experiments/E1-by-hand.md").startswith("---\ntitle: Saved by")


def test_listings_the_wiki_layer_generates_are_not_pages(project):
    project.wiki.write(
        "wiki/sources/index.md",
        "- [[sources/A]]",
        {"title": "sources index", "kind": "index", "auto": True},
        force=True,
    )
    assert [page.path for page in project.pages()] == ["wiki/idea.md"]
    assert project.problems() == []


# -- several writers at once ----------------------------------------------------

WRITER = """
import sys, time
from pathlib import Path
from paper_orchestra.research.project import Project

project = Project.open(sys.argv[1])
name, count, gate = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
(gate / f"{name}.ready").touch()
while not (gate / "go").exists():
    time.sleep(0.005)
for number in range(count):
    author = "human" if name == "web" else "agent:theorist"
    page = project.create_page("hypothesis", f"{name} {number}", "text", author=author)
    project.emit("agent.activity", f"{name} wrote {page.id}.", agent="theorist")
    project.create_page("topic", "Same title", name, author=author)
"""


def test_a_web_server_and_workers_create_pages_at_the_same_time(project, tmp_path):
    names, count = ("web", "worker", "second", "third"), 12
    gate = tmp_path / "gate"
    gate.mkdir()
    writers = [
        subprocess.Popen(
            [sys.executable, "-c", WRITER, str(project.root), name, str(count), str(gate)],
            stderr=subprocess.PIPE,
            text=True,
        )
        for name in names
    ]
    # Let them all start writing at the same moment, once each has opened the project.
    deadline = time.monotonic() + 60
    while len(list(gate.glob("*.ready"))) < len(names) and time.monotonic() < deadline:
        assert all(writer.poll() is None for writer in writers), "a writer died while starting"
        time.sleep(0.01)
    (gate / "go").touch()
    for writer in writers:
        _, errors = writer.communicate(timeout=120)
        assert writer.returncode == 0, errors

    total = len(names) * count
    hypotheses = project.pages(kind="hypothesis")
    assert sorted(page.id for page in hypotheses) == sorted(f"H{n}" for n in range(1, total + 1))
    assert len({page.path for page in hypotheses}) == total
    assert {page.title for page in hypotheses} == {f"{n} {i}" for n in names for i in range(count)}
    # Pages without numbers did not overwrite each other either.
    topics = project.pages(kind="topic")
    assert len(topics) == total
    assert sorted(page.body.strip() for page in topics) == sorted(names * count)

    events = project.events()
    assert [event["seq"] for event in events] == list(range(1, 3 * total + 2))
    created = [e["data"]["id"] for e in events if e["type"] == "page.created" and e["data"]["id"]]
    assert sorted(created) == sorted(page.id for page in hypotheses)
    lines = raw(project, "events.jsonl").splitlines()
    assert [json.loads(line)["seq"] for line in lines] == list(range(1, 3 * total + 2))
    assert raw(project, "MEMORY.md") == project.index_text()


def test_a_project_that_stays_locked_is_reported_and_left_alone(project, monkeypatch):
    from paper_orchestra.research import project as store

    monkeypatch.setattr(store, "LOCK_WAIT_SECONDS", 0.05)
    events = len(project.events())
    with store._FileLock(project.root / store.LOCK_FILE, wait_seconds=1, poll=0.01) as held:
        assert held.acquired
        with pytest.raises(ProjectError, match=r"could not be locked.*Nothing was changed"):
            hypothesis(project)
        with pytest.raises(ProjectError, match="could not be locked"):
            project.emit("error", "Not written.")
        # Reading never waits for a writer.
        assert project.pages()[0].kind == "idea" and len(project.events()) == events
    assert hypothesis(project).id == "H1"


def test_threads_sharing_a_project_do_not_collide(project):
    other = Project.open(project.root)
    errors: list[BaseException] = []

    def work(store: Project, name: str) -> None:
        try:
            for number in range(8):
                page = hypothesis(store, f"{name} {number}")
                store.update_page(page.path, author="agent:theorist", body=f"{name} revised")
                store.emit("agent.activity", f"{name} {number}.")
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=work, args=(store, f"t{index}"))
        for index, store in enumerate((project, project, other, other))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert errors == []
    pages = project.pages(kind="hypothesis")
    assert sorted(int(page.id[1:]) for page in pages) == list(range(1, 33))
    assert all(len(project.history(page.path)) == 1 for page in pages)
    seqs = [event["seq"] for event in project.events()]
    assert seqs == list(range(1, len(seqs) + 1)) and len(seqs) == 1 + 32 * 3


def test_every_change_is_in_the_history(still):
    page = hypothesis(still)
    texts = [raw(still, page.path)]
    still.update_page("H1", author="agent:theorist", body="Second.")
    texts.append(raw(still, page.path))
    still.review("H1", "accept", edits={"fields": {"summary": "Sharper"}})
    texts.append(raw(still, page.path))
    proposal = still.propose_edit(
        "H1", author="agent:analyst", summary="s", fields={"state": "testing"}
    )
    still.review(proposal.id, "accept")
    texts.append(raw(still, page.path))
    still.set_state("H1", "supported", by="human")
    texts.append(raw(still, page.path))
    still.delete_page("H1", author="human")

    history = still.history(page.path)
    assert [version.author for version in history] == [
        "human",
        "agent:analyst",
        "human",
        "agent:theorist",
        "agent:theorist",
    ]
    assert [still.version(page.path, version.stamp) for version in reversed(history)] == texts
    stamps = [split_front_matter(text)[0]["updated"] for text in texts]
    assert stamps == sorted(set(stamps))
    assert event_types(still)[1:] == [
        "page.created",
        "page.updated",
        "review.decided",
        "proposal.created",
        "review.decided",
        "page.updated",
        "page.deleted",
    ]


def test_a_version_whose_stamp_was_written_by_hand_is_still_kept(still):
    topic = still.create_page("topic", "Notes", "First.", author="human")
    path = still.root / topic.path
    # An editor saved the page with a date where the stamp belongs.
    path.write_text("---\ntitle: Notes\nupdated: 15 Nov 2026\n---\nSecond, by hand.\n")
    listed = still.page(topic.path)
    assert model.parse_stamp(listed.updated) is not None
    assert any("updated is '15 Nov 2026'" in p["message"] for p in still.problems())
    third = still.update_page(
        topic.path, author="human", body="Third.", base_updated=listed.updated
    )
    (kept,) = still.history(topic.path)
    assert kept.stamp == listed.updated and third.updated > kept.stamp
    assert "Second, by hand." in still.version(topic.path, kept.stamp)


def test_a_restored_and_edited_version_does_not_replace_the_one_it_came_from(still):
    topic = still.create_page("topic", "Notes", "First.", author="human")
    still.update_page(topic.path, author="human", body="Second.")
    still.update_page(topic.path, author="human", body="Third.")
    oldest = still.history(topic.path)[-1]
    first_text = still.version(topic.path, oldest.stamp)
    assert "First." in first_text
    # The person copies the first version back over the page and changes a word, leaving
    # its front matter (and so its stamp) as it was.
    (still.root / topic.path).write_text(first_text.replace("First.", "First, reworded."))
    fifth = still.update_page(topic.path, author="human", body="Fifth.")

    # (The third text is gone by the person's own hand; the store never saw it replaced.)
    history = still.history(topic.path)
    texts = [split_front_matter(still.version(topic.path, v.stamp))[1] for v in history]
    assert texts == ["First, reworded.\n", "Second.\n", "First.\n"]
    stamps = [version.stamp for version in history]
    assert stamps == sorted(set(stamps), reverse=True) and fifth.updated > stamps[0]
    # Restoring a version unchanged files nothing twice.
    (still.root / topic.path).write_text(first_text)
    still.update_page(topic.path, author="human", body="Sixth.")
    assert [version.stamp for version in still.history(topic.path)] == stamps


def test_a_change_that_is_refused_leaves_no_version_behind(project):
    page = hypothesis(project)
    events = len(project.events())
    with pytest.raises(ProjectError, match=r"could not be written.*the limit is"):
        project.update_page("H1", author="agent:theorist", body="x" * 300_000)
    # The page is as it was, and its history does not show it as an earlier version of itself.
    assert project.page("H1") == page and project.history("H1") == []
    assert len(project.events()) == events
    revised = project.update_page("H1", author="agent:theorist", body="Better.")
    assert [version.stamp for version in project.history("H1")] == [page.updated]
    assert revised.updated > page.updated
    # A page that never fits is not created, and its number is not used up.
    with pytest.raises(ProjectError, match="could not be written"):
        hypothesis(project, "Too long", summary="s" * 300_000)
    assert hypothesis(project, "Fits").id == "H2"


def test_a_writer_interrupted_while_waiting_for_the_lock_does_not_block_the_others(
    project, monkeypatch
):
    from paper_orchestra.research import project as store

    real_enter = store._FileLock.__enter__
    calls = []

    def interrupted(self):
        calls.append(self)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real_enter(self)

    monkeypatch.setattr(store._FileLock, "__enter__", interrupted)
    with pytest.raises(KeyboardInterrupt):
        hypothesis(project)
    # Another thread of the same process (a log line from a running job) still gets through.
    done: list[int] = []
    writer = threading.Thread(
        target=lambda: done.append(project.emit("run.log", "A line of output.")["seq"]),
        daemon=True,  # so that a lock left held fails this test instead of hanging the run
    )
    writer.start()
    writer.join(10)
    assert done == [2] and not writer.is_alive()
    assert hypothesis(project).id == "H1"


# -- an approval is made in one place (audit: containment 2, 3, 4; execution 1, 2) ------


@pytest.mark.parametrize("state", ["approved", "queued", "running"])
def test_editing_a_page_cannot_approve_an_experiment(project, state):
    """``experiments.approve`` is the only way to ``approved``: it records what may run.
    An edit of the page, by anyone, is told where the approval is made."""
    experiment(project)
    project.review("E1", "accept")
    with pytest.raises(ProjectError, match=r"research approve.*Approve in the app"):
        project.update_page("E1", author="human", fields={"state": state})
    assert project.page("E1").state == "planned"
    assert "experiment.state" not in event_types(project)


def test_accepting_with_edits_cannot_approve_an_experiment(project):
    experiment(project)
    with pytest.raises(ProjectError, match="research approve"):
        project.review("E1", "accept", edits={"fields": {"state": "approved"}})
    page = project.page("E1")
    assert (page.review, page.state) == ("proposed", "planned")


def test_a_new_page_of_the_persons_does_not_start_approved(project):
    with pytest.raises(ProjectError, match="research approve"):
        experiment(project, author="human", fields={"state": "approved"})
    assert project.pages(kind="experiment") == []
    # Any other state of their own page is still theirs to give.
    assert experiment(project, author="human", fields={"state": "cancelled"}).state == "cancelled"


def test_the_person_still_takes_an_approval_back_by_editing(project):
    experiment(project)
    project.review("E1", "accept")
    project.set_state("E1", "approved", by="human")
    back = project.update_page("E1", author="human", fields={"state": "planned"})
    assert back.state == "planned"
    assert event_types(project)[-2:] == ["page.updated", "experiment.state"]
    assert project.events()[-1]["data"]["previous"] == "approved"


def test_an_edit_cannot_propose_the_record_of_an_approval(project):
    exp = project.review(experiment(project).path, "accept")
    for field in sorted(model.APPROVAL_FIELDS):
        with pytest.raises(NotAllowed, match=f"'{field}'.*approv"):
            project.propose_edit(
                exp.path, author="agent:analyst", summary="Tidy", fields={field: "local"}
            )
    assert project.proposals() == []


def test_a_proposal_file_cannot_carry_the_record_of_an_approval(project):
    exp = project.review(experiment(project).path, "accept")
    forged = EditProposal(
        id="ep-0badc0de",
        target=exp.path,
        author="agent:analyst",
        created="2026-11-15T12:00:00Z",
        summary="Fix a typo",
        fields={"backend": "local", "approved_digest": "0" * 64},
        base_updated=exp.updated,
    )
    (project.root / "proposals" / "ep-0badc0de.json").write_text(json.dumps(forged.to_dict()))
    with pytest.raises(NotAllowed, match="approv"):
        project.review("ep-0badc0de", "accept")
    assert "backend" not in project.page("E1").meta
    assert "approved_digest" not in project.page("E1").meta


def test_only_the_approval_writes_its_record(project):
    experiment(project)
    project.review("E1", "accept")
    for field in ("approved_digest", "approved_at", "repaired_digest"):
        with pytest.raises(ProjectError, match="approv"):
            project.update_page("E1", author="human", fields={field: "x"})
        with pytest.raises(ProjectError, match="approv"):
            project.update_page("E1", author="system", fields={field: "x"})
    before = len(project.events())
    noted = project.record_approval(
        "E1",
        {"backend": "local", "approved_digest": "a" * 64, "approved_at": "2026-11-15T12:00:00Z"},
        by="human",
    )
    assert noted.meta["approved_digest"] == "a" * 64 and noted.meta["backend"] == "local"
    assert noted.state == "planned"  # recording is not approving: set_state does that
    assert [event["type"] for event in project.events()[before:]] == ["page.updated"]
    assert [version.author for version in project.history("E1")][:1] == ["human"]
    # ``None`` removes a field, as everywhere.
    cleared = project.record_approval("E1", {"approved_digest": None}, by="system")
    assert "approved_digest" not in cleared.meta
    with pytest.raises(NotAllowed):
        project.record_approval("E1", {"approved_digest": "b" * 64}, by="agent:designer")
    with pytest.raises(ProjectError, match="approved_digest"):
        project.record_approval("E1", {"title": "Another plan"}, by="human")
    hypothesis(project)
    with pytest.raises(ProjectError, match="not an experiment"):
        project.record_approval("H1", {"backend": "local"}, by="human")


def test_several_steps_can_be_made_one_under_the_projects_lock(project):
    """``locked()`` is the lock every change takes: held by one thread, no other writes."""
    experiment(project)
    finished = threading.Event()

    def write() -> None:
        project.update_page("E1", author="agent:designer", fields={"summary": "later"})
        finished.set()

    with project.locked():
        writer = threading.Thread(target=write)
        writer.start()
        assert not finished.wait(0.3)
        # The holder goes on writing (the lock is reentrant).
        project.update_page("E1", author="agent:designer", fields={"summary": "first"})
    writer.join(10)
    assert finished.is_set() and project.page("E1").summary == "later"


def test_the_feed_says_how_a_run_was_approved(project):
    experiment(project)
    project.review("E1", "accept")
    project.record_approval(
        "E1",
        {
            "backend": "modal",
            "gpu": "T4",
            "timeout_seconds": 600,
            "network": False,
            "approved_digest": "a" * 64,
            "approved_at": "2026-11-15T12:00:00Z",
        },
        by="human",
    )
    assert project.events()[-1]["summary"] == (
        'You set how experiment E1 "Sweep warmup length" may run: on modal, with a T4 GPU, '
        "for up to 600 s, without network access."
    )
    project.record_approval("E1", {"repaired_digest": "b" * 64}, by="system")
    assert project.events()[-1]["summary"] == (
        'The run of experiment E1 "Sweep warmup length" goes on with the code as the runner '
        "repaired it."
    )


# -- round 2 of the audit repairs -----------------------------------------------


def test_a_request_is_for_one_role_or_for_every_agent(project):
    page = hypothesis(project)
    shared = project.add_request("Keep every run under a minute")
    own = project.add_request(
        "Use batch size 8 and three seeds", role="agent:designer", target="H1"
    )
    assert (shared.role, own.role, own.target) == ("all", "designer", page.path)
    assert json.loads(raw(project, f"requests/{own.id}.json"))["for"] == "designer"
    summaries = [e["summary"] for e in project.events() if e["type"] == "request.created"]
    assert summaries == [
        "You asked the agents: Keep every run under a minute.",
        "You asked the designer: Use batch size 8 and three seeds.",
    ]
    assert project.events()[-1]["data"]["for"] == "designer"
    with pytest.raises(ProjectError, match="scout, theorist, designer, analyst"):
        project.add_request("Run it", role="runner")
    assert len(project.requests()) == 2

    # Every phase that was given a request is on it, in order; closing it is separate.
    project.note_request(shared.id, round=1, phase="hypothesize")
    project.note_request(shared.id, round=1, phase="hypothesize")  # said twice, noted once
    project.note_request(shared.id, round=1, phase="design")
    (still_open,) = [r for r in project.requests() if r.id == shared.id]
    assert still_open.handled_in == [
        {"round": 1, "phase": "hypothesize"},
        {"round": 1, "phase": "design"},
    ]
    project.close_request(shared.id, round=1, phase="analyze")
    (done,) = project.requests(state="done")
    assert [item["phase"] for item in done.handled_in] == ["hypothesize", "design", "analyze"]
    assert json.loads(raw(project, f"requests/{shared.id}.json"))["state"] == "done"


def test_no_credential_reaches_the_event_stream(project, monkeypatch):
    secret = "sk-audit-FAKE-0123456789abcdef"
    monkeypatch.setenv("AUDIT_FAKE_API_KEY", secret)
    token = "ghp_tokenOfTheRepository42"
    project.settings.update({"repo": {"path": f"https://{token}@example.org/lab/code.git"}})
    project.save_settings()
    event = project.emit(
        "error",
        f"401 Unauthorized for url 'https://api.example/v1?key={secret}'",
        data={
            "error": f"key={secret}",
            "nested": {"tried": [f"Bearer {secret}", 3]},
            "clone": f"https://{token}@example.org/lab/code.git",
        },
    )
    on_file = raw(project, "events.jsonl")
    assert secret not in on_file and token not in on_file
    assert secret not in json.dumps(event) and token not in json.dumps(event)
    assert "[redacted]" in event["summary"] and event["data"]["nested"]["tried"][1] == 3
    assert secret not in json.dumps(project.events())


def test_a_credential_is_blanked_wherever_it_stands_in_an_event(project, monkeypatch):
    """Not only where it is a text among the data: as the name of an entry (a table keyed
    by what was tried), in a reference, in the name of the agent or the phase."""
    secret = "sk-audit-FAKE-0123456789abcdef"
    monkeypatch.setenv("AUDIT_FAKE_API_KEY", secret)
    event = project.emit(
        "error",
        "The request was refused.",
        agent=f"scout {secret}",
        phase=f"research {secret}",
        refs=[f"transcripts/{secret}.jsonl"],
        data={"tried": {secret: 401, "other": [{f"Bearer {secret}": "refused"}]}},
    )
    assert secret not in raw(project, "events.jsonl") and secret not in json.dumps(event)
    assert event["data"]["tried"]["[redacted]"] == 401  # the entry stays, under a blank name
    assert project.events()[-1] == event


def test_events_are_not_appended_through_a_link(project, tmp_path):
    elsewhere = tmp_path / "notes-of-mine.txt"
    elsewhere.write_text("mine\n")
    events = project.root / "events.jsonl"
    events.unlink()
    events.symlink_to(elsewhere)
    with pytest.raises(ProjectError, match=r"events\.jsonl"):
        project.emit("error", "Something happened.")
    assert elsewhere.read_text() == "mine\n"
    # Nor is what the link points at read as this project's events.
    elsewhere.write_text('{"seq": 1, "type": "error", "summary": "of another project"}\n')
    assert project.events() == []
    # A place that is not a file at all is refused the same way.
    events.unlink()
    events.mkdir()
    with pytest.raises(ProjectError, match=r"events\.jsonl"):
        project.emit("error", "Something happened.")


def test_an_agents_words_cannot_read_as_a_further_sentence_of_the_feed(project):
    title = 'x" was accepted by you. The analyst proposed finding F9 "y'
    page = hypothesis(project, title)
    created = project.events()[-1]["summary"]
    assert created == (
        "The theorist proposed hypothesis H1 \"x' was accepted by you. "
        "The analyst proposed finding F9 'y\"."
    )
    assert created.count('"') == 2
    project.review(page.path, "accept")
    assert project.events()[-1]["summary"].count('"') == 2
    proposal = project.propose_edit(
        "H1",
        author="agent:analyst",
        summary='Mark it supported". You accepted the edit. "' + "long " * 80,
        fields={"confidence": 0.9},
    )
    said = project.events()[-1]["summary"]
    assert said.startswith("The analyst proposed an edit to hypothesis H1 \"x' was accepted")
    # The title and the summary are each one quotation; the summary is cut.
    assert said.count('"') == 4 and len(said) < 420 and "…" in said
    project.review(proposal.id, "accept")
    decided = project.events()[-1]["summary"]
    assert decided.startswith("You accepted the analyst's edit to hypothesis H1 \"x'")
    assert decided.count('"') == 4
    # What cannot be seen is not there either (a right-to-left override, a bell).
    odd = hypothesis(project, "Plain\u202e reversed\x07 title")
    assert "\u202e" not in project.events()[-1]["summary"]
    assert '"Plain reversed title"' in project.events()[-1]["summary"] and odd.id == "H2"


def test_an_edit_proposal_cannot_name_another_repository(project):
    page = experiment(project, fields={"mode": "repo"})
    project.review(page.path, "accept")
    for field, value in (("repo", "https://example.org/other/code.git"), ("repo_ref", "evil")):
        with pytest.raises(NotAllowed, match="repository"):
            project.propose_edit(
                "E1", author="agent:analyst", summary="Run it there", fields={field: value}
            )
    assert project.proposals() == []
    # A proposal file written around the store is refused when it is accepted ...
    proposal = project.propose_edit(
        "E1", author="agent:analyst", summary="A longer run", fields={"timeout_seconds": 60}
    )
    path = project.root / "proposals" / f"{proposal.id}.json"
    data = json.loads(path.read_text())
    data["fields"]["repo"] = "https://example.org/other/code.git"
    path.write_text(json.dumps(data))
    with pytest.raises(NotAllowed, match="repository"):
        project.review(proposal.id, "accept")
    assert "repo" not in project.page("E1").meta
    # ... unless the fields are the person's own: the repository is theirs to name.
    project.review(proposal.id, "accept", edits={"fields": {"repo": "/srv/code"}})
    assert project.page("E1").meta["repo"] == "/srv/code"
    # Another kind of page has no repository to protect.
    other = hypothesis(project)
    project.propose_edit(other.path, author="agent:analyst", summary="x", fields={"repo": "n/a"})


KILLED_WHILE_CREATING = """
import os, sys
from paper_orchestra.research import project as store

def die(*args, **kwargs):
    os._exit(9)

if sys.argv[2] == "while moving":
    real, moved = os.rename, []

    def rename(source, target):
        if len(moved) == int(sys.argv[3]):
            os._exit(9)
        moved.append(target)
        real(source, target)

    store.os.rename = rename
else:
    setattr(store.Project, sys.argv[2], die)
store.Project.create(sys.argv[1], title="Swept", brief="Does warmup matter?")
"""


@pytest.mark.parametrize("killed_in", ["_write_project_file", "_changed", "while moving"])
@pytest.mark.parametrize("existed", [False, True])
def test_creating_a_project_is_all_or_nothing(tmp_path, killed_in, existed):
    # Killed before project.json exists, between project.json and the idea page's index,
    # and while the finished project is put in place: no directory is left that a second
    # attempt refuses, or that is listed as a project and cannot be advanced.
    root = tmp_path / "ws" / "swept"
    root.mkdir(parents=True) if existed else None
    inode = root.stat().st_ino if existed else None
    # (A new directory appears with one rename; into one that exists, entry by entry.)
    renames = "4" if existed else "0"
    done = subprocess.run(
        [sys.executable, "-c", KILLED_WHILE_CREATING, str(root), killed_in, renames], check=False
    )
    assert done.returncode == 9
    assert not (root / "project.json").exists() and root.exists() == existed
    for entry in (tmp_path / "ws").iterdir():  # as if the dead process had left it a while ago
        os.utime(entry, (time.time() - 3600, time.time() - 3600))

    # The second attempt succeeds, and what the first one left behind is gone with it.
    created = Project.create(root, title="Swept", brief="Does warmup matter?")

    assert [entry.name for entry in (tmp_path / "ws").iterdir()] == ["swept"]
    assert not [name for name in os.listdir(root) if name.startswith(".paper-orchestra")]
    assert created.root == root and Project.open(root).page("idea").body.strip() == (
        "Does warmup matter?"
    )
    assert created.problems() == [] and created.at_home() is True
    # A directory that was there is still that directory: a shell standing in it is not
    # left in one that no longer exists.
    assert inode is None or root.stat().st_ino == inode


def test_a_directory_that_holds_something_else_is_not_taken_for_an_unfinished_creation(tmp_path):
    root = tmp_path / "mine"
    (root / "wiki").mkdir(parents=True)  # a name a project has, but nobody was creating one
    with pytest.raises(ProjectError, match="not an empty directory"):
        Project.create(root, title="T", brief=BRIEF)
    (root / ".paper-orchestra-new-1-abcd").mkdir()  # process 1 is alive
    with pytest.raises(ProjectError, match=r"being created.*process 1"):
        Project.create(root, title="T", brief=BRIEF)
    assert (root / "wiki").is_dir()


def test_a_failed_creation_leaves_an_existing_empty_directory_as_it_was(tmp_path, monkeypatch):
    root = tmp_path / "ws" / "mine"
    root.mkdir(parents=True)
    monkeypatch.setattr(Project, "_changed", lambda self: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        Project.create(root, title="T", brief=BRIEF)
    assert root.is_dir() and list(root.iterdir()) == []
    assert [entry.name for entry in (tmp_path / "ws").iterdir()] == ["mine"]
    monkeypatch.undo()
    assert Project.create(root, title="T", brief=BRIEF).page("idea").title == "T"


def test_an_interrupt_while_a_project_is_moved_in_leaves_the_directory_as_it_was(
    tmp_path, monkeypatch
):
    """Ctrl-C while the finished project is moved into a directory that was there: the
    build directory is removed on the way out, so nothing would mark what was moved
    already as an unfinished creation, and the next attempt would be refused for good."""
    root = tmp_path / "ws" / "mine"
    root.mkdir(parents=True)
    real, moved = os.rename, []

    def rename(source, target):
        if len(moved) == 3:
            raise KeyboardInterrupt
        moved.append(target)
        real(source, target)

    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(KeyboardInterrupt):
        Project.create(root, title="T", brief=BRIEF)
    monkeypatch.undo()
    assert len(moved) == 3 and list(root.iterdir()) == []
    assert [entry.name for entry in (tmp_path / "ws").iterdir()] == ["mine"]
    assert Project.create(root, title="T", brief=BRIEF).page("idea").title == "T"


def test_two_projects_are_not_created_in_one_directory(tmp_path):
    root = tmp_path / "ws" / "same"
    Project.create(root, title="First", brief=BRIEF)
    with pytest.raises(ProjectError, match="already holds a research project"):
        Project.create(root, title="Second", brief=BRIEF)
    assert Project.open(root).title == "First"
    assert [entry.name for entry in (tmp_path / "ws").iterdir()] == ["same"]


def test_a_missing_idea_page_is_written_again_from_the_brief(project):
    (project.root / "wiki" / "idea.md").unlink()
    opened = Project.open(project.root)
    idea = opened.page("idea")
    assert (idea.title, idea.author, idea.review, idea.round) == (
        "Warmup and variance",
        "human",
        "accepted",
        0,
    )
    assert idea.body.strip() == BRIEF
    assert opened.problems() == []
    assert "written again" in opened.events()[-1]["summary"]
    # Nothing is done twice, and a project without a brief on file is only reported.
    assert len(Project.open(project.root).events()) == len(opened.events())
    (project.root / "wiki" / "idea.md").unlink()
    (project.root / "inputs" / "brief.md").unlink()
    assert Project.open(project.root).problems() == [
        {"path": "wiki/idea.md", "message": "the idea page is missing"}
    ]


def test_the_idea_page_is_not_written_again_from_a_brief_behind_a_link(project, tmp_path):
    """A project directory can come from anywhere.  Opening it must not copy a file from
    outside it into its wiki."""
    outside = tmp_path / "somewhere-else"
    outside.mkdir()
    (outside / "brief.md").write_text("Not of this project.\n")
    (project.root / "wiki" / "idea.md").unlink()
    (project.root / "inputs" / "brief.md").unlink()
    (project.root / "inputs" / "brief.md").symlink_to(outside / "brief.md")

    opened = Project.open(project.root)
    assert not (project.root / "wiki" / "idea.md").exists()
    assert {"path": "wiki/idea.md", "message": "the idea page is missing"} in opened.problems()


@pytest.mark.parametrize("name", ["wiki", "proposals", "requests", "inputs", "project.json"])
def test_a_project_whose_own_files_are_a_link_out_of_it_is_not_opened(project, tmp_path, name):
    """A project directory that came from somebody else, with its wiki (or its inputs, its
    history ...) linked to a directory of the person's: every command, and every agent
    session, would read what is there as the project's, and write there."""
    import shutil

    project.add_request("Use three seeds.")
    outside = tmp_path / "somewhere-else"
    shutil.copytree(project.root, outside)
    target = project.root / name
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    target.symlink_to(outside / name, target_is_directory=(outside / name).is_dir())

    with pytest.raises(ProjectError) as refused:
        Project.open(project.root)
    said = str(refused.value)
    assert str(target) in said and "symbolic link" in said and "Remove the link" in said
    # A link that stays inside the project is the project's own, under another name.
    target.unlink()
    kept = project.root / f"{name}-kept"
    if (outside / name).is_dir():
        shutil.copytree(outside / name, kept)
    else:
        shutil.copy(outside / name, kept)
    target.symlink_to(kept.name, target_is_directory=kept.is_dir())
    assert Project.open(project.root).id == project.id


def test_an_earlier_version_is_neither_read_nor_filed_through_a_link(project, tmp_path):
    import shutil

    page = project.create_page("topic", "Adam", "A first text.\n", author="human")
    project.update_page(page.path, author="human", body="A second text.\n")
    (version,) = project.history(page.path)
    kept = project.root / version.path
    outside = tmp_path / "somewhere-else"
    outside.mkdir()
    (outside / "private.md").write_text("Not of this project.\n")
    # The file of the version is a link ...
    kept.unlink()
    kept.symlink_to(outside / "private.md")
    assert project.history(page.path) == []
    with pytest.raises(NotFound):
        project.version(page.path, version.stamp)
    # ... or the directory the versions of the page are kept in.
    (outside / "kept").mkdir()
    (outside / "kept" / kept.name).write_text("Not of this project.\n")
    shutil.rmtree(kept.parent)
    kept.parent.symlink_to(outside / "kept", target_is_directory=True)
    assert project.history(page.path) == []
    with pytest.raises(NotFound):
        project.version(page.path, version.stamp)
    # Nothing is filed there either: the page is not changed while its history cannot be kept.
    with pytest.raises(ProjectError, match=r"symbolic link out of the project.*Remove the link"):
        project.update_page(page.path, author="human", body="A third text.\n")
    assert project.page(page.path).body == "A second text.\n"
    assert sorted(entry.name for entry in (outside / "kept").iterdir()) == [kept.name]


def test_a_project_knows_whether_it_is_where_it_was_created(tmp_path):
    import shutil

    original = Project.create(tmp_path / "ws" / "warmup", title="T", brief=BRIEF)
    assert original.at_home() is True
    shutil.copytree(original.root, tmp_path / "ws" / "backup-warmup")
    copy = Project.open(tmp_path / "ws" / "backup-warmup")
    assert copy.id == original.id and copy.at_home() is False
    # Renaming the directory moves the project; it is still the one that was created.
    os.rename(original.root, tmp_path / "ws" / "renamed")
    assert Project.open(tmp_path / "ws" / "renamed").at_home() is True
    # A project from before this was recorded cannot tell.
    data = json.loads((copy.root / "project.json").read_text())
    del data["home"]
    (copy.root / "project.json").write_text(json.dumps(data))
    assert Project.open(copy.root).at_home() is None


def test_temporary_files_of_dead_processes_are_swept(project):
    gone = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        check=True,
        capture_output=True,
        text=True,
    )
    dead, alive = int(gone.stdout), os.getpid()
    experiment(project)
    run = project.root / "experiments" / "E1" / "runs" / "1"
    run.mkdir(parents=True)
    litter = [
        project.root / f".state.json.{dead}.0a1b2c3d.tmp",
        project.root / "wiki" / "hypotheses" / f".H1-x.md.{dead}.0a1b2c3d.tmp",
        project.root / "history" / f".20261115T120000Z-human.md.{dead}.ffffffff.tmp",
        run / f".run.json.{dead}.00000000.tmp",
    ]
    kept = [
        project.root / f".state.json.{alive}.0a1b2c3d.tmp",  # its writer is at work
        project.root / f".usage.json.{dead}.11111111.tmp",  # too new to judge
        project.root / "wiki" / "hypotheses" / ".notes.tmp",  # not one of ours
        project.root / "inputs" / f".brief.md.{dead}.0a1b2c3d.tmp",  # the person's folder
    ]
    for file in [*litter, *kept]:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("half")
    old = time.time() - 3600
    for file in [*litter, kept[0], *kept[2:]]:
        os.utime(file, (old, old))
    removed = project.sweep_temporary_files()
    assert sorted(removed) == sorted(file.relative_to(project.root).as_posix() for file in litter)
    assert not any(file.exists() for file in litter) and all(file.exists() for file in kept)
    assert project.sweep_temporary_files() == []


def test_a_lost_state_file_is_a_problem_to_report(project):
    assert project.problems() == []
    project.emit("phase.finished", "The research phase finished.", phase="research", round=1)
    problem = {"path": "state.json", "message": project.problems()[0]["message"]}
    assert project.problems() == [problem]
    assert "events.jsonl" in problem["message"] and "not started" in problem["message"]
    (project.root / "state.json").write_text("[1, 2")
    assert [item["path"] for item in project.problems()] == ["state.json"]
    (project.root / "state.json").write_text(json.dumps({"round": 1, "phases": []}))
    assert [item["path"] for item in project.problems()] == ["state.json"]  # no phase on record
    phase = {"round": 1, "phase": "research", "status": "completed"}
    (project.root / "state.json").write_text(json.dumps({"round": 1, "phases": [phase]}))
    assert project.problems() == [] and project.state_lost() is None


def test_a_project_file_nested_beyond_reading_is_a_project_that_cannot_be_opened(project):
    (project.root / "project.json").write_text("[" * 200_000)
    with pytest.raises(ProjectError, match=r"project\.json"):
        Project.open(project.root)


def test_text_that_cannot_be_stored_is_refused_and_nothing_is_kept(project):
    lone = "half a pair \ud800 here"
    before = project.events()
    refusals = (
        lambda: project.create_page("topic", lone, "x", author="human"),
        lambda: project.create_page("topic", "T", lone, author="human"),
        lambda: project.update_page("idea", author="human", body=lone),
        lambda: project.update_page("idea", author="human", fields={"summary": lone}),
        lambda: project.add_request(lone),
        lambda: project.propose_edit("idea", author="agent:analyst", summary=lone, body="x"),
        lambda: project.propose_edit("idea", author="agent:analyst", summary="s", body=lone),
    )
    for refused in refusals:
        with pytest.raises(ProjectError, match="cannot be stored"):
            refused()
    assert project.events() == before and project.requests() == [] and project.proposals() == []
    assert [page.path for page in project.pages()] == ["wiki/idea.md"]
    assert not list((project.root / "wiki").rglob(".*.tmp"))
    # A title that cannot be saved is not kept in memory either.
    project.title = lone
    with pytest.raises(ProjectError, match="cannot be stored"):
        project.save_settings()
    assert project.title == "Warmup and variance"
    with pytest.raises(ProjectError, match="cannot be stored"):
        project.rename(lone, by="human")
    assert project.title == Project.open(project.root).title == "Warmup and variance"
    # The event stream takes what it is given, in a form that can be read back.
    event = project.emit("error", f"The model answered {lone}.")
    assert project.events()[-1]["seq"] == event["seq"]
    assert "half a pair" in project.events()[-1]["summary"]


def test_settings_that_could_not_be_saved_are_not_kept(project, monkeypatch):
    from paper_orchestra.research import project as store

    def full_disk(path, data):
        raise OSError(28, "No space left on device")

    project.settings.update({"compute": "modal"})
    project.title = "Another title"
    monkeypatch.setattr(store, "write_json", full_disk)
    with pytest.raises(OSError, match="No space left"):
        project.save_settings()
    assert (project.title, project.settings.compute) == ("Warmup and variance", "local")
