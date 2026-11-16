from __future__ import annotations

import json

import pytest

from paper_orchestra.research import model
from paper_orchestra.research.model import (
    EditProposal,
    NotAllowed,
    ProjectError,
    ProjectSettings,
    Request,
)

SPEC_DEFAULT_SETTINGS = {
    "models": {
        "research": "claude-fable-5-2",
        "reasoning": "claude-fable-5-2",
        "coding": "claude-fable-5-2",
        "fast": "claude-fable-5-2",
        "effort": "high",
    },
    "research_cutoff": None,
    "compute": "local",
    "review": {
        "source": "auto",
        "topic": "review",
        "question": "review",
        "hypothesis": "review",
        "experiment": "review",
        "finding": "review",
        "journal": "auto",
        "lesson": "auto",
    },
    "limits": {
        "model_usd": 25.0,
        "experiment_seconds": 900,
        "experiment_usd": 5.0,
        "repair_rounds": 3,
        "max_sources": 40,
        "max_hypotheses": 6,
        "max_experiments": 6,
    },
    "lab_memory": True,
    # Section 10.1: the person's repository for experiments in repo mode.
    "repo": None,
}


# -- vocabulary ---------------------------------------------------------------


def test_every_kind_has_a_place_in_the_wiki():
    assert model.KINDS == (
        "idea",
        "question",
        "source",
        "topic",
        "hypothesis",
        "experiment",
        "result",
        "finding",
        "journal",
        "lesson",
    )
    assert model.page_path("idea") == "wiki/idea.md"
    assert model.page_path("question", id="Q3", slug="does-it-scale") == (
        "wiki/questions/Q3-does-it-scale.md"
    )
    assert model.page_path("hypothesis", id="H2", slug="warmup") == "wiki/hypotheses/H2-warmup.md"
    assert model.page_path("experiment", id="E1", slug="sweep") == "wiki/experiments/E1-sweep.md"
    assert model.page_path("finding", id="F4", slug="gap") == "wiki/findings/F4-gap.md"
    assert model.page_path("source", name="Smith2020Deep") == "wiki/sources/Smith2020Deep.md"
    assert model.page_path("result", name="E1-run1") == "wiki/results/E1-run1.md"
    assert model.page_path("topic", slug="warmup") == "wiki/topics/warmup.md"
    assert model.page_path("journal", slug="round-1") == "wiki/journal/round-1.md"
    assert model.page_path("lesson", slug="seeds") == "wiki/lessons/seeds.md"


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("wiki/idea.md", "idea"),
        ("wiki/hypotheses/H2-warmup.md", "hypothesis"),
        ("wiki/sources/Smith2020Deep.md", "source"),
        ("wiki/journal/round-1.md", "journal"),
        ("wiki/elsewhere/x.md", None),
        ("wiki/loose.md", None),
        ("notes/x.md", None),
    ],
)
def test_the_section_of_a_path_decides_its_kind(path, kind):
    assert model.kind_for_path(path) == kind


def test_ids_have_a_prefix_per_kind_and_are_read_from_names():
    assert model.ID_PREFIXES == {
        "question": "Q",
        "hypothesis": "H",
        "experiment": "E",
        "finding": "F",
    }
    assert model.format_id("hypothesis", 12) == "H12"
    assert model.parse_id("h2") == ("H", 2)
    assert model.parse_id("H2-warmup") is None
    assert model.id_in_name("hypothesis", "H2-warmup") == "H2"
    assert model.id_in_name("hypothesis", "h10") == "H10"
    assert model.id_in_name("hypothesis", "E2-warmup") is None
    assert model.id_in_name("topic", "H2-warmup") is None


def test_records_and_roles():
    assert model.RECORD_KINDS == ("source", "result")
    assert model.is_record("source") and model.is_record("result")
    assert not model.is_record("finding")
    assert model.ROLE_MAY_CREATE == {
        "intake": ("topic", "question"),
        "scout": ("topic", "question"),
        "theorist": ("hypothesis", "question"),
        "designer": ("experiment",),
        "analyst": ("finding", "question", "hypothesis", "experiment", "journal"),
    }
    assert model.may_create("theorist", "hypothesis")
    assert not model.may_create("theorist", "experiment")
    # No role writes a record, the idea or a lesson by hand; the runner writes no page at all.
    for role in model.ROLES:
        assert not model.may_create(role, "source")
        assert not model.may_create(role, "result")
        assert not model.may_create(role, "idea")
    assert model.creatable_kinds("runner") == ()
    assert not model.may_create("stranger", "topic")


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("human", "human"),
        ("system", "system"),
        ("agent:scout", "agent:scout"),
        ("scout", "agent:scout"),
        (" Human ", "human"),
    ],
)
def test_authors_are_normalised(given, expected):
    assert model.normalize_author(given) == expected


@pytest.mark.parametrize("bad", ["", "agent:", "bob", "agent:Not Valid", None])
def test_an_unknown_author_is_refused_with_the_accepted_forms(bad):
    with pytest.raises(ProjectError, match=r"human.*agent:<name>.*system"):
        model.normalize_author(bad)


def test_who_wrote_a_page_and_the_policy_decide_its_first_review():
    policy = {"hypothesis": "review", "journal": "auto"}
    assert model.initial_review("hypothesis", "human", policy) == "accepted"
    assert model.initial_review("hypothesis", "agent:theorist", policy) == "proposed"
    assert model.initial_review("journal", "agent:analyst", policy) == "accepted"
    assert model.initial_review("journal", "system", policy) == "accepted"
    # A kind the policy does not mention is reviewed: the safe side.
    assert model.initial_review("finding", "agent:analyst", policy) == "proposed"


# -- lifecycle ----------------------------------------------------------------


def test_states_per_kind():
    assert model.STATES["question"] == ("open", "answered", "dropped")
    assert model.STATES["hypothesis"] == (
        "open",
        "testing",
        "supported",
        "refuted",
        "inconclusive",
        "dropped",
    )
    assert model.STATES["experiment"] == (
        "planned",
        "approved",
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    )
    assert model.STATES["result"] == ("succeeded", "failed", "cancelled")
    assert model.INITIAL_STATE == {
        "question": "open",
        "hypothesis": "open",
        "experiment": "planned",
    }


@pytest.mark.parametrize("current", model.STATES["experiment"])
@pytest.mark.parametrize("by", ["system", "agent:runner", "agent:designer", "agent:analyst"])
def test_only_a_person_approves_an_experiment(current, by):
    if current == "approved":
        return
    with pytest.raises(ProjectError) as caught:
        model.check_transition("experiment", current, "approved", by)
    if current in ("planned", "succeeded", "failed", "cancelled"):
        assert isinstance(caught.value, NotAllowed)
        assert "person" in str(caught.value)


def test_the_person_approves_and_code_runs():
    model.check_transition("experiment", "planned", "approved", "human")
    model.check_transition("experiment", "approved", "planned", "human")
    for step in (("approved", "queued"), ("queued", "running"), ("running", "succeeded")):
        model.check_transition("experiment", *step, "system")
        model.check_transition("experiment", *step, "agent:runner")
        with pytest.raises(NotAllowed):
            model.check_transition("experiment", *step, "agent:designer")
    model.check_transition("experiment", "approved", "running", "system")
    model.check_transition("experiment", "running", "failed", "system")
    # An interrupted run goes back to the queue without a new approval.
    model.check_transition("experiment", "running", "queued", "system")
    # Anyone in charge can cancel; running again takes a new approval by the person.
    model.check_transition("experiment", "running", "cancelled", "human")
    model.check_transition("experiment", "running", "cancelled", "system")
    model.check_transition("experiment", "failed", "approved", "human")
    with pytest.raises(NotAllowed):
        model.check_transition("experiment", "planned", "cancelled", "system")


def test_illegal_moves_and_unknown_states_say_what_is_possible():
    with pytest.raises(ProjectError, match=r"planned.*cannot go straight to.*succeeded"):
        model.check_transition("experiment", "planned", "succeeded", "system")
    with pytest.raises(ProjectError, match=r"'done' is not a state of an experiment.*planned"):
        model.check_transition("experiment", "planned", "done", "human")
    with pytest.raises(ProjectError, match="no lifecycle"):
        model.check_transition("topic", None, "open", "human")
    with pytest.raises(ProjectError, match="record"):
        model.check_transition("result", "succeeded", "failed", "system")


def test_verdicts_on_hypotheses_and_questions_are_the_persons():
    model.check_transition("hypothesis", "open", "testing", "system")
    model.check_transition("hypothesis", "testing", "open", "system")
    model.check_transition("hypothesis", "testing", "supported", "human")
    model.check_transition("hypothesis", "supported", "open", "human")
    model.check_transition("question", "open", "answered", "human")
    for by in ("system", "agent:analyst"):
        with pytest.raises(NotAllowed, match="propose"):
            model.check_transition("hypothesis", "testing", "supported", by)
        with pytest.raises(NotAllowed):
            model.check_transition("question", "open", "answered", by)
    with pytest.raises(NotAllowed):
        model.check_transition("hypothesis", "open", "testing", "agent:theorist")
    # A page written by hand may lack its state: it counts as the kind's first one.
    model.check_transition("hypothesis", None, "testing", "system")


# -- relations ----------------------------------------------------------------


def test_relation_fields_know_where_they_start_and_end():
    assert tuple(model.RELATIONS) == (
        "about",
        "cites",
        "answers",
        "motivated_by",
        "tests",
        "run_of",
        "supports",
        "refutes",
        "based_on",
        "follows",
    )
    assert model.RELATIONS["tests"].sources == ("experiment",)
    assert model.RELATIONS["tests"].targets == ("hypothesis",)
    assert model.RELATIONS["cites"].sources == ()  # any kind
    assert model.RELATIONS["cites"].targets == ("source",)
    assert model.RELATIONS["motivated_by"].targets == ("source", "topic", "question", "finding")
    assert model.RELATIONS["follows"].sources == ("hypothesis", "experiment", "question")
    assert model.relations_from("experiment") == ("cites", "tests", "follows")
    assert model.relations_from("finding") == (
        "cites",
        "answers",
        "supports",
        "refutes",
        "based_on",
    )


def test_relation_targets_are_read_however_they_were_written():
    cleaned = model.clean_relations(
        "finding",
        {
            "supports": "h2",
            "cites": "[Smith2020Deep, Lee2021Wide]",
            "based_on": ["[[results/E1-run1]]", " results/E1-run2.md ", ""],
            "refutes": "",
        },
    )
    assert cleaned == {
        "supports": ["H2"],
        "cites": ["Smith2020Deep", "Lee2021Wide"],
        "based_on": ["results/E1-run1", "results/E1-run2"],
    }


def test_an_empty_relation_the_kind_does_not_have_is_not_an_error():
    # A form sends every relation field; the ones that do not apply are empty.
    everything = {name: [] for name in model.RELATIONS} | {"cites": ["Smith2020Deep"]}
    assert model.clean_relations("hypothesis", everything) == {"cites": ["Smith2020Deep"]}


def test_a_relation_on_the_wrong_kind_is_refused_with_the_ones_that_fit():
    with pytest.raises(ProjectError, match=r"'tests'.*hypothesis.*cites, motivated_by, follows"):
        model.clean_relations("hypothesis", {"tests": ["H1"]})
    with pytest.raises(ProjectError, match="'likes' is not a relation"):
        model.clean_relations("hypothesis", {"likes": ["H1"]})


# -- pages --------------------------------------------------------------------


def test_front_matter_becomes_a_page_and_back():
    text = (
        "---\n"
        "title: Warmup reduces variance\n"
        "kind: hypothesis\n"
        "id: H2\n"
        "summary: Linear warmup lowers the variance of early updates\n"
        "review: proposed\n"
        "state: open\n"
        'author: "agent:theorist"\n'
        "round: 1\n"
        'created: "2026-11-15T12:00:00Z"\n'
        'updated: "2026-11-15T12:00:05Z"\n'
        "motivated_by:\n"
        "- Smith2020Deep\n"
        "- topics/warmup\n"
        'cites: ["Smith2020Deep"]\n'
        "confidence: 0.6\n"
        "---\n"
        "## Statement\n\nWarmup helps, see [[topics/warmup]].\n"
    )
    page = model.parse_page("wiki/hypotheses/H2-warmup.md", text)
    assert (page.kind, page.id, page.title) == ("hypothesis", "H2", "Warmup reduces variance")
    assert (page.review, page.state, page.author, page.round) == (
        "proposed",
        "open",
        "agent:theorist",
        1,
    )
    assert (page.created, page.updated) == ("2026-11-15T12:00:00Z", "2026-11-15T12:00:05Z")
    assert page.relations == {
        "cites": ["Smith2020Deep"],
        "motivated_by": ["Smith2020Deep", "topics/warmup"],
    }
    assert page.meta["confidence"] == 0.6
    assert page.body.startswith("## Statement")
    assert page.label == "H2" and page.link == "hypotheses/H2-warmup"
    assert model.page_problems(page) == []

    again = model.parse_page(page.path, model.render_page(page))
    assert again == page
    # The fields a person looks for first come first.
    assert list(again.meta)[:5] == ["title", "kind", "id", "summary", "review"]
    as_json = json.loads(json.dumps(page.to_dict()))
    assert as_json["relations"]["cites"] == ["Smith2020Deep"]
    assert as_json["label"] == "H2"


def test_a_page_written_by_hand_still_loads_and_says_what_is_missing():
    page = model.parse_page(
        "wiki/hypotheses/my-hunch.md",
        "---\nkind: essay\nstate: maybe\ntests: H1\n---\n# A hunch\n\nIt might work.\n",
        file_stamp="2026-11-15T09:00:00Z",
    )
    # The section says what it is; a hand-written page is the person's, hence accepted.
    assert (page.kind, page.title, page.summary) == ("hypothesis", "A hunch", "It might work.")
    assert (page.author, page.review, page.id) == ("human", "accepted", None)
    assert page.state == "open"
    assert page.updated == "2026-11-15T09:00:00Z"
    problems = "\n".join(model.page_problems(page))
    assert "kind 'essay'" in problems
    assert "no id" in problems
    assert "'maybe' is not a state" in problems
    assert "'tests'" in problems
    assert "title" in problems


def test_an_updated_field_that_is_not_a_stamp_gives_way_to_the_time_of_the_file():
    text = "---\ntitle: Notes\nupdated: 15 November\n---\nText.\n"
    page = model.parse_page("wiki/topics/notes.md", text, file_stamp="2026-11-15T09:00:00Z")
    # The stamp names the version in the history, so it has to be one.
    assert page.updated == "2026-11-15T09:00:00Z" and page.meta["updated"] == "15 November"
    assert any("updated is '15 November'" in problem for problem in model.page_problems(page))
    # Without a file to ask, what was written is all there is.
    assert model.parse_page("wiki/topics/notes.md", text).updated == "15 November"
    kept = model.parse_page(
        "wiki/topics/notes.md",
        "---\ntitle: Notes\nupdated: 2026-11-14T08:00:00Z\n---\nText.\n",
        file_stamp="2026-11-15T09:00:00Z",
    )
    assert kept.updated == "2026-11-14T08:00:00Z"
    assert not any("updated" in problem for problem in model.page_problems(kept))


def test_a_page_has_moved_on_once_it_left_its_first_state():
    def page(path: str, state: str | None):
        front = f"---\ntitle: T\nstate: {state}\n---\nText.\n" if state else "Text.\n"
        return model.parse_page(path, front)

    assert not model.moved_on(page("wiki/experiments/E1-x.md", "planned"))
    assert not model.moved_on(page("wiki/experiments/E1-x.md", None))
    for state in ("approved", "queued", "running", "succeeded", "failed", "cancelled"):
        assert model.moved_on(page("wiki/experiments/E1-x.md", state))
    assert not model.moved_on(page("wiki/hypotheses/H1-x.md", "open"))
    assert model.moved_on(page("wiki/hypotheses/H1-x.md", "testing"))
    assert model.moved_on(page("wiki/questions/Q1-x.md", "answered"))
    # Kinds without a lifecycle never do.
    assert not model.moved_on(page("wiki/topics/x.md", "approved"))


def test_an_agent_page_without_a_valid_review_counts_as_proposed():
    page = model.parse_page(
        "wiki/topics/x.md", '---\ntitle: X\nauthor: "agent:scout"\nreview: sure\n---\nText.\n'
    )
    assert page.review == "proposed"
    assert any("review" in problem for problem in model.page_problems(page))
    loose = model.parse_page("wiki/elsewhere/x.md", "---\ntitle: X\nkind: essay\n---\nText.\n")
    assert loose.kind == "topic"
    assert any("essay" in problem for problem in model.page_problems(loose))


def test_a_source_is_known_by_its_key():
    page = model.parse_page(
        "wiki/sources/Smith2020Deep.md", "---\ntitle: Deep things\nkind: source\n---\nAbstract.\n"
    )
    assert page.id == "Smith2020Deep" and page.label == "Smith2020Deep"
    assert page.state is None


# -- settings -----------------------------------------------------------------


def test_settings_default_to_the_documented_values():
    assert ProjectSettings().to_dict() == SPEC_DEFAULT_SETTINGS
    assert json.dumps(ProjectSettings().to_dict()["limits"]["experiment_seconds"]) == "900"


def test_settings_round_trip_and_keep_what_they_do_not_know():
    raw = json.loads(json.dumps(SPEC_DEFAULT_SETTINGS))
    raw["compute"] = "modal"
    raw["research_cutoff"] = "2024-06"
    raw["limits"]["model_usd"] = 3
    raw["limits"]["max_rounds"] = 9  # written by a newer version
    raw["theme"] = "dark"
    settings = ProjectSettings.from_dict(raw)
    assert settings.load_problems == []
    assert settings.compute == "modal" and settings.limits.model_usd == 3
    assert settings.to_dict() == raw
    assert ProjectSettings.from_dict(settings.to_dict()).to_dict() == raw


def test_partial_settings_from_an_older_project_get_the_defaults():
    settings = ProjectSettings.from_dict({"models": {"research": "other-model"}, "review": {}})
    assert settings.models.research == "other-model"
    assert settings.models.coding == "claude-fable-5-2"
    assert settings.review["hypothesis"] == "review" and settings.review["journal"] == "auto"
    assert settings.limits.max_sources == 40
    assert ProjectSettings.from_dict(None).to_dict() == SPEC_DEFAULT_SETTINGS
    assert ProjectSettings.from_dict("nonsense").to_dict() == SPEC_DEFAULT_SETTINGS


def test_sections_read_like_objects_and_like_mappings():
    settings = ProjectSettings()
    assert settings.limits.model_usd == settings.limits["model_usd"] == 25.0
    assert settings.models["effort"] == settings.models.effort == "high"
    assert settings.limits.get("nope", 7) == 7
    # All of a mapping's read side, so code written against the JSON shape does not trip.
    limits = settings.limits
    assert dict(limits) == dict(limits.items()) == {**limits} == limits.to_dict()
    assert list(limits) == limits.keys() and len(limits) == 7 and "model_usd" in limits
    assert limits.values()[0] == 25.0
    assert [key for key in settings.models] == ["research", "reasoning", "coding", "fast", "effort"]
    assert settings.policy("hypothesis") == "review" and settings.policy("journal") == "auto"
    # Results and the idea are never held for review.
    assert settings.policy("result") == "auto"


def test_invalid_settings_load_with_defaults_and_say_so():
    settings = ProjectSettings.from_dict(
        {
            "compute": "mainframe",
            "limits": {"model_usd": "lots", "repair_rounds": -1, "max_sources": 12},
            "review": {"hypothesis": "maybe", "topic": "auto"},
            "models": {"effort": "extreme", "fast": ""},
            "lab_memory": "yes",
            "research_cutoff": "last year",
        }
    )
    assert settings.compute == "local"
    assert settings.limits.model_usd == 25.0 and settings.limits.repair_rounds == 3
    assert settings.limits.max_sources == 12
    assert settings.review["hypothesis"] == "review" and settings.review["topic"] == "auto"
    assert settings.models.effort == "high" and settings.models.fast == "claude-fable-5-2"
    assert settings.lab_memory is True and settings.research_cutoff is None
    report = "\n".join(settings.load_problems)
    for needle in (
        "compute",
        "'mainframe'",
        "local, modal",
        "limits.model_usd",
        "limits.repair_rounds",
        "review.hypothesis",
        "auto, review",
        "models.effort",
        "models.fast",
        "lab_memory",
        "research_cutoff",
        "YYYY-MM",
    ):
        assert needle in report
    assert settings.problems() == []


def test_whole_numbers_written_as_floats_are_read_as_whole_numbers():
    settings = ProjectSettings.from_dict({"limits": {"max_sources": 12.0, "repair_rounds": 2.5}})
    assert settings.limits.max_sources == 12 and isinstance(settings.limits.max_sources, int)
    assert settings.limits.repair_rounds == 3
    assert any("limits.repair_rounds" in problem for problem in settings.load_problems)


def test_strict_loading_and_validation_refuse_with_every_problem_named():
    with pytest.raises(ProjectError) as caught:
        ProjectSettings.from_dict(
            {"compute": "mainframe", "limits": {"model_usd": -1}}, strict=True
        )
    assert "compute" in str(caught.value) and "limits.model_usd" in str(caught.value)
    settings = ProjectSettings()
    settings.limits.max_hypotheses = 2.5
    settings.review["finding"] = "never"
    with pytest.raises(ProjectError) as caught:
        settings.validate()
    assert "limits.max_hypotheses" in str(caught.value) and "review.finding" in str(caught.value)


def test_update_merges_a_partial_change_or_changes_nothing():
    settings = ProjectSettings()
    settings.update(
        {"limits": {"model_usd": 50}, "review": {"hypothesis": "auto"}, "compute": "modal"}
    )
    assert settings.limits.model_usd == 50 and settings.limits.experiment_usd == 5.0
    assert settings.review["hypothesis"] == "auto" and settings.review["finding"] == "review"
    assert settings.compute == "modal"
    before = settings.to_dict()
    with pytest.raises(ProjectError, match=r"limits.model_usd"):
        settings.update({"limits": {"model_usd": "free"}, "compute": "local"})
    assert settings.to_dict() == before


def test_the_repository_for_repo_experiments_is_a_setting():
    assert ProjectSettings().repo is None
    settings = ProjectSettings.from_dict({"repo": {"path": " /home/me/code ", "ref": None}})
    assert settings.load_problems == []
    # It is the object of project.json, and reads as attributes like the other sections.
    assert settings.repo == {"path": "/home/me/code", "ref": None}
    assert settings.repo.path == settings.repo["path"] == "/home/me/code"
    assert settings.repo.ref is None and settings.repo.get("ref") is None
    assert json.loads(json.dumps(settings.to_dict()))["repo"] == {
        "path": "/home/me/code",
        "ref": None,
    }
    assert ProjectSettings.from_dict(settings.to_dict()).repo == settings.repo

    # A partial change merges into it; null takes it away.
    settings.update({"repo": {"ref": "main"}})
    assert settings.repo == {"path": "/home/me/code", "ref": "main"} and settings.repo.ref == "main"
    settings.update({"repo": None})
    assert settings.repo is None
    settings.update({"repo": {"path": "https://example.org/me/code.git"}})
    assert settings.repo == {"path": "https://example.org/me/code.git", "ref": None}
    # A bare path is taken as the path.
    assert ProjectSettings.from_dict({"repo": "/srv/code"}).repo == {
        "path": "/srv/code",
        "ref": None,
    }


@pytest.mark.parametrize("bad", [{"ref": "main"}, {"path": ""}, {"path": "/x", "ref": 3}, 7, ""])
def test_an_invalid_repository_is_reported_and_never_half_used(bad):
    loaded = ProjectSettings.from_dict({"repo": bad})
    assert loaded.repo is None
    assert any(problem.startswith("settings.repo ") for problem in loaded.load_problems)
    with pytest.raises(ProjectError, match=r"settings\.repo"):
        ProjectSettings.from_dict({"repo": bad}, strict=True)
    settings = ProjectSettings()
    with pytest.raises(ProjectError, match=r"settings\.repo"):
        settings.update({"repo": bad})
    assert settings.repo is None
    settings.repo = bad
    with pytest.raises(ProjectError, match=r"settings\.repo"):
        settings.validate()


# -- proposals and requests ---------------------------------------------------


def test_an_edit_proposal_has_the_documented_shape():
    proposal = EditProposal(
        id="ep-0a1b2c3d",
        target="wiki/hypotheses/H2-warmup.md",
        author="agent:analyst",
        round=2,
        created="2026-11-15T12:00:00Z",
        summary="Mark H2 as supported by F1",
        fields={"state": "supported"},
        relations_add={"cites": ["Smith2020Deep"]},
        base_updated="2026-11-15T11:00:00Z",
    )
    assert proposal.to_dict() == {
        "id": "ep-0a1b2c3d",
        "target": "wiki/hypotheses/H2-warmup.md",
        "author": "agent:analyst",
        "round": 2,
        "created": "2026-11-15T12:00:00Z",
        "summary": "Mark H2 as supported by F1",
        "body": None,
        "fields": {"state": "supported"},
        "relations_add": {"cites": ["Smith2020Deep"]},
        "base_updated": "2026-11-15T11:00:00Z",
        "review": "proposed",
        "decided_by": None,
        "decided_at": None,
        "decision_reason": None,
    }
    assert EditProposal.from_dict(proposal.to_dict()) == proposal
    partial = EditProposal.from_dict({"id": "ep-1", "target": "wiki/idea.md"})
    assert partial.review == "proposed" and partial.fields == {} and partial.body is None


def test_a_request_has_the_documented_shape():
    request = Request(
        id="rq-0a1b2c3d", text="Look at seeds", target=None, created="2026-11-15T12:00:00Z"
    )
    assert request.to_dict() == {
        "id": "rq-0a1b2c3d",
        "text": "Look at seeds",
        "target": None,
        "created": "2026-11-15T12:00:00Z",
        "state": "open",
        "handled_in": None,
    }
    done = Request.from_dict(
        {**request.to_dict(), "state": "done", "handled_in": {"round": 2, "phase": "hypothesize"}}
    )
    assert done.state == "done" and done.handled_in == {"round": 2, "phase": "hypothesize"}


def test_stamps():
    assert model.compact_stamp("2026-11-15T12:00:05Z") == "20261115T120005Z"
    assert model.format_stamp(model.parse_stamp("20261115T120005Z")) == "2026-11-15T12:00:05Z"
    assert model.parse_stamp("yesterday") is None and model.parse_stamp(None) is None
    assert (
        model.stamp_after("2026-11-15T12:00:05Z", "2026-11-15T12:00:05Z") == "2026-11-15T12:00:06Z"
    )
    assert (
        model.stamp_after("2026-11-15T12:00:09Z", "2026-11-15T12:00:05Z") == "2026-11-15T12:00:09Z"
    )
    assert model.stamp_after("2026-11-15T12:00:09Z", "") == "2026-11-15T12:00:09Z"


# -- what an approval pins (audit: containment 2, 3; execution 2) -----------------


def test_the_record_of_an_approval_is_not_an_agents_to_write():
    """Where a run goes and what was approved are written by the approval itself: an
    agent's page cannot carry them, whatever its text says."""
    assert sorted(model.APPROVAL_FIELDS) == [
        "approved_at",
        "approved_digest",
        "backend",
        "repaired_digest",
    ]
    assert model.APPROVAL_FIELDS <= model.AGENT_LOCKED_FIELDS
    # The fields a plan is made of are named once, for the digest and for every view.
    assert model.RUN_FIELDS == (
        "mode",
        "entrypoint",
        "command",
        "packages",
        "repo",
        "repo_ref",
        "gpu",
        "network",
        "backend",
        "timeout_seconds",
    )


def test_code_takes_an_approval_back_but_never_gives_one():
    """A run that finds the experiment changed since its approval puts it back to
    ``planned``; approving stays the person's alone."""
    for current in ("approved", "queued"):
        model.check_transition("experiment", current, "planned", "system")
        model.check_transition("experiment", current, "planned", "agent:runner")
        with pytest.raises(NotAllowed):
            model.check_transition("experiment", current, "planned", "agent:designer")
    with pytest.raises(NotAllowed):
        model.check_transition("experiment", "planned", "approved", "system")
