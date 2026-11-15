from __future__ import annotations

import json
import logging
import re
import threading

import pytest

from paper_orchestra.eval.agent_review import (
    AGENTREVIEW_INSTRUCTIONS,
    SCORE_CALCULATION,
    Review,
    ReviewError,
    generate_from_client,
    get_agentreview_system_prompt,
    get_meta_review,
    perform_review_agentreview,
)
from paper_orchestra.llm.base import Capabilities, LLMError, UsageTracker
from paper_orchestra.llm.fake import FakeLLM, FakeRequest

PAPER = "We propose sparse routing. It improves accuracy by 2.1 points on the benchmark."


def review_json(**overrides) -> str:
    review = {
        "Summary": "A paper about sparse routing.",
        "Strengths": ["clear idea"],
        "Weaknesses": ["thin ablations"],
        "Originality": 2,
        "Quality": 2,
        "Clarity": 2,
        "Significance": 2,
        "Questions": ["What is the latency?"],
        "Limitations": ["small scale"],
        "EthicalConcerns": False,
        "Soundness": 2,
        "Presentation": 2,
        "Contribution": 2,
        "Overall": 5,
        "Confidence": 4,
        "Decision": "Reject",
    }
    review.update(overrides)
    return json.dumps(review)


def reviewer_index(request: FakeRequest) -> int:
    return int(re.search(r"You are reviewer (\d+) of \d+", request.text).group(1))


def ensemble(members: dict[int, str], meta: str | None = None) -> FakeLLM:
    """A client whose reviewer ``n`` answers ``members[n]`` and whose area chair answers ``meta``."""

    def member(request: FakeRequest) -> str:
        return members[reviewer_index(request)]

    script: dict = {"review": member}
    if meta is not None:
        script["review.meta"] = meta
    return FakeLLM(script)


def test_ensemble_scores_are_rounded_means_of_the_in_range_member_scores():
    client = ensemble(
        {
            1: review_json(Overall=5, Clarity=2, Quality=1, Originality=9, Confidence=5),
            2: review_json(Overall=6, Clarity=3, Quality=2, Originality=0, Confidence=4),
            3: review_json(Overall=8, Clarity=3, Quality=4, Originality=3, Confidence=4),
        },
        meta=review_json(
            Summary="The reviewers agree the idea is clear.",
            Weaknesses=["ablations are thin", "no latency numbers"],
            Overall=1,
            Clarity=1,
            Decision="Accept",
            EthicalConcerns=True,
        ),
    )

    review = perform_review_agentreview(PAPER, "fake", generate=generate_from_client(client))

    # Text fields and the decision come from the meta-review ...
    assert review["Summary"] == "The reviewers agree the idea is clear."
    assert review["Weaknesses"] == ["ablations are thin", "no latency numbers"]
    assert review["Decision"] == "Accept"
    assert review["Ethical Concerns"] is True and "EthicalConcerns" not in review
    # ... and every score is the rounded mean over the reviewers, not the area chair's number.
    assert review["Overall"] == 6  # mean(5, 6, 8) = 6.33
    assert review["Clarity"] == 3  # mean(2, 3, 3) = 2.67
    assert review["Quality"] == 2  # mean(1, 2, 4) = 2.33
    assert review["Originality"] == 3  # 9 and 0 are outside 1..4 and do not count
    assert review["Confidence"] == 4  # mean(5, 4, 4) = 4.33
    assert all(isinstance(review[axis], int) for axis in ("Overall", "Clarity", "Confidence"))
    assert list(review) == [
        "Summary",
        "Strengths",
        "Weaknesses",
        "Originality",
        "Quality",
        "Clarity",
        "Significance",
        "Questions",
        "Limitations",
        "Ethical Concerns",
        "Soundness",
        "Presentation",
        "Contribution",
        "Overall",
        "Confidence",
        "Decision",
    ]

    labels = client.labels()
    assert sorted(labels) == ["review", "review", "review", "review.meta"]
    assert labels[-1] == "review.meta"
    members = [r for r in client.requests if r.label == "review"]
    assert sorted(reviewer_index(r) for r in members) == [1, 2, 3]
    for request in members:
        assert request.system == get_agentreview_system_prompt()
        assert request.schema is Review
        assert f"<paper>\n{PAPER}\n</paper>" in request.text
        assert request.memory is None and not request.tools
    meta_request = client.requests[-1]
    assert "area chair" in meta_request.system and "reviewed by 3 reviewers" in meta_request.system
    assert meta_request.text.count("<review index=") == 3
    assert '"Ethical Concerns": false' in meta_request.text


def test_half_way_means_round_to_even_like_the_original():
    client = ensemble(
        {1: review_json(Overall=5, Clarity=2), 2: review_json(Overall=6, Clarity=3)},
        meta=review_json(),
    )
    review = perform_review_agentreview(
        PAPER, "fake", generate=generate_from_client(client), num_review_ensemble=2
    )
    assert review["Overall"] == 6  # 5.5 rounds to the even neighbour
    assert review["Clarity"] == 2  # so does 2.5


def test_axis_without_a_valid_member_score_keeps_the_clamped_meta_score():
    client = ensemble(
        {1: review_json(Overall=0, Confidence=9), 2: review_json(Overall=14, Confidence=0)},
        meta=review_json(Overall=12, Confidence=-3),
    )
    review = perform_review_agentreview(
        PAPER, "fake", generate=generate_from_client(client), num_review_ensemble=2
    )
    assert review["Overall"] == 10
    assert review["Confidence"] == 1


def test_malformed_members_are_dropped_with_a_warning(caplog):
    client = ensemble(
        {
            1: review_json(Overall=4),
            2: "I would rather not review this paper.",
            3: review_json(Overall=7),
        },
        meta=review_json(Summary="meta"),
    )
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.eval.agent_review"):
        review = perform_review_agentreview(PAPER, "fake", generate=generate_from_client(client))

    assert review["Overall"] == 6  # mean(4, 7) = 5.5 -> 6; the dropped reviewer does not count
    assert review["Summary"] == "meta"
    assert "Review 2 of 3 was dropped" in caplog.text
    # The malformed reviewer was asked again before being given up on, and the area chair
    # only saw the two usable reviews.
    assert sum(reviewer_index(r) == 2 for r in client.requests if r.label == "review") == 3
    assert client.requests[-1].text.count("<review index=") == 2


def test_one_surviving_reviewer_needs_no_meta_review():
    client = ensemble({1: "nope", 2: review_json(Overall=7, Summary="the only one"), 3: "nope"})
    review = perform_review_agentreview(PAPER, "fake", generate=generate_from_client(client))
    assert review["Summary"] == "the only one" and review["Overall"] == 7
    assert "review.meta" not in client.labels()


def test_every_member_failing_raises_a_clear_error():
    client = FakeLLM({"review": "not a review"})
    with pytest.raises(ReviewError, match=r"all 3 review\(s\) by model 'fake' failed"):
        perform_review_agentreview(PAPER, "fake", generate=generate_from_client(client))


def test_provider_errors_count_as_failed_members():
    def generate(parts, **kwargs):
        raise LLMError("request req_123 failed: overloaded")

    with pytest.raises(ReviewError, match="overloaded") as info:
        perform_review_agentreview(PAPER, "fake", generate=generate, num_review_ensemble=2)
    assert isinstance(info.value.__cause__, LLMError)


def test_failed_meta_review_falls_back_to_the_first_review(caplog):
    client = ensemble(
        {1: review_json(Summary="first", Overall=4), 2: review_json(Summary="second", Overall=8)},
        meta="The reviewers disagree.",
    )
    with caplog.at_level(logging.WARNING, logger="paper_orchestra.eval.agent_review"):
        review = perform_review_agentreview(
            PAPER, "fake", generate=generate_from_client(client), num_review_ensemble=2
        )
    assert review["Summary"] == "first"
    assert review["Overall"] == 6
    assert "meta-review failed" in caplog.text


def test_single_review_makes_one_call_and_clamps_its_scores():
    client = FakeLLM({"review": review_json(Overall=11, Clarity=0, Decision="accept")})
    review = perform_review_agentreview(
        PAPER, "fake", generate=generate_from_client(client), num_review_ensemble=1
    )
    assert client.labels() == ["review"]
    assert "You are reviewer" not in client.requests[0].text
    assert (review["Overall"], review["Clarity"], review["Decision"]) == (10, 1, "Accept")


def test_ensemble_members_are_reviewed_in_parallel():
    barrier = threading.Barrier(3, timeout=5)

    def member(request: FakeRequest) -> str:
        barrier.wait()  # only passes when all three reviewers are in flight at once
        return review_json()

    client = FakeLLM({"review": member, "review.meta": review_json()})
    review = perform_review_agentreview(PAPER, "fake", generate=generate_from_client(client))
    assert review["Overall"] == 5


def test_persona_and_label_are_configurable():
    client = FakeLLM({"refinement.review": review_json(), "refinement.review.meta": review_json()})
    persona = get_agentreview_system_prompt(is_knowledgeable=False, is_benign=False)
    perform_review_agentreview(
        PAPER,
        "fake",
        generate=generate_from_client(client),
        num_review_ensemble=2,
        reviewer_system_prompt=persona,
        label="refinement.review",
    )
    assert sorted(client.labels()) == [
        "refinement.review",
        "refinement.review",
        "refinement.review.meta",
    ]
    assert all(r.system == persona for r in client.requests if r.label == "refinement.review")


def test_reviewer_personas():
    best = get_agentreview_system_prompt()
    assert best.startswith("You are a reviewer.")
    assert "You are knowledgeable" in best
    assert "As a responsible reviewer" in best
    assert "As a benign reviewer" in best
    for score, text in SCORE_CALCULATION.items():
        assert f"* {score}: {text}" in best

    worst = get_agentreview_system_prompt(
        is_knowledgeable=False, is_responsible=False, is_benign=False
    )
    assert "You are not knowledgeable" in worst
    assert "As a lazy reviewer" in worst
    assert "As a mean reviewer" in worst
    assert "## Rubrics for Overall Rating" in worst


def test_instructions_ask_for_no_written_out_reasoning():
    assert "THOUGHT" not in AGENTREVIEW_INSTRUCTIONS
    assert "reasoning" not in {name.lower() for name in Review.model_fields}
    for field in Review.model_fields:
        assert f'"{field}"' in AGENTREVIEW_INSTRUCTIONS


def test_review_schema_fits_structured_output_limits():
    schema = Review.model_json_schema()
    assert set(schema["required"]) == set(Review.model_fields)
    for name, spec in schema["properties"].items():
        assert " " not in name
        # No dict-shaped fields and no numeric or length constraints.
        assert spec.get("type") != "object"
        assert not {"minimum", "maximum", "minLength", "maxLength"} & set(spec)

    loose = json.loads(review_json(Limitations="none worth noting", Decision=" reject "))
    del loose["EthicalConcerns"]
    lenient = Review.model_validate(loose | {"Ethical Concerns": True, "Overall": "7"})
    assert lenient.Limitations == ["none worth noting"]
    assert (lenient.Decision, lenient.Overall, lenient.EthicalConcerns) == ("Reject", 7, True)


def test_standalone_helper_works_without_native_structured_output():
    usage = UsageTracker()
    seen: list[FakeRequest] = []

    def member(request: FakeRequest) -> str:
        seen.append(request)
        return "Here is my review.\n```json\n" + review_json(Overall=7) + "\n```"

    client = FakeLLM(
        {"review": member},
        capabilities=Capabilities(tools=False, memory_tool=False, structured_output=False),
    )
    review = perform_review_agentreview(
        PAPER,
        "fake",
        generate=generate_from_client(client, usage=usage),
        num_review_ensemble=1,
    )
    assert review["Overall"] == 7
    # The schema travelled in the prompt instead of as a response format.
    assert seen[0].schema is None and '"Overall"' in seen[0].text and "JSON Schema" in seen[0].text
    assert usage.total().requests == 1


def test_get_meta_review_returns_none_when_the_area_chair_fails():
    client = FakeLLM({"review.meta": "no"})
    reviews = [json.loads(review_json()), json.loads(review_json())]
    assert get_meta_review("fake", None, reviews, generate=generate_from_client(client)) is None
