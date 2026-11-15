# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgentReview-style paper review: an ensemble of reviewers aggregated by a meta-reviewer.

The module is provider-neutral.  Every model call goes through a ``generate``
callable with the keyword interface of ``ContextHarness.generate`` already
bound to an agent and a stage, so the caller decides which harness, memory
policy and usage ledger the reviews run under.  :func:`generate_from_client`
builds such a callable from a bare :class:`LLMClient` for standalone use.
"""

from __future__ import annotations

import functools
import json
import logging
import statistics
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol

from pydantic import AliasChoices, BaseModel, Field, ValidationError, field_validator

from paper_orchestra.llm.base import LLMClient, LLMError, LLMResponse, Part, Schema, UsageTracker

log = logging.getLogger(__name__)

# --- AGENTREVIEW SPECIFIC CONSTANTS (From role_descriptions.py) ---

SCORE_CALCULATION = {
    10: "This study is among the top 2% of all papers. It is one of the most thorough I have seen. It changed my thinking on this topic. I would fight for it to be accepted",
    8: "This study is among the top 10% of all papers. It provides sufficient support for all of its claims/arguments. Some extra experiments are needed, but not essential. The method is highly original and generalizable to various fields. It deepens the understanding of some phenomenons or lowers the barriers to an existing research direction",
    6: "This study provides sufficient support for its major claims/arguments, some minor points may need extra support or details. The method is moderately original and generalizable to various relevant fields. The work it describes is not particularly interesting and/or novel, so it will not be a big loss if people don’t see it in this conference",
    5: "Some of the main claims/arguments are not sufficiently supported, there are major technical/methodological problems. The proposed method is somewhat original and generalizable to various relevant fields. I am leaning towards rejection, but I can be persuaded if my co-reviewers think otherwise",
    3: "This paper makes marginal contributions",
    1: "This study is not yet sufficiently thorough to warrant publication or is not relevant to the conference",
}

AGENT_REVIEW_RUBRICS = (
    f"* 10: {SCORE_CALCULATION[10]};\n"
    f"* 8: {SCORE_CALCULATION[8]};\n"
    f"* 6: {SCORE_CALCULATION[6]};\n"
    f"* 5: {SCORE_CALCULATION[5]};\n"
    f"* 3: {SCORE_CALCULATION[3]};\n"
    f"* 1: {SCORE_CALCULATION[1]}. "
)

#: Legal range of every numeric field of a review.
SCORE_LIMITS: dict[str, tuple[int, int]] = {
    "Originality": (1, 4),
    "Quality": (1, 4),
    "Clarity": (1, 4),
    "Significance": (1, 4),
    "Soundness": (1, 4),
    "Presentation": (1, 4),
    "Contribution": (1, 4),
    "Overall": (1, 10),
    "Confidence": (1, 5),
}


class ReviewError(RuntimeError):
    """No usable review could be obtained."""


class GenerateFn(Protocol):
    """``ContextHarness.generate`` with ``agent`` and ``stage`` already bound."""

    def __call__(
        self,
        parts: Part | Sequence[Part],
        *,
        model: str,
        system: str | None = ...,
        schema: Schema | None = ...,
        temperature: float | None = ...,
        label: str | None = ...,
    ) -> LLMResponse: ...


def generate_from_client(
    client: LLMClient,
    *,
    agent: str = "Reviewer",
    stage: str = "review",
    usage: UsageTracker | None = None,
) -> GenerateFn:
    """A ``generate`` callable for reviewing outside a pipeline run.

    The calls go through a memory-less harness, which keeps the structured
    output fallback for providers without native support and records token
    usage in ``usage`` when one is given.
    """
    from paper_orchestra.memory.harness import ContextHarness

    harness = ContextHarness(client_for=lambda _model: client, usage=usage)
    return functools.partial(harness.generate, agent=agent, stage=stage)


# --- REVIEW SCHEMA ---


class Review(BaseModel):
    """One peer review.

    The ratings carry no numeric bounds in the schema (structured output does
    not support them); out-of-range values are handled when reviews are
    aggregated.  Dump with ``by_alias=True`` to get the keys of the review
    files (``"Ethical Concerns"``).
    """

    Summary: str = Field(description="A summary of the paper content and its contributions.")
    Strengths: list[str] = Field(description="A list of strengths of the paper.")
    Weaknesses: list[str] = Field(description="A list of weaknesses of the paper.")
    Originality: int = Field(description="A rating from 1 to 4 (low, medium, high, very high).")
    Quality: int = Field(description="A rating from 1 to 4 (low, medium, high, very high).")
    Clarity: int = Field(description="A rating from 1 to 4 (low, medium, high, very high).")
    Significance: int = Field(description="A rating from 1 to 4 (low, medium, high, very high).")
    Questions: list[str] = Field(
        description="A set of clarifying questions to be answered by the paper authors."
    )
    Limitations: list[str] = Field(
        description="A set of limitations and potential negative societal impacts of the work."
    )
    EthicalConcerns: bool = Field(
        validation_alias=AliasChoices("EthicalConcerns", "Ethical Concerns"),
        serialization_alias="Ethical Concerns",
        description="Whether there are ethical concerns.",
    )
    Soundness: int = Field(description="A rating from 1 to 4 (poor, fair, good, excellent).")
    Presentation: int = Field(description="A rating from 1 to 4 (poor, fair, good, excellent).")
    Contribution: int = Field(description="A rating from 1 to 4 (poor, fair, good, excellent).")
    Overall: int = Field(description="A rating from 1 to 10 (very strong reject to award quality).")
    Confidence: int = Field(
        description="A rating from 1 to 5 (low, medium, high, very high, absolute)."
    )
    Decision: Literal["Accept", "Reject"] = Field(description="Accept or Reject.")

    @field_validator("Strengths", "Weaknesses", "Questions", "Limitations", mode="before")
    @classmethod
    def _one_item_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @field_validator("Decision", mode="before")
    @classmethod
    def _decision_case(cls, value: Any) -> Any:
        return value.strip().capitalize() if isinstance(value, str) else value


# --- PROMPT GENERATORS ---


def get_agentreview_system_prompt(
    is_knowledgeable: bool = True, is_responsible: bool = True, is_benign: bool = True
) -> str:
    """
    Constructs the system prompt based on AgentReview's role_descriptions.py.
    Defaults to the 'Best Case' reviewer (Knowledgeable, Responsible, Benign).
    """
    bio = "You are a reviewer. You write peer review of academic papers by evaluating their technical quality, originality, and clarity.\n\n"

    if is_knowledgeable:
        bio += "Knowledgeability: You are knowledgeable, with a strong background and a PhD degree in the subject areas related to this paper. You possess the expertise necessary to scrutinize and provide insightful feedback to this paper.\n\n"
    else:
        bio += "Knowledgeability: You are not knowledgeable and do not have strong background in the subject areas related to this paper.\n\n"

    if is_responsible:
        bio += "Responsibility: As a responsible reviewer, you highly responsibly write paper reviews. You meticulously assess a research paper's technical accuracy, innovation, and relevance. You thoroughly read the paper, critically analyze the methodologies, and carefully consider the paper's contribution to the field.\n\n"
    else:
        bio += "Responsibility: As a lazy reviewer, your reviews tend to be superficial and hastily done. Your assessments might overlook critical details, lack depth in analysis, fail to recognize novel contributions, or offer generic feedback.\n\n"

    if is_benign:
        bio += "Intention: As a benign reviewer, your approach to reviewing is guided by a genuine intention to aid authors in enhancing their work. You provide detailed, constructive feedback, aimed at both validating robust research and guiding authors to refine and improve their work. You are also critical of technical flaws in the paper.\n\n"
    else:
        bio += "Intention: As a mean reviewer, your reviewing style is often harsh and overly critical, with a tendency towards negative bias. Your reviews may focus excessively on faults, sometimes overlooking the paper's merits.\n\n"

    bio += f"## Rubrics for Overall Rating\n{AGENT_REVIEW_RUBRICS}"

    return bio


# AgentReview's review form.  The original asked for a free-form THOUGHT section before
# the JSON; the review is now returned through the `Review` schema alone.
AGENTREVIEW_INSTRUCTIONS = """Write your review as a structured record with the following fields:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions to be answered by the paper authors.
- "Limitations": A set of limitations and potential negative societal impacts of the work.
- "EthicalConcerns": A boolean value indicating whether there are ethical concerns.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall": A rating from 1 to 10 (very strong reject to award quality).
- "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
- "Decision": A decision that has to be one of the following: Accept, Reject.

For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
Do not make generic comments: be specific to the paper in front of you."""

META_REVIEWER_SYSTEM_PROMPT = """You are a very knowledgeable and experienced area chair at a top-tier machine learning conference.
You are in charge of meta-reviewing a paper that was reviewed by {reviewer_count} reviewers.
Your job is to aggregate the reviews into a single meta-review in the same format.
You are an inclusive area chair. You tend to hear from all reviewers' opinions and combine them with your own judgments to make the final decision."""


def _review_dict(parsed: Any) -> dict[str, Any]:
    """The review as stored on disk; raises ``ValidationError`` if it is not one."""
    review = parsed if isinstance(parsed, Review) else Review.model_validate(parsed)
    return review.model_dump(by_alias=True)


def _valid_score(value: Any, limits: tuple[int, int]) -> int | None:
    try:
        score = int(value)
    except (ValueError, TypeError):
        return None
    return score if limits[0] <= score <= limits[1] else None


def _clamped(value: Any, limits: tuple[int, int]) -> int:
    try:
        score = int(value)
    except (ValueError, TypeError):
        return limits[0]
    return min(max(score, limits[0]), limits[1])


def perform_review_agentreview(
    text: str,
    model_name: str,
    *,
    generate: GenerateFn,
    num_review_ensemble: int = 3,
    temperature: float | None = 0.75,
    reviewer_system_prompt: str | None = None,
    review_instruction_form: str = AGENTREVIEW_INSTRUCTIONS,
    label: str = "review",
) -> dict[str, Any]:
    """Review ``text`` and return the review as a dict (``"Overall"``, ``"Weaknesses"`` ...).

    With ``num_review_ensemble`` > 1 the reviews are written in parallel and
    aggregated by a meta-reviewer; each numeric score of the result is the
    rounded mean of the in-range scores of the individual reviews.  A reviewer
    whose call fails or whose output is not a review is dropped with a warning;
    :class:`ReviewError` is raised when none is left.

    ``label`` names the reviewer calls in the usage ledger and the transcript
    (the meta-review uses ``<label>.meta``).
    """
    log.info("Reviewing the paper with model %s (%d reviewers).", model_name, num_review_ensemble)
    if reviewer_system_prompt is None:
        # The "best case" AgentReview persona.
        reviewer_system_prompt = get_agentreview_system_prompt(
            is_knowledgeable=True, is_responsible=True, is_benign=True
        )
    prompt = (
        f"{review_instruction_form}\n\n"
        f"Here is the paper you are asked to review:\n<paper>\n{text}\n</paper>"
    )
    count = max(1, num_review_ensemble)
    failures: list[BaseException] = []

    def write_review(index: int) -> dict[str, Any] | None:
        member_prompt = prompt
        if count > 1:
            # Sampling parameters are not available on every model, so the reviewers are
            # also told apart in the prompt.
            member_prompt += (
                f"\n\nYou are reviewer {index + 1} of {count}. "
                "The reviewers assess the paper independently of each other."
            )
        try:
            response = generate(
                member_prompt,
                model=model_name,
                system=reviewer_system_prompt,
                schema=Review,
                temperature=temperature,
                label=label,
            )
            return _review_dict(response.parsed)
        except (LLMError, ValidationError) as exc:
            failures.append(exc)
            log.warning(
                "Review %d of %d was dropped (%s: %s).",
                index + 1,
                count,
                type(exc).__name__,
                exc,
            )
            return None

    if count == 1:
        results = [write_review(0)]
    else:
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="review") as pool:
            results = list(pool.map(write_review, range(count)))
    reviews = [review for review in results if review is not None]
    if not reviews:
        raise ReviewError(
            f"all {count} review(s) by model '{model_name}' failed; last error: {failures[-1]}"
        ) from failures[-1]

    review = dict(reviews[0])
    if len(reviews) > 1:
        review = (
            get_meta_review(
                model_name, temperature, reviews, generate=generate, label=f"{label}.meta"
            )
            or review
        )
    for axis, limits in SCORE_LIMITS.items():
        scores = [s for r in reviews if (s := _valid_score(r.get(axis), limits)) is not None]
        if scores:
            review[axis] = round(statistics.fmean(scores))
        else:
            review[axis] = _clamped(review.get(axis), limits)
    return review


def get_meta_review(
    model: str,
    temperature: float | None,
    reviews: Sequence[dict[str, Any]],
    *,
    generate: GenerateFn,
    label: str = "review.meta",
) -> dict[str, Any] | None:
    """Aggregate ``reviews`` into one review; ``None`` when the meta-reviewer fails."""
    review_text = "".join(
        f'\n<review index="{index}" of="{len(reviews)}">\n{json.dumps(review)}\n</review>\n'
        for index, review in enumerate(reviews, start=1)
    )
    try:
        response = generate(
            AGENTREVIEW_INSTRUCTIONS + "\n" + review_text,
            model=model,
            system=META_REVIEWER_SYSTEM_PROMPT.format(reviewer_count=len(reviews)),
            schema=Review,
            temperature=temperature,
            label=label,
        )
        return _review_dict(response.parsed)
    except (LLMError, ValidationError) as exc:
        log.warning(
            "The meta-review failed (%s: %s); using the first review as the base.",
            type(exc).__name__,
            exc,
        )
        return None
