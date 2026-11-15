from __future__ import annotations

import json
import logging
import socket
import threading
import time
from itertools import product
from pathlib import Path

import pymupdf
import pytest
from pydantic import BaseModel

from paper_orchestra.config import DEFAULT_MODEL
from paper_orchestra.eval import (
    JsonResultsFile,
    RetryPolicy,
    aggregate_sxs_runs,
    call_with_retries,
    generate_structured,
    load_json_object,
    normalize_winner,
    run_batch,
    summarize_sxs_results,
)
from paper_orchestra.eval.lit_review_quality import (
    LitReviewRating,
    perform_batch_rating,
    perform_batch_rating_folder,
    perform_batch_rating_json,
    print_literature_review_stats,
    rate_single_paper,
)
from paper_orchestra.eval.prompts.lit_review_quality_prompts import (
    lit_review_quality_system_prompt,
)
from paper_orchestra.eval.prompts.sxs_quality_prompts import (
    sxs_lit_review_quality_system_prompt,
    sxs_paper_quality_system_prompt,
)
from paper_orchestra.eval.sxs_lit_review_quality import (
    perform_sxs_lit_review_batch,
    rate_sxs_lit_review,
)
from paper_orchestra.eval.sxs_paper_quality import (
    MAX_PAGE_IMAGES,
    perform_sxs_paper_quality_batch,
    rate_sxs_paper_quality,
)
from paper_orchestra.llm.base import (
    Capabilities,
    Image,
    LLMConfigError,
    LLMError,
    LLMParseError,
    LLMRefusal,
    Pdf,
    Text,
)
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.llm.router import register_client, reset_clients

MODEL = "claude-fable-5-2"
TEXT_ONLY = Capabilities(pdf_input=False, image_input=True, structured_output=False)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    reset_clients()
    yield
    reset_clients()


def instant_retry(attempts: int = 3) -> tuple[RetryPolicy, list[float]]:
    sleeps: list[float] = []
    return RetryPolicy(attempts=attempts, base_delay=1.0, sleep=sleeps.append), sleeps


def make_pdf(path: Path, body: str, pages: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page()
        page.insert_text(
            (72, 72), f"{body} page {number + 1}. " + "Related work is discussed. " * 6
        )
    doc.save(path)
    return path


def rating_payload(overall: float = 62, coverage: float = 70) -> dict:
    axis = {"score": 55, "justification": "Not evidenced in the text."}
    return {
        "paper_title": "A Paper",
        "citation_statistics": {
            "estimated_unique_citations": 40,
            "citation_density_assessment": "Appropriate",
            "breadth_across_subareas": "moderate",
            "comparison_to_baseline": "About 70% of the reference average.",
            "notes": "",
        },
        "axis_scores": {
            "coverage_and_completeness": {"score": coverage, "justification": "Broad."},
            "relevance_and_focus": axis,
            "critical_analysis_and_synthesis": axis,
            "positioning_and_novelty": axis,
            "organization_and_writing": axis,
            "citation_practices_and_rigor": axis,
        },
        "penalties": [{"reason": "Generic gap statement", "points_deducted": -5}],
        "summary": {"strengths": ["s"], "weaknesses": ["w"], "top_improvements": ["i"]},
        "overall_score": overall,
    }


def verdict(winner: str, *, holistic: bool = False) -> str:
    prefix = "holistic_" if holistic else ""
    return json.dumps(
        {
            f"paper_1_{prefix}analysis": "analysis of the first paper",
            f"paper_2_{prefix}analysis": "analysis of the second paper",
            "comparison_justification": "because",
            "winner": winner,
        }
    )


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


class Answer(BaseModel):
    value: int


def test_generate_structured_uses_native_structured_output_when_available():
    fake = FakeLLM(['{"value": 3}'])
    answer = generate_structured(
        fake, "question", model=MODEL, schema=Answer, system="sys", label="t"
    )
    assert answer == Answer(value=3)
    request = fake.requests[0]
    assert request.schema is Answer
    assert request.text == "question"  # nothing appended: the schema travels natively
    assert request.system == "sys" and request.label == "t"


def test_generate_structured_falls_back_to_json_in_text_without_native_support():
    fake = FakeLLM(['Here you go:\n```json\n{"value": 4}\n```'], capabilities=TEXT_ONLY)
    answer = generate_structured(
        fake, [Text("question")], model="gemini-3.1-pro-preview", schema=Answer
    )
    assert answer == Answer(value=4)
    request = fake.requests[0]
    assert request.schema is None
    assert request.parts[0] == Text("question")
    assert "JSON Schema" in request.parts[1].text and '"value"' in request.parts[1].text


def test_generate_structured_falls_back_when_the_provider_rejects_the_schema():
    class RejectsSchemas(FakeLLM):
        def conversation(self, **options):
            conversation = super().conversation(**options)
            send = conversation.send

            def guarded(parts, *, schema=None, **kwargs):
                if schema is not None:
                    raise LLMConfigError("structured outputs are disabled for this organization")
                return send(parts, **kwargs)

            conversation.send = guarded
            return conversation

    fake = RejectsSchemas(['{"value": 5}'])
    assert generate_structured(fake, "q", model=MODEL, schema=Answer) == Answer(value=5)
    assert "JSON Schema" in fake.requests[0].text


def test_generate_structured_reports_a_reply_that_does_not_fit_the_schema():
    for capabilities in (None, TEXT_ONLY):
        fake = FakeLLM(['{"value": "many"}'], capabilities=capabilities)
        with pytest.raises(LLMParseError):
            generate_structured(fake, "q", model=MODEL, schema=Answer)
        assert len(fake.requests) == 1  # retrying is left to call_with_retries


def test_call_with_retries_backs_off_and_reraises_the_last_error():
    retry, sleeps = instant_retry(attempts=4)
    calls: list[int] = []

    def flaky() -> str:
        calls.append(len(calls))
        if len(calls) < 3:
            raise LLMError(f"overloaded #{len(calls)}")
        return "ok"

    assert call_with_retries(flaky, what="flaky call", retry=retry) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2
    assert 0.75 <= sleeps[0] <= 1.0 and 1.5 <= sleeps[1] <= 2.0

    calls.clear()
    sleeps.clear()

    def broken() -> str:
        calls.append(len(calls))
        raise LLMError(f"overloaded #{len(calls)}")

    with pytest.raises(LLMError, match="overloaded #4") as excinfo:
        call_with_retries(broken, what="broken call", retry=retry)
    assert len(calls) == 4 and len(sleeps) == 3
    assert excinfo.value.__notes__ == ["broken call failed after 4 attempt(s)."]


def test_call_with_retries_caps_the_delay():
    policy = RetryPolicy(base_delay=10.0, max_delay=25.0)
    assert all(policy.delay(failed) <= 25.0 for failed in range(1, 12))
    assert policy.delay(1) <= 10.0


@pytest.mark.parametrize("error", [LLMConfigError("no credentials"), LLMRefusal("declined")])
def test_call_with_retries_does_not_repeat_what_cannot_change(error):
    retry, sleeps = instant_retry()
    calls: list[int] = []

    def fail() -> None:
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        call_with_retries(fail, what="call", retry=retry)
    assert len(calls) == 1 and sleeps == []


def test_call_with_retries_lets_programming_errors_through_at_once():
    retry, sleeps = instant_retry()
    with pytest.raises(ZeroDivisionError):
        call_with_retries(lambda: 1 / 0, what="call", retry=retry)
    assert sleeps == []


def test_json_results_file_survives_concurrent_writers(tmp_path):
    path = tmp_path / "nested" / "results.json"
    results = JsonResultsFile(path)
    barrier = threading.Barrier(8)

    def writer(worker: int) -> None:
        barrier.wait()
        for index in range(25):
            results.set(f"w{worker}-{index}", {"worker": worker, "index": index})

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    on_disk = json.loads(path.read_text())
    assert len(on_disk) == 200 and on_disk == results.snapshot()
    assert "w3-7" in results and len(results) == 200
    assert not list(path.parent.glob("*.tmp"))
    # A new handle picks the results up again.
    assert JsonResultsFile(path).snapshot() == on_disk


def test_results_files_are_never_silently_reset(tmp_path):
    damaged = tmp_path / "damaged.json"
    damaged.write_text('{"paper": {"overall_score": 7')
    with pytest.raises(ValueError, match="not valid JSON"):
        JsonResultsFile(damaged)
    assert damaged.read_text() == '{"paper": {"overall_score": 7'

    wrong_shape = tmp_path / "list.json"
    wrong_shape.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        load_json_object(wrong_shape)

    blank = tmp_path / "blank.json"
    blank.write_text("  \n")
    assert load_json_object(blank) == {}
    assert load_json_object(tmp_path / "absent.json") == {}


def test_run_batch_isolates_failures_but_stops_on_configuration_errors(caplog):
    done: list[str] = []

    def worker(name: str, item: int) -> None:
        if item < 0:
            raise RuntimeError(f"bad item {item}")
        done.append(name)

    with caplog.at_level(logging.ERROR):
        failures = run_batch({"a": 1, "b": -1, "c": 3}, worker, max_workers=2, what="demo")
    assert sorted(done) == ["a", "c"]
    assert failures == {"b": "RuntimeError: bad item -1"}
    assert "demo failed for 1 of 3 item(s): b" in caplog.text
    assert run_batch({}, worker, max_workers=2, what="demo") == {}

    started: list[str] = []

    def misconfigured(name: str, item: int) -> None:
        started.append(name)
        if len(started) > 1:
            time.sleep(0.05)  # give the batch time to cancel what is still queued
        raise LLMConfigError("no credentials")

    with pytest.raises(LLMConfigError):
        run_batch({str(n): n for n in range(50)}, misconfigured, max_workers=1, what="demo")
    assert len(started) < 50  # queued items were cancelled


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_lit_review_prompt_renders_with_format_and_keeps_the_rubric():
    rendered = lit_review_quality_system_prompt.format(avg_citation_count=137)
    assert "{" not in rendered and "}" not in rendered
    assert rendered.count("137") == 6
    for phrase in (
        "Reference Average Citation Count: 137",
        "Coverage: 20%",
        "Critical Analysis: 25%",
        "93–100 = Exceptional",
        "Do not quote more than 25 total words",
        "citation_practices_and_rigor",
    ):
        assert phrase in rendered
    # Every field the prompt describes exists in the schema, and vice versa.
    for field in LitReviewRating.model_fields:
        assert field in rendered


@pytest.mark.parametrize(
    "prompt", [sxs_lit_review_quality_system_prompt, sxs_paper_quality_system_prompt]
)
def test_sxs_prompts_are_plain_strings_without_format_escapes(prompt):
    assert "{{" not in prompt and "}}" not in prompt
    assert '"paper_1", "paper_2", or "tie"' in prompt
    assert "ordering of the papers is arbitrary" in prompt


# ---------------------------------------------------------------------------
# Literature review quality
# ---------------------------------------------------------------------------


def test_rate_single_paper_sends_the_pdf_and_returns_the_validated_rating(tmp_path):
    pdf = make_pdf(tmp_path / "final_refined_paper.pdf", "Our method")
    fake = FakeLLM(lambda request: json.dumps(rating_payload()))
    rating = rate_single_paper(str(pdf), MODEL, avg_citation_count=41, client=fake)

    assert rating["overall_score"] == 62
    assert rating["axis_scores"]["coverage_and_completeness"] == {
        "score": 70,
        "justification": "Broad.",
    }
    assert rating["citation_statistics"]["citation_density_assessment"] == "appropriate"
    assert rating["penalties"] == [{"reason": "Generic gap statement", "points_deducted": -5}]
    assert set(rating) == set(rating_payload())

    request = fake.requests[0]
    assert request.label == "eval.lit_review_quality"
    assert request.schema is LitReviewRating
    assert "Reference Average Citation Count: 41" in request.system
    assert request.text == "Rate this paper."
    # The PDF travels as a document whose title does not give the file name away.
    assert [type(part) for part in request.parts] == [Pdf, Text]
    assert request.pdfs[0].data == pdf.read_bytes()
    assert request.pdfs[0].title == "Paper under review"


def test_rate_single_paper_sends_extracted_text_to_providers_without_pdf_input(tmp_path):
    pdf = make_pdf(tmp_path / "paper.pdf", "Our method")
    fake = FakeLLM(lambda request: json.dumps(rating_payload()), capabilities=TEXT_ONLY)
    rating = rate_single_paper(str(pdf), "gpt-5", client=fake)
    assert rating["paper_title"] == "A Paper"
    request = fake.requests[0]
    assert request.pdfs == []
    assert request.text.startswith("<paper>\nOur method page 1.")
    assert "</paper>\n\nRate this paper." in request.text
    assert "JSON Schema" in request.text


def test_rate_single_paper_raises_instead_of_returning_an_empty_string(tmp_path):
    fake = FakeLLM(lambda request: "I would rather not.")
    retry, sleeps = instant_retry(attempts=3)
    with pytest.raises(FileNotFoundError, match=r"nope\.pdf"):
        rate_single_paper(str(tmp_path / "nope.pdf"), MODEL, client=fake, retry=retry)
    assert fake.requests == []

    pdf = make_pdf(tmp_path / "paper.pdf", "Our method")
    with pytest.raises(LLMParseError):
        rate_single_paper(str(pdf), MODEL, client=fake, retry=retry)
    assert len(fake.requests) == 3  # bounded
    assert len(sleeps) == 2 and sleeps[0] < sleeps[1]  # and it waits between attempts


def test_batch_rating_saves_incrementally_skips_done_papers_and_reports_failures(tmp_path, caplog):
    papers = {
        "good": str(make_pdf(tmp_path / "good.pdf", "GOOD paper")),
        "bad": str(make_pdf(tmp_path / "bad.pdf", "BAD paper")),
        "missing": str(tmp_path / "missing.pdf"),
    }
    output = tmp_path / "ratings.json"
    output.write_text(json.dumps({"done_before": {"overall_score": 50}}))

    def respond(request: FakeRequest) -> str:
        text = request.pdfs[0].data
        return (
            "garbage" if text == Path(papers["bad"]).read_bytes() else json.dumps(rating_payload())
        )

    fake = FakeLLM(respond)
    retry, _ = instant_retry(attempts=2)
    with caplog.at_level(logging.ERROR):
        ratings = perform_batch_rating(
            {**papers, "done_before": "unused.pdf"},
            str(output),
            max_workers=3,
            model_name=MODEL,
            client=fake,
            retry=retry,
        )

    assert sorted(ratings) == ["done_before", "good"]
    assert ratings["good"]["pdf_path"] == papers["good"]
    assert ratings["done_before"] == {"overall_score": 50}
    assert json.loads(output.read_text()) == ratings
    assert "literature-review rating failed for bad" in caplog.text
    assert "literature-review rating failed for missing: FileNotFoundError" in caplog.text
    assert "failed for 2 of 3 item(s)" in caplog.text

    # Nothing left to do for the rated papers: no further model call.
    calls = len(fake.requests)
    perform_batch_rating({"good": papers["good"]}, str(output), model_name=MODEL, client=fake)
    assert len(fake.requests) == calls


def test_batch_rating_folder_names_the_file_and_averages_the_scores(tmp_path):
    root = tmp_path / "runs"
    make_pdf(root / "p1" / "final_paper.pdf", "FIRST")
    make_pdf(root / "p2" / "final_paper.pdf", "SECOND")
    (root / "p3").mkdir()  # no paper produced
    first = (root / "p1" / "final_paper.pdf").read_bytes()

    def respond(request: FakeRequest) -> str:
        is_first = request.pdfs[0].data == first
        return json.dumps(
            rating_payload(overall=60 if is_first else 80, coverage=40 if is_first else 50)
        )

    register_client("anthropic", FakeLLM(respond))
    ratings = perform_batch_rating_folder(str(root), max_workers=2)

    output = root / f"lit_review_quality_ratings_{DEFAULT_MODEL}_final_paper.json"
    assert json.loads(output.read_text()) == ratings
    assert sorted(ratings) == ["overall_metrics", "p1", "p2"]
    assert ratings["p1"]["pdf_path"] == str(root / "p1" / "final_paper.pdf")
    assert ratings["overall_metrics"]["overall_score"] == 70
    assert ratings["overall_metrics"]["coverage_and_completeness"] == 45
    assert ratings["overall_metrics"]["relevance_and_focus"] == 55
    assert next(iter(ratings["overall_metrics"])) == "overall_score"

    with pytest.raises(FileNotFoundError):
        perform_batch_rating_folder(str(tmp_path / "absent"))


def test_batch_rating_folder_reports_an_empty_scan(tmp_path, caplog):
    (tmp_path / "p1").mkdir()
    with caplog.at_level(logging.WARNING):
        ratings = perform_batch_rating_folder(str(tmp_path), client=FakeLLM([]))
    assert ratings == {}
    assert "contains final_paper.pdf" in caplog.text
    assert not list(tmp_path.glob("*.json"))


def test_batch_rating_json_reads_the_paper_list(tmp_path, caplog):
    pdf = make_pdf(tmp_path / "a.pdf", "Paper A")
    listing = tmp_path / "papers.json"
    listing.write_text(
        json.dumps(
            [
                {"paper_id": "a", "pdf_path": str(pdf)},
                {"paper_id": "ghost", "pdf_path": str(tmp_path / "ghost.pdf")},
            ]
        )
    )
    fake = FakeLLM(lambda request: json.dumps(rating_payload(overall=66)))
    with caplog.at_level(logging.WARNING):
        ratings = perform_batch_rating_json(str(listing), model_name=MODEL, client=fake)
    output = tmp_path / "papers_lit_review_quality_ratings.json"
    assert json.loads(output.read_text()) == ratings
    assert ratings["a"]["overall_score"] == 66
    assert ratings["overall_metrics"]["overall_score"] == 66
    assert "Skipping ghost" in caplog.text


def test_stats_understand_the_older_rating_format(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "a": {
                    "overall_quality_score": 40,
                    "clarity": {"score": 30},
                    "presentation": {"score": 50},
                },
                "b": {"overall_quality_score": 60, "clarity": {"score": 70}},
                "overall_metrics": {"overall_score": 999},
            }
        )
    )
    assert print_literature_review_stats(str(path)) == {
        "overall_score": 50,
        "clarity": 50,
        "presentation": 50,
    }
    assert json.loads(path.read_text())["overall_metrics"]["overall_score"] == 50
    with pytest.raises(FileNotFoundError):
        print_literature_review_stats(str(tmp_path / "absent.json"))


# ---------------------------------------------------------------------------
# Side-by-side aggregation
# ---------------------------------------------------------------------------


def test_winner_spellings_are_normalised():
    assert normalize_winner("Paper 1") == "paper_1"
    assert normalize_winner(" PAPER_2 ") == "paper_2"
    assert normalize_winner("Tie") == "tie"
    assert normalize_winner("paper 2 is better") == "paper_2"
    assert normalize_winner("neither") == "neither"
    assert normalize_winner(None) is None


@pytest.mark.parametrize(("run1", "run2"), list(product(["paper_1", "paper_2", "tie"], repeat=2)))
def test_aggregate_scores_both_orders_from_paper_one_s_point_of_view(run1, run2):
    entry = aggregate_sxs_runs(
        {"winner": run1},
        {"winner": run2},
        paper1_path="a.pdf",
        paper2_path="b.pdf",
        exp1_name="ours",
        exp2_name="base",
    )
    # In run 2 the papers were swapped, so "paper_2" there means paper 1 won.
    first = {"paper_1": 1, "paper_2": -1, "tie": 0}[run1]
    second = {"paper_1": -1, "paper_2": 1, "tie": 0}[run2]
    total = first + second
    assert entry["total_score"] == total
    assert entry["p1_result_run1"] == {1: "p1", -1: "p2", 0: "tie"}[first]
    assert entry["p1_result_run2"] == {1: "p1", -1: "p2", 0: "tie"}[second]
    assert (
        entry["category_5pt"]
        == {2: "win", 1: "leaning_win", 0: "tie", -1: "leaning_loss", -2: "loss"}[total]
    )
    assert entry["category_3pt"] == ("win" if total > 0 else "loss" if total < 0 else "tie")
    assert list(entry) == [
        "run1",
        "run2",
        "p1_result_run1",
        "p1_result_run2",
        "total_score",
        "category_5pt",
        "category_3pt",
        "paper_1_path",
        "paper_2_path",
        "exp1_name",
        "exp2_name",
    ]


def test_aggregate_rejects_a_rating_without_a_usable_winner():
    with pytest.raises(ValueError, match="winner"):
        aggregate_sxs_runs(
            {"winner": "paper_1"},
            {},
            paper1_path="a",
            paper2_path="b",
            exp1_name="x",
            exp2_name="y",
        )


def test_summary_counts_and_rates():
    categories = ["win", "win", "leaning_win", "tie", "leaning_loss", "loss", "loss", "loss"]
    three = {
        "win": "win",
        "leaning_win": "win",
        "tie": "tie",
        "leaning_loss": "loss",
        "loss": "loss",
    }
    results = {
        f"p{i}": {"category_5pt": cat, "category_3pt": three[cat]}
        for i, cat in enumerate(categories)
    }
    results["_summary_stats"] = {"stale": True}
    results["junk"] = "not an entry"
    assert summarize_sxs_results(results, exp1_name="ours", exp2_name="base") == {
        "exp1_name": "ours",
        "exp2_name": "base",
        "total_valid_evals": 8,
        "five_point": {
            "wins": 2,
            "leaning_wins": 1,
            "ties": 1,
            "leaning_losses": 1,
            "losses": 3,
            "win_rate": 0.25,
            "leaning_win_rate": 0.125,
            "tie_rate": 0.125,
        },
        "three_point": {"wins": 3, "ties": 1, "losses": 4, "win_rate": 0.375, "tie_rate": 0.125},
    }
    empty = summarize_sxs_results({}, exp1_name="a", exp2_name="b")
    assert empty["total_valid_evals"] == 0 and empty["five_point"]["win_rate"] == 0


# ---------------------------------------------------------------------------
# SxS literature review
# ---------------------------------------------------------------------------


def prefers_strong(request: FakeRequest) -> str:
    """A judge without position bias: the paper containing STRONG wins."""
    first = request.text.split("</paper_1>")[0]
    return verdict("paper_1" if "STRONG" in first else "paper_2")


def test_rate_sxs_lit_review_works_for_a_claude_model_through_the_router():
    """The old code only knew 'gemini' and 'gpt' model names and never returned for anything else."""
    fake = FakeLLM(lambda request: verdict("Paper 2"))
    register_client("anthropic", fake)
    rating = rate_sxs_lit_review("first paper text", "second paper text", MODEL)

    assert rating == {
        "paper_1_analysis": "analysis of the first paper",
        "paper_2_analysis": "analysis of the second paper",
        "comparison_justification": "because",
        "winner": "paper_2",
    }
    request = fake.requests[0]
    assert request.model == MODEL and request.label == "eval.sxs_lit_review"
    assert request.system.startswith("You are an expert AI researcher")
    assert request.text == (
        "<paper_1>\nfirst paper text\n</paper_1>\n\n<paper_2>\nsecond paper text\n</paper_2>\n\n"
        "Task: Compare the literature review quality of these two papers side by side."
    )


def test_rate_sxs_lit_review_retries_are_bounded_and_the_error_surfaces():
    fake = FakeLLM(lambda request: verdict("both are fine"))
    retry, sleeps = instant_retry(attempts=4)
    with pytest.raises(LLMParseError):
        rate_sxs_lit_review("a", "b", MODEL, client=fake, retry=retry)
    assert len(fake.requests) == 4
    assert len(sleeps) == 3 and sleeps == sorted(sleeps)

    with pytest.raises(LLMConfigError, match="Cannot tell which provider"):
        rate_sxs_lit_review("a", "b", "mystery-model-1")


def make_sxs_projects(tmp_path: Path) -> tuple[Path, Path]:
    ours, base = tmp_path / "ours_iclr2025", tmp_path / "baseline"
    make_pdf(ours / "p1" / "final_paper.pdf", "STRONG ours p1")
    make_pdf(base / "p1" / "paper.pdf", "weak base p1")
    make_pdf(ours / "p2" / "final_paper.pdf", "weak ours p2")
    make_pdf(base / "p2" / "paper.pdf", "STRONG base p2")
    make_pdf(ours / "only_ours" / "final_paper.pdf", "STRONG unmatched")
    (base / "no_pdf").mkdir()
    (ours / "no_pdf").mkdir()
    return ours, base


def test_sxs_lit_review_batch_judges_both_orders_and_writes_the_summary(tmp_path):
    ours, base = make_sxs_projects(tmp_path)
    fake = FakeLLM(prefers_strong)
    output = tmp_path / "out"

    results = perform_sxs_lit_review_batch(
        str(ours), str(base), "final_paper.pdf", "paper.pdf", "ours", "base",
        output_folder=str(output), model_name=MODEL, client=fake,
    )  # fmt: skip

    path = output / f"sxs_lit_review_enhanced_{MODEL}_ours_vs_base_iclr.json"
    assert json.loads(path.read_text()) == results
    assert sorted(results) == ["_summary_stats", "p1", "p2"]
    assert results["p1"]["run1"]["winner"] == "paper_1"
    assert results["p1"]["run2"]["winner"] == "paper_2"  # swapped order, same paper wins
    assert (results["p1"]["total_score"], results["p1"]["category_5pt"]) == (2, "win")
    assert (results["p2"]["total_score"], results["p2"]["category_5pt"]) == (-2, "loss")
    assert results["p1"]["paper_1_path"] == str(ours / "p1" / "final_paper.pdf")
    assert results["p1"]["paper_2_path"] == str(base / "p1" / "paper.pdf")
    stats = results["_summary_stats"]
    assert stats["total_valid_evals"] == 2
    assert stats["five_point"]["wins"] == 1 and stats["five_point"]["losses"] == 1
    assert stats["three_point"]["win_rate"] == 0.5
    assert len(fake.requests) == 4  # two pairs, two orders each

    # A second run has nothing left to rate and keeps the file intact.
    again = perform_sxs_lit_review_batch(
        str(ours), str(base), "final_paper.pdf", "paper.pdf", "ours", "base",
        output_folder=str(output), model_name=MODEL, client=fake,
    )  # fmt: skip
    assert again == results and len(fake.requests) == 4


def test_a_position_biased_judge_yields_ties_and_failures_are_left_for_a_rerun(tmp_path, caplog):
    ours, base = make_sxs_projects(tmp_path)

    def biased(request: FakeRequest) -> str:
        if "p2" in request.text:
            return "no verdict"
        return verdict("paper_1")

    fake = FakeLLM(biased)
    retry, _ = instant_retry(attempts=2)
    with caplog.at_level(logging.ERROR):
        results = perform_sxs_lit_review_batch(
            str(ours), str(base), "final_paper.pdf", "paper.pdf", "ours", "base",
            output_folder=str(tmp_path / "out"), model_name=MODEL, client=fake, retry=retry,
        )  # fmt: skip
    assert sorted(results) == ["_summary_stats", "p1"]
    assert results["p1"]["total_score"] == 0 and results["p1"]["category_3pt"] == "tie"
    assert results["_summary_stats"]["five_point"]["ties"] == 1
    assert "SxS literature review failed for p2" in caplog.text


# ---------------------------------------------------------------------------
# SxS paper quality
# ---------------------------------------------------------------------------


def test_rate_sxs_paper_quality_sends_both_pdfs_labelled_and_in_order(tmp_path):
    paper1 = make_pdf(tmp_path / "ours" / "final_paper.pdf", "ONE")
    paper2 = make_pdf(tmp_path / "theirs" / "original_paper.pdf", "TWO")
    fake = FakeLLM(lambda request: verdict("paper_1", holistic=True))
    register_client("anthropic", fake)

    rating = rate_sxs_paper_quality(str(paper1), str(paper2))
    assert rating["winner"] == "paper_1"
    assert set(rating) == {
        "paper_1_holistic_analysis",
        "paper_2_holistic_analysis",
        "comparison_justification",
        "winner",
    }
    request = fake.requests[0]
    assert request.model == DEFAULT_MODEL and request.label == "eval.sxs_paper_quality"
    assert [type(part) for part in request.parts] == [Text, Pdf, Text, Pdf, Text]
    assert [pdf.data for pdf in request.pdfs] == [paper1.read_bytes(), paper2.read_bytes()]
    # Neutral titles: the file names would tell the judge which paper is the original.
    assert [pdf.title for pdf in request.pdfs] == ["Paper 1", "Paper 2"]
    assert request.text.endswith("writing quality of these two papers side by side.")


def test_rate_sxs_paper_quality_sends_text_and_page_images_without_pdf_input(tmp_path):
    paper1 = make_pdf(tmp_path / "one.pdf", "ONE", pages=MAX_PAGE_IMAGES + 3)
    paper2 = make_pdf(tmp_path / "two.pdf", "TWO", pages=2)
    fake = FakeLLM(lambda request: verdict("tie", holistic=True), capabilities=TEXT_ONLY)

    assert rate_sxs_paper_quality(str(paper1), str(paper2), "gpt-5", client=fake)["winner"] == "tie"
    request = fake.requests[0]
    assert request.pdfs == []
    assert len(request.images) == MAX_PAGE_IMAGES + 2
    assert {image.media_type for image in request.images} == {"image/jpeg"}
    assert all(image.data.startswith(b"\xff\xd8\xff") for image in request.images)
    kinds = [type(part) for part in request.parts]
    assert kinds[:2] == [Text, Text] and kinds[2] is Image and kinds[-1] is Text
    texts = [part.text for part in request.parts if isinstance(part, Text)]
    assert texts[0].startswith("<paper_1>\nONE page 1.") and texts[1] == "Paper 1 visual pages:"
    assert texts[2].startswith("<paper_2>\nTWO page 1.") and texts[3] == "Paper 2 visual pages:"

    no_vision = FakeLLM(
        lambda request: verdict("tie", holistic=True),
        capabilities=Capabilities(pdf_input=False, image_input=False),
    )
    rate_sxs_paper_quality(str(paper1), str(paper2), "gpt-5", client=no_vision)
    assert no_vision.requests[0].images == []


def test_rate_sxs_paper_quality_fails_loudly(tmp_path):
    paper = make_pdf(tmp_path / "one.pdf", "ONE")
    fake = FakeLLM(lambda request: "{}")
    retry, sleeps = instant_retry(attempts=3)
    with pytest.raises(FileNotFoundError, match=r"two\.pdf"):
        rate_sxs_paper_quality(
            str(paper), str(tmp_path / "two.pdf"), MODEL, client=fake, retry=retry
        )
    assert fake.requests == []
    with pytest.raises(LLMParseError):
        rate_sxs_paper_quality(str(paper), str(paper), MODEL, client=fake, retry=retry)
    assert len(fake.requests) == 3 and len(sleeps) == 2


def test_sxs_paper_quality_batch_end_to_end(tmp_path):
    ours, base = make_sxs_projects(tmp_path)
    strong = {
        (ours / "p1" / "final_paper.pdf").read_bytes(),
        (base / "p2" / "paper.pdf").read_bytes(),
    }

    def judge(request: FakeRequest) -> str:
        return verdict("paper_1" if request.pdfs[0].data in strong else "paper_2", holistic=True)

    fake = FakeLLM(judge)
    results = perform_sxs_paper_quality_batch(
        str(ours), str(base), "final_paper.pdf", "paper.pdf", "ours", "base",
        output_folder=str(tmp_path / "out"), model_name=MODEL, max_workers=2, client=fake,
    )  # fmt: skip

    path = tmp_path / "out" / f"sxs_paper_quality_enhanced_{MODEL}_ours_vs_base_iclr.json"
    assert json.loads(path.read_text()) == results
    assert results["p1"]["category_5pt"] == "win" and results["p2"]["category_5pt"] == "loss"
    assert "paper_1_holistic_analysis" in results["p1"]["run1"]
    assert results["_summary_stats"]["exp1_name"] == "ours"
    assert results["_summary_stats"]["three_point"] == {
        "wins": 1,
        "ties": 0,
        "losses": 1,
        "win_rate": 0.5,
        "tie_rate": 0.0,
    }
    with pytest.raises(FileNotFoundError, match="project folder not found"):
        perform_sxs_paper_quality_batch(
            str(tmp_path / "absent"), str(base), "a.pdf", "b.pdf", "x", "y",
            output_folder=str(tmp_path / "out"), model_name=MODEL, client=fake,
        )  # fmt: skip
