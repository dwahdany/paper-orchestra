from __future__ import annotations

import json
import logging
from pathlib import Path

import pymupdf
import pytest
import requests

from paper_orchestra.eval import BatchFailedError, RetryPolicy
from paper_orchestra.eval.citation_f1 import (
    S2_SEARCH_URL,
    ReferenceF1V1Evaluator,
    ScholarLookupError,
    SemanticScholarLookup,
    analyze_citation_metrics,
    generate_gt_citations_batch,
    get_citation_f1_metrics,
    parse_priorities,
    review_citation_f1_batch,
    split_reference_list,
)
from paper_orchestra.eval.references import (
    NO_REFERENCES,
    extract_reference_from_pdf_text,
    get_paper_references_from_pdf,
)
from paper_orchestra.llm.base import LLMConfigError, LLMParseError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.llm.router import register_client, reset_clients
from paper_orchestra.utils.pdf import PdfError

MODEL = "claude-fable-5-2"

# Title -> Semantic Scholar id of the papers the fake index knows.
INDEX = {
    "Deep Residual Learning": "id-resnet",
    "Attention Is All You Need": "id-transformer",
    "ImageNet": "id-imagenet",
    "Adam": "id-adam",
    "Dropout": "id-dropout",
}

ORIGINAL_REFS = (
    "[1] He et al. | Deep Residual Learning | CVPR 2016 "
    "[2] Vaswani et al. | Attention Is All You Need | NeurIPS 2017 "
    "[3] Deng et al. | ImageNet | CVPR 2009 "
    "[4] Kingma and Ba | Adam | ICLR 2015 "
    "[5] Nobody | A Paper Nobody Indexed | arXiv 2020"
)
GENERATED_REFS = (
    "[1] He et al. | Deep Residual Learning | CVPR 2016 "
    "[2] Srivastava et al. | Dropout | JMLR 2014 "
    "[3] Deng et al. | ImageNet | CVPR 2009"
)
ORIGINAL_PRIORITIES = {"1": "P0", "2": "P1", "3": "P0", "4": "P1", "5": "P0"}


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: object = None, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeScholarSession:
    """Answers title searches from ``INDEX``; ``script`` overrides the next responses."""

    def __init__(self, script: list[FakeResponse | Exception] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[dict] = []

    def get(self, url: str, *, headers, params, timeout) -> FakeResponse:
        self.calls.append(
            {"url": url, "headers": dict(headers), "params": dict(params), "timeout": timeout}
        )
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        paper_id = INDEX.get(params["query"])
        data = [{"paperId": paper_id, "title": params["query"].upper()}] if paper_id else []
        return FakeResponse(200, {"total": len(data), "data": data})


class Clock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_lookup(
    session: FakeScholarSession, clock: Clock | None = None, **options
) -> SemanticScholarLookup:
    clock = clock or Clock()
    options.setdefault("min_interval", 0.0)
    return SemanticScholarLookup(session=session, sleep=clock.sleep, clock=clock, **options)


def make_pdf(path: Path, marker: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), f"{marker} " + "We study residual networks at scale. " * 6)
    doc.save(path)
    return path


def fake_judge(priorities: dict[str, str] | str = ORIGINAL_PRIORITIES, **overrides) -> FakeLLM:
    """A model that extracts references, titles and priorities the way the prompts ask."""

    def references(request: FakeRequest) -> str:
        if "ORIGINAL" in request.text:
            return ORIGINAL_REFS
        if "NOREFS" in request.text:
            return NO_REFERENCES
        return GENERATED_REFS

    def title(request: FakeRequest) -> str:
        citation = request.text.split("<citation>")[1].split("</citation>")[0]
        return citation.split("|")[1].strip()

    def classify(request: FakeRequest) -> str:
        return priorities if isinstance(priorities, str) else json.dumps(priorities)

    responders = {
        "eval.reference_extraction": references,
        "eval.title_extraction": title,
        "eval.citation_priorities": classify,
    }
    responders.update(overrides)
    return FakeLLM(responders)


def instant_retry(attempts: int = 3) -> tuple[RetryPolicy, list[float]]:
    sleeps: list[float] = []
    return RetryPolicy(attempts=attempts, base_delay=1.0, sleep=sleeps.append), sleeps


@pytest.fixture(autouse=True)
def _fresh_router():
    reset_clients()
    yield
    reset_clients()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def test_split_reference_list_drops_preamble_and_normalises_whitespace():
    raw = "Here are the references:\n[1] First  paper,\n 2020. [2] Second paper. [3]   Third."
    assert split_reference_list(raw) == ["First paper, 2020.", "Second paper.", "Third."]
    assert split_reference_list("no numbered entries at all") == []
    assert split_reference_list("") == []


def test_parse_priorities_accepts_the_documented_shape_only():
    fenced = 'Sure.\n```json\n{"1": "P0", "[2]": "p1", " 3 ": "P0"}\n```'
    assert parse_priorities(fenced) == {"1": "P0", "2": "P1", "3": "P0"}
    # Invalid labels and keys are dropped; nothing valid left means "did not parse".
    assert parse_priorities('{"1": "P2", "x": "P0", "2": "P1"}') == {"2": "P1"}
    assert parse_priorities('{"1": "maybe"}') is None
    assert parse_priorities("[1, 2]") is None
    assert parse_priorities("not json") is None


# ---------------------------------------------------------------------------
# Semantic Scholar lookup
# ---------------------------------------------------------------------------


def test_lookup_returns_the_best_match_and_sends_the_key_only_as_a_header(caplog):
    session = FakeScholarSession()
    lookup = make_lookup(session, timeout=7.5, api_key="s2-secret-key")
    with caplog.at_level(logging.DEBUG):
        found = lookup.find("Deep Residual Learning")
        missing = lookup.find("A Paper Nobody Indexed")

    assert found == {
        "fetched_paper_id": "id-resnet",
        "fetched_paper_title": "DEEP RESIDUAL LEARNING",
    }
    assert missing is None
    call = session.calls[0]
    assert call["url"] == S2_SEARCH_URL
    assert call["headers"] == {"X-API-KEY": "s2-secret-key"}
    assert call["params"] == {
        "query": "Deep Residual Learning",
        "limit": 1,
        "fields": "paperId,title",
    }
    assert call["timeout"] == 7.5
    assert "s2-secret-key" not in caplog.text
    assert "s2-secret-key" not in repr(lookup)


def test_lookup_reads_the_key_from_the_environment_at_call_time(monkeypatch):
    session = FakeScholarSession()
    lookup = make_lookup(session)
    lookup.find("Adam")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "late-key")
    lookup.find("Adam")
    assert [call["headers"] for call in session.calls] == [{}, {"X-API-KEY": "late-key"}]


def test_lookup_skips_blank_titles_without_a_request():
    session = FakeScholarSession()
    assert make_lookup(session).find("   ") is None
    assert session.calls == []


def test_rate_limit_is_retried_with_growing_backoff_then_succeeds():
    session = FakeScholarSession([FakeResponse(429), FakeResponse(503), FakeResponse(429)])
    clock = Clock()
    lookup = make_lookup(session, clock, base_delay=2.0, max_delay=60.0)

    assert lookup.find("Adam") == {"fetched_paper_id": "id-adam", "fetched_paper_title": "ADAM"}
    assert len(session.calls) == 4
    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_rate_limit_retries_are_bounded_and_end_in_an_error(caplog):
    """The old implementation recursed on HTTP 429 without any limit."""
    session = FakeScholarSession([FakeResponse(429) for _ in range(50)])
    clock = Clock()
    lookup = make_lookup(
        session, clock, max_attempts=4, base_delay=1.0, max_delay=3.0, api_key="k3y"
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(ScholarLookupError, match="HTTP 429"):
        lookup.find("Adam")
    assert len(session.calls) == 4
    assert clock.sleeps == [1.0, 2.0, 3.0]  # capped at max_delay, no sleep after the last try
    assert "k3y" not in caplog.text


def test_retry_after_header_stretches_the_backoff_up_to_the_cap():
    session = FakeScholarSession(
        [
            FakeResponse(429, headers={"Retry-After": "9"}),
            FakeResponse(429, headers={"Retry-After": "500"}),
        ]
    )
    clock = Clock()
    lookup = make_lookup(session, clock, base_delay=1.0, max_delay=30.0)
    assert lookup.find("Dropout") is not None
    assert clock.sleeps == [9.0, 30.0]


def test_network_errors_and_bad_json_are_retried_then_raised():
    session = FakeScholarSession(
        [requests.ConnectionError("connection reset"), FakeResponse(200, ValueError("bad json"))]
    )
    lookup = make_lookup(session)
    assert lookup.find("ImageNet")["fetched_paper_id"] == "id-imagenet"
    assert len(session.calls) == 3

    failing = FakeScholarSession([requests.Timeout("timed out") for _ in range(10)])
    with pytest.raises(ScholarLookupError, match="Timeout"):
        make_lookup(failing, max_attempts=3).find("ImageNet")
    assert len(failing.calls) == 3


def test_auth_errors_fail_immediately_and_other_client_errors_count_as_not_found(caplog):
    denied = FakeScholarSession([FakeResponse(403)])
    with pytest.raises(ScholarLookupError, match="SEMANTIC_SCHOLAR_API_KEY"):
        make_lookup(denied).find("Adam")
    assert len(denied.calls) == 1

    bad_request = FakeScholarSession([FakeResponse(400)])
    with caplog.at_level(logging.WARNING):
        assert make_lookup(bad_request).find("Adam") is None
    assert len(bad_request.calls) == 1
    assert "HTTP 400" in caplog.text


@pytest.mark.parametrize(
    "payload", [{"data": {"paperId": "x"}}, {"data": "oops"}, {"data": [None]}, [], None]
)
def test_an_unexpected_search_payload_counts_as_no_match(payload):
    session = FakeScholarSession([FakeResponse(200, payload)])
    assert make_lookup(session).find("Adam") is None
    assert len(session.calls) == 1


def test_requests_are_spaced_by_the_minimum_interval():
    session = FakeScholarSession()
    clock = Clock()
    lookup = make_lookup(session, clock, min_interval=1.5)
    for title in ("Adam", "Dropout", "ImageNet"):
        lookup.find(title)
    # The first request goes out at once; each later one waits for its slot.
    assert clock.sleeps == [1.5, 1.5]


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def test_reference_extraction_returns_text_or_empty_for_a_paper_without_references():
    fake = fake_judge()
    assert extract_reference_from_pdf_text("ORIGINAL body", MODEL, client=fake) == ORIGINAL_REFS
    assert extract_reference_from_pdf_text("NOREFS body", MODEL, client=fake) == ""
    request = fake.requests[0]
    assert "<paper_text>\nORIGINAL body\n</paper_text>" in request.text
    assert "References" in request.system


def test_an_answer_that_is_not_a_reference_list_is_a_failed_extraction():
    """Without this, an apology from the model scored the paper 0 as if it cited nothing."""
    fake = fake_judge(**{"eval.reference_extraction": "Sorry, I can only see part of the paper."})
    retry, sleeps = instant_retry(attempts=3)
    with pytest.raises(LLMParseError):
        extract_reference_from_pdf_text("GENERATED body", MODEL, client=fake, retry=retry)
    assert fake.labels() == ["eval.reference_extraction"] * 3 and len(sleeps) == 2

    # The sentinel next to an actual list does not discard the list.
    listed = fake_judge(**{"eval.reference_extraction": f"{NO_REFERENCES}\n[1] A | Adam | X\n"})
    assert extract_reference_from_pdf_text("body", MODEL, client=listed) == (
        f"{NO_REFERENCES}\n[1] A | Adam | X"
    )


def test_reference_extraction_raises_after_bounded_retries_on_blank_output(tmp_path):
    """The old helper swallowed every failure and returned None."""
    fake = fake_judge(**{"eval.reference_extraction": "   "})
    retry, sleeps = instant_retry(attempts=3)
    pdf = make_pdf(tmp_path / "paper.pdf", "ORIGINAL")
    with pytest.raises(LLMParseError):
        get_paper_references_from_pdf(pdf, MODEL, client=fake, retry=retry)
    assert fake.labels() == ["eval.reference_extraction"] * 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]

    with pytest.raises(FileNotFoundError, match=r"missing\.pdf"):
        get_paper_references_from_pdf(tmp_path / "missing.pdf", MODEL, client=fake)
    empty = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(empty)
    with pytest.raises(PdfError):
        get_paper_references_from_pdf(empty, MODEL, client=fake)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def test_process_text_to_data_resolves_ids_and_splits_priorities(tmp_path):
    fake = fake_judge()
    session = FakeScholarSession()
    evaluator = ReferenceF1V1Evaluator(
        log_folder=str(tmp_path), model_name=MODEL, client=fake, scholar=make_lookup(session)
    )
    data = evaluator.process_text_to_data(ORIGINAL_REFS, tag="gt", paper_text="the paper body")

    assert [item["extracted_paper_title"] for item in data["citation_info"]] == [
        "Deep Residual Learning",
        "Attention Is All You Need",
        "ImageNet",
        "Adam",
        "A Paper Nobody Indexed",
    ]
    assert data["citation_info"][0] == {
        "citation_text": "He et al. | Deep Residual Learning | CVPR 2016",
        "extracted_paper_title": "Deep Residual Learning",
        "fetched_paper_id": "id-resnet",
        "fetched_paper_title": "DEEP RESIDUAL LEARNING",
        "priority": "P0",
    }
    # The unindexed reference keeps its priority but contributes no id.
    assert data["citation_info"][4]["fetched_paper_id"] == "gt_NA"
    assert data["citation_info"][4]["fetched_paper_title"] == "gt_NA"
    assert data["all_ids"] == ["id-adam", "id-imagenet", "id-resnet", "id-transformer"]
    assert data["p0_ids"] == ["id-imagenet", "id-resnet"]
    assert data["p1_ids"] == ["id-adam", "id-transformer"]
    assert (data["all_ids_count"], data["p0_ids_count"], data["p1_ids_count"]) == (4, 2, 2)

    # The priority prompt carries the paper and the renumbered reference list.
    priority_request = next(r for r in fake.requests if r.label == "eval.citation_priorities")
    assert "<paper_text>\nthe paper body\n</paper_text>" in priority_request.text
    assert "[5] Nobody | A Paper Nobody Indexed | arXiv 2020" in priority_request.text
    assert '"1": "P0"' in priority_request.system


def test_p0_wins_when_the_same_paper_is_cited_twice_and_missing_labels_default_to_p1(caplog):
    refs = "[1] A | Adam | X [2] B | Adam | Y [3] C | Dropout | Z"
    fake = fake_judge(priorities={"1": "P1", "2": "P0"})
    evaluator = ReferenceF1V1Evaluator(
        model_name=MODEL, client=fake, scholar=make_lookup(FakeScholarSession())
    )
    with caplog.at_level(logging.WARNING):
        data = evaluator.process_text_to_data(refs, tag="gt", paper_text="body")
    assert data["p0_ids"] == ["id-adam"]
    assert data["p1_ids"] == ["id-dropout"]
    assert [item["priority"] for item in data["citation_info"]] == ["P1", "P0", "P1"]
    assert "1 of 3 reference(s) without a priority" in caplog.text


def test_an_answer_that_labels_only_a_few_references_is_not_a_classification():
    """Filling the gaps with P1 would cache a mostly failed answer as the ground truth."""
    fake = fake_judge(priorities={"1": "P0", "2": "P1", "77": "P0"})  # 2 of the 5 references
    retry, sleeps = instant_retry(attempts=3)
    evaluator = ReferenceF1V1Evaluator(
        model_name=MODEL, client=fake, scholar=make_lookup(FakeScholarSession()), retry=retry
    )
    with pytest.raises(LLMParseError, match="labelled only 2 of 5 references"):
        evaluator.process_text_to_data(ORIGINAL_REFS, tag="gt", paper_text="body")
    assert fake.labels().count("eval.citation_priorities") == 3 and len(sleeps) == 2


def test_without_paper_text_no_priority_call_is_made_and_everything_is_p1():
    fake = fake_judge()
    evaluator = ReferenceF1V1Evaluator(
        model_name=MODEL, client=fake, scholar=make_lookup(FakeScholarSession())
    )
    data = evaluator.process_text_to_data(GENERATED_REFS, tag="gen")
    assert "eval.citation_priorities" not in fake.labels()
    assert data["p0_ids"] == []
    assert data["p1_ids"] == ["id-dropout", "id-imagenet", "id-resnet"]


def test_evaluate_from_data_computes_the_metrics_and_writes_the_reports(tmp_path):
    evaluator = ReferenceF1V1Evaluator(log_folder=str(tmp_path / "logs"), model_name=MODEL)
    gt = {"all_ids": ["a", "b", "c", "d"], "p0_ids": ["a", "b"], "p1_ids": ["c", "d"]}
    gen = {"all_ids": ["a", "c", "x", "gen_NA", ""]}
    metrics = evaluator.evaluate_from_data(gt, gen)

    assert metrics["overall_precision"] == pytest.approx(2 / 3)
    assert metrics["overall_recall"] == pytest.approx(2 / 4)
    assert metrics["overall_f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert metrics["p0_recall"] == pytest.approx(0.5)
    assert metrics["p1_recall"] == pytest.approx(0.5)
    assert metrics["counts"] == {
        "ground_truth_ids_all": 4,
        "ground_truth_ids_p0": 2,
        "ground_truth_ids_p1": 2,
        "generated_ids_all": 3,
        "matches_overall": 2,
        "matches_p0": 1,
        "matches_p1": 1,
    }
    logs = tmp_path / "logs"
    assert json.loads((logs / "metrics.json").read_text()) == metrics
    assert json.loads((logs / "gt_ref_info.json").read_text()) == gt
    assert json.loads((logs / "gen_ref_info.json").read_text()) == gen


def test_evaluate_from_data_handles_empty_sides_without_dividing_by_zero(tmp_path):
    evaluator = ReferenceF1V1Evaluator(log_folder=str(tmp_path), model_name=MODEL)
    metrics = evaluator.evaluate_from_data({"all_ids": []}, {"all_ids": []})
    assert [metrics[key] for key in ("overall_precision", "overall_recall", "overall_f1")] == [
        0.0
    ] * 3
    assert metrics["p0_recall"] == 0.0 and metrics["p1_recall"] == 0.0


@pytest.mark.parametrize("legacy_key", ["citation_info", "pairwise"])
def test_evaluate_from_data_reads_caches_written_by_older_versions(tmp_path, legacy_key):
    evaluator = ReferenceF1V1Evaluator(log_folder=str(tmp_path), model_name=MODEL)
    gt = {
        legacy_key: [
            {"fetched_paper_id": "a", "priority": "P1"},
            {"fetched_paper_id": "a", "priority": "P0"},
            {"fetched_paper_id": "b", "priority": "P1"},
            {"fetched_paper_id": "gt_NA", "priority": "P0"},
        ]
    }
    metrics = evaluator.evaluate_from_data(gt, {"all_ids": ["a"]})
    assert metrics["counts"]["ground_truth_ids_all"] == 2
    assert metrics["counts"]["ground_truth_ids_p0"] == 1
    assert metrics["counts"]["ground_truth_ids_p1"] == 1
    assert metrics["p0_recall"] == 1.0 and metrics["p1_recall"] == 0.0


def test_default_log_folder_is_created_only_when_reports_are_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator = ReferenceF1V1Evaluator(model_name=MODEL)
    assert not (tmp_path / "logs").exists()
    evaluator.evaluate_from_data({"all_ids": ["a"]}, {"all_ids": ["a"]})
    folder = Path(evaluator.log_folder)
    assert folder.name.startswith("citation_autorater_log_")
    assert folder.parent.parent == Path("logs/paper_writing")
    assert (folder / "metrics.json").is_file()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_get_citation_f1_metrics_end_to_end_through_the_router_and_reuses_the_cache(tmp_path):
    """A Claude model id resolves through the router; no provider-specific branch is involved."""
    fake = fake_judge()
    register_client("anthropic", fake)
    session = FakeScholarSession()
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    generated = make_pdf(tmp_path / "generated.pdf", "GENERATED")

    metrics = get_citation_f1_metrics(
        str(original),
        str(generated),
        log_folder=str(tmp_path / "logs"),
        model_name=MODEL,
        scholar=make_lookup(session),
    )
    # Ground truth: resnet (P0), transformer (P1), imagenet (P0), adam (P1).
    # Generated: resnet, dropout, imagenet.
    assert metrics["overall_precision"] == pytest.approx(2 / 3)
    assert metrics["overall_recall"] == pytest.approx(2 / 4)
    assert metrics["p0_recall"] == 1.0
    assert metrics["p1_recall"] == 0.0
    assert metrics["counts"]["matches_overall"] == 2
    assert {request.model for request in fake.requests} == {MODEL}

    cache = tmp_path / f"original_gt_citations_{MODEL}.json"
    cached = json.loads(cache.read_text())
    assert cached["p0_ids"] == ["id-imagenet", "id-resnet"]
    assert json.loads((tmp_path / "logs" / "gt_ref_info.json").read_text()) == cached

    # Second evaluation: the ground truth comes from the cache, only the generated paper is processed.
    first_run_requests = len(fake.requests)
    again = get_citation_f1_metrics(
        str(original),
        str(generated),
        log_folder=str(tmp_path / "logs"),
        model_name=MODEL,
        scholar=make_lookup(session),
    )
    assert again == metrics
    new_labels = fake.labels()[first_run_requests:]
    assert "eval.citation_priorities" not in new_labels
    assert new_labels.count("eval.reference_extraction") == 1
    assert new_labels.count("eval.title_extraction") == 3


def test_failed_priority_classification_raises_and_caches_nothing(tmp_path):
    """The old code turned this failure into 'everything is P1' and cached it for good."""
    fake = fake_judge(priorities="I cannot classify these.")
    retry, sleeps = instant_retry(attempts=3)
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    generated = make_pdf(tmp_path / "generated.pdf", "GENERATED")

    with pytest.raises(LLMParseError) as excinfo:
        get_citation_f1_metrics(
            str(original),
            str(generated),
            log_folder=str(tmp_path / "logs"),
            model_name=MODEL,
            client=fake,
            scholar=make_lookup(FakeScholarSession()),
            retry=retry,
        )
    assert fake.labels().count("eval.citation_priorities") == 3
    assert len(sleeps) == 2
    assert "failed after 3 attempt(s)" in "".join(excinfo.value.__notes__)
    assert not list(tmp_path.glob("*_gt_citations_*.json"))
    assert not (tmp_path / "logs" / "metrics.json").exists()


def test_scholar_outage_fails_the_evaluation_instead_of_scoring_zero(tmp_path):
    fake = fake_judge()
    session = FakeScholarSession([FakeResponse(500) for _ in range(100)])
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    generated = make_pdf(tmp_path / "generated.pdf", "GENERATED")
    with pytest.raises(ScholarLookupError):
        get_citation_f1_metrics(
            str(original),
            str(generated),
            log_folder=str(tmp_path / "logs"),
            model_name=MODEL,
            client=fake,
            scholar=make_lookup(session, max_attempts=2),
        )
    assert not list(tmp_path.glob("*_gt_citations_*.json"))


def test_original_without_references_is_an_error_but_a_generated_one_scores_zero(tmp_path, caplog):
    fake = fake_judge()
    scholar = make_lookup(FakeScholarSession())
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    bare = make_pdf(tmp_path / "bare.pdf", "NOREFS")

    with pytest.raises(ValueError, match="No numbered references"):
        get_citation_f1_metrics(
            str(bare),
            str(original),
            log_folder=str(tmp_path),
            model_name=MODEL,
            client=fake,
            scholar=scholar,
        )
    assert not list(tmp_path.glob("bare_gt_citations_*.json"))

    with caplog.at_level(logging.WARNING):
        metrics = get_citation_f1_metrics(
            str(original),
            str(bare),
            log_folder=str(tmp_path),
            model_name=MODEL,
            client=fake,
            scholar=scholar,
        )
    assert metrics["overall_precision"] == 0.0 and metrics["overall_recall"] == 0.0
    assert metrics["counts"]["generated_ids_all"] == 0
    assert metrics["counts"]["ground_truth_ids_all"] == 4
    assert "no extractable references" in caplog.text

    with pytest.raises(FileNotFoundError, match=r"nowhere\.pdf"):
        get_citation_f1_metrics(
            str(tmp_path / "nowhere.pdf"), str(bare), model_name=MODEL, client=fake
        )


def test_failed_extraction_of_the_generated_paper_is_an_error_not_a_zero_score(tmp_path):
    def references(request: FakeRequest) -> str:
        return ORIGINAL_REFS if "ORIGINAL" in request.text else "I could not find the paper text."

    fake = fake_judge(**{"eval.reference_extraction": references})
    retry, _ = instant_retry(attempts=2)
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    generated = make_pdf(tmp_path / "generated.pdf", "GENERATED")
    with pytest.raises(LLMParseError):
        get_citation_f1_metrics(
            str(original),
            str(generated),
            log_folder=str(tmp_path / "logs"),
            model_name=MODEL,
            client=fake,
            scholar=make_lookup(FakeScholarSession()),
            retry=retry,
        )
    assert not (tmp_path / "logs" / "metrics.json").exists()


def test_a_ground_truth_without_any_resolved_reference_is_refused(tmp_path):
    """Scoring against it would give recall 0 for every paper, and it would be cached."""
    nothing_indexed = FakeScholarSession(
        [FakeResponse(200, {"total": 0, "data": []}) for _ in range(5)]
    )
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    generated = make_pdf(tmp_path / "generated.pdf", "GENERATED")
    with pytest.raises(ValueError, match="None of the 5 reference"):
        get_citation_f1_metrics(
            str(original),
            str(generated),
            log_folder=str(tmp_path / "logs"),
            model_name=MODEL,
            client=fake_judge(),
            scholar=make_lookup(nothing_indexed),
        )
    assert not list(tmp_path.glob("*_gt_citations_*.json"))


@pytest.mark.parametrize(
    "poisoned",
    [
        # What the old code cached when reference extraction failed ...
        {"citation_info": [], "all_ids": [], "all_ids_count": 0, "p0_ids": [], "p1_ids": []},
        # ... and when every Semantic Scholar lookup failed.
        {
            "citation_info": [{"fetched_paper_id": "gt_NA", "priority": "P1"}],
            "all_ids": [],
            "p0_ids": [],
            "p1_ids": [],
        },
        {"citation_info": "not a list"},
        {},
    ],
)
def test_a_ground_truth_cache_left_by_a_failed_run_is_rebuilt(tmp_path, caplog, poisoned):
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    generated = make_pdf(tmp_path / "generated.pdf", "GENERATED")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(poisoned))
    with caplog.at_level(logging.WARNING):
        metrics = get_citation_f1_metrics(
            str(original),
            str(generated),
            log_folder=str(tmp_path / "logs"),
            gt_cache_path=str(cache),
            model_name=MODEL,
            client=fake_judge(),
            scholar=make_lookup(FakeScholarSession()),
        )
    assert "Ignoring the ground-truth cache" in caplog.text
    assert metrics["counts"]["ground_truth_ids_all"] == 4
    assert metrics["overall_recall"] == pytest.approx(0.5)
    assert json.loads(cache.read_text())["all_ids_count"] == 4


def test_a_ground_truth_without_must_have_references_is_flagged(tmp_path, caplog):
    """A failed classification used to be cached as 'everything is P1': P0 recall 0 for good."""
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    generated = make_pdf(tmp_path / "generated.pdf", "GENERATED")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"all_ids": ["id-resnet"], "p0_ids": [], "p1_ids": ["id-resnet"]}))
    with caplog.at_level(logging.WARNING):
        metrics = get_citation_f1_metrics(
            str(original),
            str(generated),
            log_folder=str(tmp_path / "logs"),
            gt_cache_path=str(cache),
            model_name=MODEL,
            client=fake_judge(),
            scholar=make_lookup(FakeScholarSession()),
        )
    assert metrics["p0_recall"] == 0.0 and metrics["p1_recall"] == 1.0
    assert "no must-have (P0) reference" in caplog.text


def test_model_ids_with_path_characters_do_not_end_up_in_file_paths(tmp_path):
    """'anthropic/claude-...' used to turn the cache and results names into sub-directories."""
    model = "anthropic/claude-fable-5-2:latest"
    project = make_project(tmp_path / "runs", ["p1"])
    original = project / "p1" / "raw_materials" / "original_paper.pdf"
    fake = fake_judge()
    scholar = make_lookup(FakeScholarSession())

    results = review_citation_f1_batch(str(project), model_name=model, client=fake, scholar=scholar)
    safe = "anthropic_claude-fable-5-2_latest"
    assert (project / f"citation_f1_results_{safe}.json").is_file()
    assert (original.parent / f"original_paper_gt_citations_{safe}.json").is_file()
    assert results["p1"]["metrics"]["overall_precision"] == pytest.approx(2 / 3)

    get_citation_f1_metrics(
        str(original),
        str(project / "p1" / "final_paper.pdf"),
        log_folder=str(tmp_path / "logs"),
        model_name=model,
        client=fake,
        scholar=scholar,
    )
    assert (original.parent / f"original_paper_gt_citations_{safe}.json").is_file()
    assert (original.parent / f"original_paper_gt_citations_{safe}.json").parent == original.parent
    assert not [path for path in project.rglob("*") if path.is_dir() and "anthropic" in path.name]


def test_a_damaged_ground_truth_cache_is_rebuilt(tmp_path, caplog):
    fake = fake_judge()
    original = make_pdf(tmp_path / "original.pdf", "ORIGINAL")
    generated = make_pdf(tmp_path / "generated.pdf", "GENERATED")
    cache = tmp_path / "cache.json"
    cache.write_text("{ truncated")
    with caplog.at_level(logging.WARNING):
        get_citation_f1_metrics(
            str(original),
            str(generated),
            log_folder=str(tmp_path / "logs"),
            gt_cache_path=str(cache),
            model_name=MODEL,
            client=fake,
            scholar=make_lookup(FakeScholarSession()),
        )
    assert "Ignoring the ground-truth cache" in caplog.text
    assert json.loads(cache.read_text())["all_ids_count"] == 4


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def make_project(root: Path, paper_ids: list[str], *, with_original: bool = True) -> Path:
    for paper_id in paper_ids:
        make_pdf(root / paper_id / "final_paper.pdf", "GENERATED")
        if with_original:
            make_pdf(root / paper_id / "raw_materials" / "original_paper.pdf", "ORIGINAL")
    return root


def test_review_batch_writes_per_paper_and_overall_metrics_and_resumes(tmp_path, caplog):
    project = make_project(tmp_path / "runs", ["p1", "p2"])
    make_pdf(project / "p3" / "final_paper.pdf", "GENERATED")  # has no original paper anywhere
    (project / "notes.txt").write_text("not a paper folder")
    fake = fake_judge()
    scholar = make_lookup(FakeScholarSession())

    with caplog.at_level(logging.ERROR):
        results = review_citation_f1_batch(
            str(project), max_workers=2, model_name=MODEL, client=fake, scholar=scholar
        )

    results_path = project / f"citation_f1_results_{MODEL}.json"
    assert json.loads(results_path.read_text()) == results
    assert sorted(results) == ["overall_metrics", "p1", "p2"]
    entry = results["p1"]
    assert entry["paper_id"] == "p1"
    assert entry["real_pdf_path"] == str(project / "p1" / "raw_materials" / "original_paper.pdf")
    assert entry["generated_pdf_path"] == str(project / "p1" / "final_paper.pdf")
    assert entry["metrics"]["overall_precision"] == pytest.approx(2 / 3)
    assert results["overall_metrics"]["overall_recall"] == pytest.approx(0.5)
    assert results["overall_metrics"]["counts"]["generated_ids_all"] == 3.0
    # Per-paper reports and the ground-truth cache sit in the paper folder.
    assert (project / "p1" / "citation_f1_metrics" / "metrics.json").is_file()
    assert (
        project / "p1" / "raw_materials" / f"original_paper_gt_citations_{MODEL}.json"
    ).is_file()
    # The paper without an original is reported, not silently dropped, and not recorded.
    assert "citation F1 failed for p3" in caplog.text
    assert "original paper not found" in caplog.text

    requests_before = len(fake.requests)
    again = review_citation_f1_batch(
        str(project), max_workers=2, model_name=MODEL, client=fake, scholar=scholar
    )
    assert again["p1"] == results["p1"]
    assert len(fake.requests) == requests_before  # p1 and p2 are skipped; p3 fails before any call


def test_review_batch_in_which_every_paper_fails_is_an_error(tmp_path, caplog):
    """It used to return (and save) an empty result that looked like a finished evaluation."""
    project = make_project(tmp_path / "runs", ["p1", "p2"])
    fake = fake_judge(**{"eval.reference_extraction": "   "})
    retry, _ = instant_retry(attempts=2)
    with caplog.at_level(logging.ERROR), pytest.raises(BatchFailedError) as excinfo:
        review_citation_f1_batch(
            str(project),
            max_workers=2,
            model_name=MODEL,
            client=fake,
            scholar=make_lookup(FakeScholarSession()),
            retry=retry,
        )
    assert "failed for all 2 item(s)" in str(excinfo.value)
    assert "LLMParseError" in str(excinfo.value)
    assert "citation F1 failed for p1" in caplog.text
    assert not (project / f"citation_f1_results_{MODEL}.json").exists()


def test_review_batch_falls_back_to_the_dataset_copy_of_the_original(tmp_path):
    """The old fallback path was a plain string with '{paper_id}' left in it."""
    project = make_project(tmp_path / "cvpr2025_runs", ["0042"], with_original=False)
    datasets = tmp_path / "datasets"
    original = make_pdf(datasets / "cvpr2025" / "papers" / "0042" / "0042.pdf", "ORIGINAL")
    output = tmp_path / "out"

    results = review_citation_f1_batch(
        str(project),
        output_folder=str(output),
        model_name=MODEL,
        dataset_dir=datasets,
        client=fake_judge(),
        scholar=make_lookup(FakeScholarSession()),
    )
    assert results["0042"]["real_pdf_path"] == str(original)
    assert (output / f"citation_f1_results_{MODEL}.json").is_file()
    assert not (project / f"citation_f1_results_{MODEL}.json").exists()


def test_review_batch_stops_at_once_on_a_configuration_error(tmp_path):
    project = make_project(tmp_path / "runs", ["p1"])
    with pytest.raises(LLMConfigError, match="Cannot tell which provider"):
        review_citation_f1_batch(str(project), model_name="mystery-model-1")
    with pytest.raises(FileNotFoundError):
        review_citation_f1_batch(str(tmp_path / "absent"), model_name=MODEL, client=fake_judge())


def test_generate_gt_citations_batch_writes_caches_progress_and_info(tmp_path, caplog):
    base = tmp_path / "cvpr2025"
    for paper in ("0001", "0002"):
        make_pdf(base / paper / f"{paper}.pdf", "ORIGINAL")
        (base / paper / "raw_materials").mkdir()
    make_pdf(base / "0003" / "0003.pdf", "ORIGINAL")  # no raw_materials subfolder: not a target
    (base / "0004" / "raw_materials").mkdir(parents=True)  # no PDF: reported as a failure
    info = [{"paper_id": "0001", "title": "t1"}, {"paper_id": "0003"}, {"paper_id": "0004"}]
    (base / "info.json").write_text(json.dumps(info))
    progress_path = tmp_path / "progress.json"
    fake = fake_judge()

    with caplog.at_level(logging.ERROR):
        progress = generate_gt_citations_batch(
            str(base),
            "raw_materials",
            progress_json_path=str(progress_path),
            max_workers=2,
            model_name=MODEL,
            client=fake,
            scholar=make_lookup(FakeScholarSession()),
        )

    out1 = base / "0001" / "raw_materials" / f"original_paper_gt_citations_{MODEL}.json"
    out2 = base / "0002" / "raw_materials" / f"original_paper_gt_citations_{MODEL}.json"
    assert json.loads(out1.read_text())["p0_ids"] == ["id-imagenet", "id-resnet"]
    assert progress == json.loads(progress_path.read_text())
    assert progress == {
        "processed_count": 2,
        "successful_papers": {"0001": str(out1), "0002": str(out2)},
    }
    assert json.loads((base / "info.json").read_text()) == [
        {"paper_id": "0001", "title": "t1", "num_gt_citations_p0": 2, "num_gt_citations_p1": 2},
        {"paper_id": "0003"},
        {"paper_id": "0004"},
    ]
    assert "failed for 0004" in caplog.text and "no PDF found" in caplog.text

    # A second run reuses the caches (no model calls) and still reports them as done.
    requests_before = len(fake.requests)
    again = generate_gt_citations_batch(
        str(base),
        "raw_materials",
        progress_json_path=str(progress_path),
        model_name=MODEL,
        client=fake,
        scholar=make_lookup(FakeScholarSession()),
    )
    assert again["processed_count"] == 2
    assert len(fake.requests) == requests_before


def test_generate_gt_citations_batch_rebuilds_poisoned_files_and_fails_when_nothing_works(
    tmp_path, caplog
):
    base = tmp_path / "cvpr2025"
    make_pdf(base / "0001" / "0001.pdf", "ORIGINAL")
    target = base / "0001" / "raw_materials" / f"original_paper_gt_citations_{MODEL}.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"citation_info": [], "all_ids": [], "p0_ids_count": 0}))

    with caplog.at_level(logging.WARNING):
        progress = generate_gt_citations_batch(
            str(base),
            "raw_materials",
            model_name=MODEL,
            client=fake_judge(),
            scholar=make_lookup(FakeScholarSession()),
        )
    assert "Ignoring the ground-truth cache" in caplog.text
    assert json.loads(target.read_text())["all_ids_count"] == 4
    assert progress["successful_papers"] == {"0001": str(target)}

    target.unlink()
    retry, _ = instant_retry(attempts=2)
    with pytest.raises(BatchFailedError, match="ground-truth citation extraction failed for all 1"):
        generate_gt_citations_batch(
            str(base),
            "raw_materials",
            model_name=MODEL,
            client=fake_judge(priorities="no labels, sorry"),
            scholar=make_lookup(FakeScholarSession()),
            retry=retry,
        )
    assert not target.exists()


def test_generate_gt_citations_batch_leaves_an_unreadable_info_json_alone(tmp_path, caplog):
    """The old code replaced an info.json it could not parse with an empty list."""
    base = tmp_path / "iclr2025"
    make_pdf(base / "0001" / "0001.pdf", "ORIGINAL")
    (base / "0001" / "raw").mkdir()
    (base / "info.json").write_text("[{broken")
    with caplog.at_level(logging.WARNING):
        generate_gt_citations_batch(
            str(base),
            "raw",
            model_name=MODEL,
            client=fake_judge(),
            scholar=make_lookup(FakeScholarSession()),
        )
    assert (base / "info.json").read_text() == "[{broken"
    assert "not valid JSON" in caplog.text
    with pytest.raises(FileNotFoundError):
        generate_gt_citations_batch(
            str(tmp_path / "absent"), "raw", model_name=MODEL, client=fake_judge()
        )


def test_analyze_citation_metrics_averages_scores_and_counts(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "a": {
                    "metrics": {
                        "overall_precision": 1.0,
                        "overall_f1": 0.5,
                        "counts": {"matches_overall": 4},
                    }
                },
                "b": {
                    "metrics": {
                        "overall_precision": 0.0,
                        "overall_f1": 0.25,
                        "counts": {"matches_overall": 1},
                    }
                },
                "c": {"paper_id": "c"},
                "overall_metrics": {"overall_precision": 123.0},
            }
        )
    )
    overall = analyze_citation_metrics(str(path))
    assert overall["overall_precision"] == 0.5
    assert overall["overall_f1"] == 0.375
    assert overall["overall_recall"] == 0.0
    assert overall["counts"]["matches_overall"] == 2.5
    assert json.loads(path.read_text())["overall_metrics"] == overall

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    assert analyze_citation_metrics(str(empty)) is None
    assert json.loads(empty.read_text()) == {}
