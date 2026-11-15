from __future__ import annotations

import pytest

from paper_orchestra.llm.base import Usage
from paper_orchestra.llm.pricing import (
    PRICES,
    WEB_SEARCH_USD,
    ModelPrice,
    estimate_cost_usd,
    price_for,
)

MILLION = 1_000_000


def test_price_table_matches_the_published_rows():
    assert PRICES["claude-fable-5-2"] == ModelPrice(8.00, 40.00, 0.20, 10.00)
    assert PRICES["claude-opus-5-5"] == ModelPrice(4.00, 20.00, 0.20, 5.00)
    assert PRICES["claude-sonnet-5-5"] == ModelPrice(2.00, 10.00, 0.10, 2.50)
    assert PRICES["claude-haiku-5-5"] == ModelPrice(0.10, 0.50, 0.01, 0.125)
    # The models a declined request can fall back to.
    assert (
        PRICES["claude-opus-5"] == PRICES["claude-opus-4-8"] == ModelPrice(5.00, 25.00, 0.50, 6.25)
    )


def test_each_token_class_is_priced_at_its_own_rate():
    assert estimate_cost_usd("claude-fable-5-2", Usage(input_tokens=MILLION)) == pytest.approx(8.0)
    assert estimate_cost_usd("claude-fable-5-2", Usage(output_tokens=MILLION)) == pytest.approx(
        40.0
    )
    assert estimate_cost_usd(
        "claude-fable-5-2", Usage(cache_read_input_tokens=MILLION)
    ) == pytest.approx(0.20)
    assert estimate_cost_usd(
        "claude-fable-5-2", Usage(cache_creation_input_tokens=MILLION)
    ) == pytest.approx(10.0)
    assert estimate_cost_usd("claude-fable-5-2", Usage(web_searches=7)) == pytest.approx(0.07)
    assert pytest.approx(0.01) == WEB_SEARCH_USD


def test_a_realistic_request_adds_up():
    usage = Usage(
        input_tokens=12_000,
        output_tokens=30_000,
        cache_read_input_tokens=400_000,
        cache_creation_input_tokens=20_000,
        requests=6,
        web_searches=3,
    )
    expected = (12_000 * 8 + 30_000 * 40 + 400_000 * 0.2 + 20_000 * 10) / MILLION + 0.03
    assert estimate_cost_usd("claude-fable-5-2", usage) == pytest.approx(expected)
    assert estimate_cost_usd("claude-sonnet-5-5", usage) < estimate_cost_usd(
        "claude-fable-5-2", usage
    )
    assert estimate_cost_usd(" Claude-Fable-5-2 ", usage) == pytest.approx(expected)


def test_unknown_models_have_no_estimate():
    usage = Usage(input_tokens=1000, output_tokens=1000, requests=1)
    assert estimate_cost_usd("gemini-3-pro", usage) is None
    assert estimate_cost_usd("gpt-5", usage) is None
    assert estimate_cost_usd("claude-fable-9", usage) is None
    assert estimate_cost_usd("", usage) is None
    assert price_for("gpt-5") is None


def test_no_usage_costs_nothing():
    assert estimate_cost_usd("claude-fable-5-2", Usage()) == 0.0


def test_haiku_long_prompts_use_the_higher_tier():
    short = Usage(
        input_tokens=60_000, cache_read_input_tokens=40_000, output_tokens=1000, requests=1
    )
    assert estimate_cost_usd("claude-haiku-5-5", short) == pytest.approx(
        (60_000 * 0.10 + 40_000 * 0.01 + 1000 * 0.50) / MILLION
    )
    # Cache reads and writes count towards the prompt length that selects the tier.
    long = Usage(
        input_tokens=60_000,
        cache_read_input_tokens=30_000,
        cache_creation_input_tokens=20_000,
        output_tokens=1000,
        requests=1,
    )
    assert estimate_cost_usd("claude-haiku-5-5", long) == pytest.approx(
        (60_000 * 0.50 + 30_000 * 0.05 + 20_000 * 0.625 + 1000 * 2.50) / MILLION
    )
    # The tier is per request: the same tokens spread over many requests stay cheap.
    spread = Usage(input_tokens=300_000, output_tokens=1000, requests=10)
    assert price_for("claude-haiku-5-5", spread) == PRICES["claude-haiku-5-5"]
    # Other models have a single tier.
    assert price_for("claude-fable-5-2", long) == PRICES["claude-fable-5-2"]
