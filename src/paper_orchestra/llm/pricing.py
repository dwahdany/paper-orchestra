"""Cost estimates from token usage.

Prices are USD per million tokens on the first-party Claude API.  They are an
estimate for run reports, not an invoice: a turn served by a server-side
fallback model is billed at that model's rate, while :class:`Usage` only
carries totals and is priced at the rate of the model that was asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

from paper_orchestra.llm.base import Usage

#: Web search is billed per search on top of the tokens its results consume.
WEB_SEARCH_USD = 10.0 / 1000

#: Claude Haiku 5.5 switches to its long-prompt rates above this prompt size.
HAIKU_LONG_PROMPT_TOKENS = 100_000


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float
    #: Five-minute cache writes, the only kind the Anthropic client makes.
    cache_write: float


PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-2": ModelPrice(input=8.00, output=40.00, cache_read=0.20, cache_write=10.00),
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.10, cache_write=2.50),
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
    # Server-side fallback targets; a five-minute cache write costs 1.25x input.
    "claude-opus-5": ModelPrice(input=5.00, output=25.00, cache_read=0.50, cache_write=6.25),
    "claude-opus-4-8": ModelPrice(input=5.00, output=25.00, cache_read=0.50, cache_write=6.25),
}

#: Rates that replace the base row when the prompt exceeds the model's threshold.
LONG_PROMPT_PRICES: dict[str, tuple[int, ModelPrice]] = {
    "claude-haiku-5-5": (
        HAIKU_LONG_PROMPT_TOKENS,
        ModelPrice(input=0.50, output=2.50, cache_read=0.05, cache_write=0.625),
    ),
}


def price_for(model: str, usage: Usage | None = None) -> ModelPrice | None:
    """The price row for ``model``, or ``None`` when the model is not in the table.

    Long-prompt tiers are decided per request by the API.  ``usage`` is a sum
    over requests, so the tier is chosen from the average prompt size, which is
    exact for a single request and an approximation for a mixed batch.
    """
    key = model.strip().lower()
    price = PRICES.get(key)
    if price is None or usage is None or key not in LONG_PROMPT_PRICES:
        return price
    threshold, long_price = LONG_PROMPT_PRICES[key]
    prompt_tokens = (
        usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    )
    return long_price if prompt_tokens / max(1, usage.requests) > threshold else price


def estimate_cost_usd(model: str, usage: Usage) -> float | None:
    """Estimated cost of ``usage`` on ``model`` in USD; ``None`` for an unknown model."""
    price = price_for(model, usage)
    if price is None:
        return None
    tokens = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    )
    return tokens / 1_000_000 + usage.web_searches * WEB_SEARCH_USD


def ceiling_cost_usd(usage: Usage) -> float:
    """What ``usage`` would cost at the highest rate of the table for each kind of token:
    a count for a model that has no price, where counting nothing would be worse (a
    cap on spend that must hold).  Not an estimate of what such a model costs."""
    rows = [*PRICES.values(), *(price for _, price in LONG_PROMPT_PRICES.values())]
    tokens = (
        usage.input_tokens * max(row.input for row in rows)
        + usage.output_tokens * max(row.output for row in rows)
        + usage.cache_read_input_tokens * max(row.cache_read for row in rows)
        + usage.cache_creation_input_tokens * max(row.cache_write for row in rows)
    )
    return tokens / 1_000_000 + usage.web_searches * WEB_SEARCH_USD
