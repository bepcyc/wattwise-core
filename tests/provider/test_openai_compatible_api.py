"""Manual provider tier: ``OpenAICompatibleModel`` against a real OpenAI-compatible REST API.

These tests pin the provider behaviour the model seam relies on (MODEL-R4/-R5/-R5a, STRUCT-R1,
AGT-OBS-R2) against a live endpoint instead of a stub: schema-constrained parsing, the
``reasoning_effort`` and ``max_completion_tokens`` parameters, ``finish_reason`` on a capped
answer, and provider-reported usage. They never run in CI; see ``conftest.py`` for the gate
and the environment variables that point them at a provider.
"""

from __future__ import annotations

from typing import cast

import pytest
from openai import AsyncOpenAI
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel

from wattwise_core.agent.model import ModelResponseError, OpenAICompatibleModel
from wattwise_core.config import Settings
from wattwise_core.observability import runtrace

pytestmark = pytest.mark.provider


class _Sentiment(BaseModel):
    label: str
    confidence: int


def _client(settings: Settings) -> AsyncOpenAI:
    key = settings.llm_api_key
    assert key is not None
    return AsyncOpenAI(
        base_url=settings.agent__base_url,
        api_key=key.get_secret_value(),
        timeout=settings.agent__request_timeout_seconds,
    )


async def test_compose_returns_prose(provider_settings: Settings) -> None:
    """Compose returns non-empty prose from the provider at the configured effort and budget."""
    model = OpenAICompatibleModel(settings=provider_settings)
    out = await model.compose(
        system="You are a concise cycling coach. Answer in one sentence.",
        context="Why do rest days matter?",
    )
    assert out.strip()


async def test_structured_output_parses_into_schema(provider_settings: Settings) -> None:
    """The provider honours ``response_format`` so the SDK parses the reply into the schema."""
    model = OpenAICompatibleModel(settings=provider_settings)
    got = await model.structured(
        system=(
            "Classify the sentiment of the user's text. label is one of positive, negative, "
            "neutral; confidence is an integer from 1 to 5."
        ),
        data="Best training block I have ever had, I feel strong.",
        schema=_Sentiment,
    )
    assert got.label in {"positive", "negative", "neutral"}
    assert 1 <= got.confidence <= 5


async def test_model_spans_record_provider_usage(provider_settings: Settings) -> None:
    """Each provider call records real prompt/completion token counts on its span."""
    model = OpenAICompatibleModel(settings=provider_settings)
    with runtrace.run_trace("provider:usage"):
        await model.compose(system="Reply with one word.", context="Say hello.")
        trace = runtrace.active_trace()
        assert trace is not None
        span = trace.spans[-1]
    assert span.model == provider_settings.agent__model
    assert span.prompt_tokens is not None and span.prompt_tokens > 0
    assert span.completion_tokens is not None and span.completion_tokens > 0


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
async def test_provider_accepts_reasoning_effort(provider_settings: Settings, effort: str) -> None:
    """Every effort label the routing policy allows is accepted by the provider (no 4xx)."""
    completion = await _client(provider_settings).chat.completions.create(
        model=provider_settings.agent__model,
        reasoning_effort=cast(ReasoningEffort, effort),
        max_completion_tokens=4096,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    assert completion.choices[0].message.content


async def test_capped_answer_reports_length_finish_reason(provider_settings: Settings) -> None:
    """A tiny output cap ends the turn with ``finish_reason == "length"``.

    The compose fail-closed path (MODEL-R5a) tells budget exhaustion from a transient empty
    answer by this field, so the provider must report it.
    """
    completion = await _client(provider_settings).chat.completions.create(
        model=provider_settings.agent__model,
        max_completion_tokens=16,
        messages=[{"role": "user", "content": "Write a 500-word essay about cycling cadence."}],
    )
    assert completion.choices[0].finish_reason == "length"


async def test_compose_fails_closed_when_budget_is_too_small(
    provider_settings: Settings,
) -> None:
    """Compose under a budget too small for the answer never returns an empty draft.

    It either raises :class:`ModelResponseError` or returns real (truncated) prose.
    """
    model = OpenAICompatibleModel(settings=provider_settings, max_output_tokens=16)
    try:
        out = await model.compose(
            system="Answer in detail.", context="Explain periodization for a marathon."
        )
    except ModelResponseError:
        return
    assert out.strip()
