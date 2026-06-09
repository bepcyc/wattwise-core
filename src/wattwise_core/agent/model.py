"""The model-routing seam: one configured OpenAI-compatible model (MODEL-R4, MODEL-R5).

This module binds the :class:`~wattwise_core.agent.contracts.ChatModel` seam to a real
provider and ships a deterministic offline fake.

:class:`OpenAICompatibleModel` is the OSS default: ONE configured OpenAI-compatible model
reached through the ``openai`` async SDK, with ``base_url``/``model``/key from
:func:`~wattwise_core.config.get_settings` (MODEL-R4 — one configured model; the OSS
engine ships no failover and no escalation). ``structured`` performs provider-enforced
JSON-schema-constrained decoding from a closed pydantic schema at temperature ``0``
(MODEL-R5 — zero temperature for verdicts and grounding-span extraction); ``compose``
runs prose at a bounded temperature (MODEL-R5). A model that cannot enforce structured
output MUST NOT back a verdict node (MODEL-R4) — this seam refuses any non-schema-valid
verdict response rather than parsing free text (STRUCT-R1).

:class:`FakeModel` is a deterministic, network-free implementation for the offline eval
suite: ``structured`` answers from a scripted ``{schema_name -> instance}`` map and
``compose`` returns canned prose. It makes NO network call, so verdict-driven tests are
reproducible (MODEL-R5, the eval-suite stability requirement).
"""

from __future__ import annotations

from openai import AsyncOpenAI
from pydantic import BaseModel

from wattwise_core.config import Settings, get_settings
from wattwise_core.observability.logging import get_logger

_VERDICT_TEMPERATURE = 0.0  # MODEL-R5: zero temperature for verdicts / span extraction.

_logger = get_logger(__name__)


class ModelResponseError(ValueError):
    """The provider returned no schema-valid structured output for a verdict call.

    Raised by :meth:`OpenAICompatibleModel.structured` when the provider refuses or
    returns an unparsed message. It subclasses :class:`ValueError` so the bounded retry
    in :func:`wattwise_core.agent.structured.run_structured` treats it as a retryable
    structured-output failure — never a free-text fallback (STRUCT-R1).
    """


class OpenAICompatibleModel:
    """The OSS default :class:`ChatModel`: one configured OpenAI-compatible model.

    All routing in OSS resolves to this single model (MODEL-R4); tier/effort selection,
    failover, and escalation are commercial and plug in through the same seam without
    touching node logic.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        cfg = settings if settings is not None else get_settings()
        self._model = cfg.agent__model
        self._compose_temperature = cfg.agent__temperature
        self._max_output_tokens = cfg.agent__max_output_tokens
        self._client = client if client is not None else _build_client(cfg)

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        """Provider-enforced JSON-schema-constrained verdict at temperature 0.

        Uses the SDK's native parse path (``response_format=schema``), which constrains
        decoding to the schema and validates the result (STRUCT-R1). The untrusted
        ``data`` region is a separate user message, never folded into ``system``
        (INJECT-R1). A refusal or an unparsed message raises :class:`ModelResponseError`
        (a ``ValueError``) so the caller's bounded retry handles it (STRUCT-R2) — there
        is no free-text fallback.
        """
        completion = await self._client.chat.completions.parse(
            model=self._model,
            temperature=_VERDICT_TEMPERATURE,
            max_completion_tokens=self._max_output_tokens,
            response_format=schema,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": data},
            ],
        )
        message = completion.choices[0].message
        if message.refusal:
            raise ModelResponseError(f"provider refused structured output for {schema.__name__!r}")
        parsed = message.parsed
        if parsed is None:
            raise ModelResponseError(f"provider returned no parseable {schema.__name__!r} object")
        return parsed

    async def compose(self, *, system: str, context: str, max_tokens: int = 1_000_000) -> str:
        """Bounded-temperature prose composition (MODEL-R5).

        Prose is not a verdict: it runs at the configured (bounded) temperature, with the
        untrusted ``context`` kept in a separate user message (INJECT-R1). Grounding (§7)
        — not this call — decides truth.
        """
        # High default = "use the full configured output budget": a node that passes no
        # max_tokens is NOT capped at 1024. A 2026 reasoning model spends output tokens on
        # its thinking trace before the answer, so a small per-call cap starved compose to
        # empty (MODEL-R5a); the budget below must hold reasoning trace + answer.
        bound = min(max_tokens, self._max_output_tokens)
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._compose_temperature,
            max_completion_tokens=bound,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
        )
        return completion.choices[0].message.content or ""


def _build_client(cfg: Settings) -> AsyncOpenAI:
    """Build the async client from config; the key is bring-your-own (MODEL-R4)."""
    key = cfg.llm_api_key.get_secret_value() if cfg.llm_api_key is not None else None
    return AsyncOpenAI(
        base_url=cfg.agent__base_url,
        api_key=key,
        timeout=cfg.agent__request_timeout_seconds,
    )


class FakeModel:
    """Deterministic, network-free :class:`ChatModel` for the offline eval suite.

    ``structured`` returns the scripted instance registered for the requested schema's
    name; ``compose`` returns the canned prose. No network is touched, so verdict-driven
    tests are reproducible (MODEL-R5). An unscripted schema raises so a test cannot
    silently pass against an absent script.
    """

    def __init__(
        self,
        *,
        scripted: dict[str, BaseModel] | None = None,
        prose: str = "",
    ) -> None:
        self._scripted: dict[str, BaseModel] = dict(scripted) if scripted else {}
        self._prose = prose

    def set_response[M: BaseModel](self, instance: M) -> None:
        """Register (or replace) the scripted instance for its schema name."""
        self._scripted[type(instance).__name__] = instance

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        """Return the scripted instance for ``schema`` (no network)."""
        instance = self._scripted.get(schema.__name__)
        if instance is None:
            raise KeyError(f"FakeModel has no scripted response for {schema.__name__!r}")
        if not isinstance(instance, schema):
            raise TypeError(
                f"scripted response for {schema.__name__!r} is {type(instance).__name__}"
            )
        return instance

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        """Return the canned prose (no network)."""
        return self._prose


__all__ = ["FakeModel", "ModelResponseError", "OpenAICompatibleModel"]
