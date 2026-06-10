"""Unit tests for the tier/effort routing-policy seam (MODEL-R1/-R2/-R3) and the
language-pack policy (LANG-R1/-R3/-R4).

The routing seam is exercised end to end at the node level: the compose node built by
:func:`~wattwise_core.agent.graph_model_nodes.make_compose` must reach the model the POLICY
resolves (not the default it was handed) — proving the per-node selection is real, not
decorative. The locale policy is exercised for supported/unsupported/unset requests, the
config-driven default-language fallback with its recorded observability counter (LANG-R4),
the compose-time localized variant (LANG-R3), and the localized abstain limitation copy
(GROUND-R6) resolving from CONFIG packs with the in-code floor only for the no-bundle seam.
"""

from __future__ import annotations

from typing import Any

import pytest

from wattwise_core.agent.graph_model_nodes import make_compose
from wattwise_core.agent.graph_state import limitation_text, render_context
from wattwise_core.agent.locale import EMPTY_LOCALE_POLICY, LocalePolicy
from wattwise_core.agent.tiering import (
    DifficultySignal,
    ModelTier,
    ReasoningEffort,
    RouteDecision,
    SingleModelRoutingPolicy,
    context_budget,
    estimate_tokens,
    model_token_counter,
    routed_model,
)
from wattwise_core.observability import metrics as obs_metrics

pytestmark = pytest.mark.unit


class _RecordingModel:
    """A minimal ChatModel double recording the compose system prompt it received."""

    def __init__(self) -> None:
        self.compose_calls: list[str] = []

    async def structured(self, *, system: str, data: str, schema: type) -> Any:
        raise NotImplementedError

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls.append(system)
        return "drafted"


class _EscalatingPolicy:
    """A plugged-in policy that escalates compose to ``pro`` and swaps the model (MODEL-R2)."""

    def __init__(self, escalation_model: Any) -> None:
        self._model = escalation_model

    def route(self, *, node: str, signal: DifficultySignal) -> RouteDecision:
        return RouteDecision(
            tier=ModelTier.PRO,
            effort=ReasoningEffort.HIGH,
            reason="hard-analytical-step",
            escalated=True,
        )

    def resolve(self, decision: RouteDecision, default: Any) -> Any:
        return self._model


def _packs() -> LocalePolicy:
    """A loaded three-language policy mirroring the shipped config shape (LANG-R1)."""
    return LocalePolicy.from_config(
        {
            "en": {"compose_directive": "Reply in English.", "limitation": "Not enough data."},
            "de": {"compose_directive": "Antworte auf Deutsch.", "limitation": "Zu wenig Daten."},
            "ru": {"compose_directive": "Otvechaj po-russki.", "limitation": "Malo dannyh."},
        },
        "en",
    )


# --- MODEL-R1/-R2: the routing-policy seam ------------------------------------------------


def test_single_model_policy_resolves_every_tier_to_the_one_model() -> None:
    """The OSS default policy routes deterministically and resolves to the configured model."""
    model = _RecordingModel()
    policy = SingleModelRoutingPolicy.from_labels("flash", "low")
    decision = policy.route(node="compose", signal=DifficultySignal())
    assert (decision.tier, decision.effort) == (ModelTier.FLASH, ReasoningEffort.LOW)
    assert decision.escalated is False
    assert routed_model(policy, model, node="compose", signal=DifficultySignal()) is model


def test_invalid_configured_tier_label_fails_closed() -> None:
    """A non-member configured tier/effort label is rejected at construction (MODEL-R1)."""
    with pytest.raises(ValueError):
        SingleModelRoutingPolicy.from_labels("turbo", "low")
    with pytest.raises(ValueError):
        SingleModelRoutingPolicy.from_labels("flash", "max")


def test_escalated_decision_is_counted_and_resolves_policy_model() -> None:
    """An escalating policy's decision is metered (AGT-OBS-R5) and its model is used."""
    other = _RecordingModel()
    policy = _EscalatingPolicy(other)
    before = obs_metrics.get_registry().counter_value(
        obs_metrics.TIER_ESCALATIONS, labels={"node": "compose", "tier": "pro"}
    )
    chosen = routed_model(policy, _RecordingModel(), node="compose", signal=DifficultySignal())
    after = obs_metrics.get_registry().counter_value(
        obs_metrics.TIER_ESCALATIONS, labels={"node": "compose", "tier": "pro"}
    )
    assert chosen is other
    assert after == before + 1


async def test_compose_node_calls_the_policy_resolved_model() -> None:
    """Per-node routing is REAL: compose drives the model the policy resolves (MODEL-R2)."""
    default = _RecordingModel()
    routed = _RecordingModel()
    node = make_compose(
        object(), default, "persona", _EscalatingPolicy(routed), EMPTY_LOCALE_POLICY, None
    )
    update = await node({"athlete_id": "ath-1", "request_text": "hi"})
    assert update["draft"] == "drafted"
    assert routed.compose_calls and not default.compose_calls


# --- MODEL-R3: token counter seam + window-derived budget ---------------------------------


def test_model_token_counter_prefers_the_models_own_tokenizer() -> None:
    """A model exposing ``count_tokens`` is used verbatim; otherwise the estimator (MODEL-R3)."""

    class _Tokenized(_RecordingModel):
        def count_tokens(self, text: str) -> int:
            return 7

    assert model_token_counter(_Tokenized())("anything at all") == 7
    counter = model_token_counter(_RecordingModel())
    assert counter is estimate_tokens


def test_context_budget_is_window_minus_output_headroom() -> None:
    """The compose input budget reserves the output headroom from the window (MODEL-R3)."""
    assert context_budget(10_000, 8_192) == 1_808
    assert context_budget(None, 8_192) is None
    # A window smaller than the headroom floors at a small positive budget, never <= 0.
    assert context_budget(2_000, 8_192) == 1_024


def test_render_context_trims_lowest_relevance_first_under_injected_budget() -> None:
    """Overflow under the injected counter/budget drops lowest-relevance records (MODEL-R3)."""
    retrieved = {
        "high": {"relevance": 0.9, "body": "keep me"},
        "low": {"relevance": 0.1, "body": "drop me"},
    }
    context, trimmed = render_context(
        "question",
        retrieved,
        token_counter=len,
        token_budget=140,
    )
    assert trimmed is True
    assert "high" in context and "low" not in context


# --- LANG-R1/-R3/-R4: language packs, fallback, limitation copy ---------------------------


def test_locale_resolves_supported_language_and_region_subtag() -> None:
    """A supported language (incl. a regioned tag) resolves to itself, no fallback (LANG-R1)."""
    packs = _packs()
    assert packs.resolve("de") == ("de", False)
    assert packs.resolve("de-AT") == ("de", False)
    assert packs.resolve("ru") == ("ru", False)


def test_unsupported_language_falls_back_to_config_default_and_is_counted() -> None:
    """An unsupported request resolves to ``default_language`` and meters a fallback (LANG-R4)."""
    packs = _packs()
    before = obs_metrics.get_registry().counter_value(
        obs_metrics.LANGUAGE_FALLBACKS, labels={"requested": "fr", "resolved": "en"}
    )
    assert packs.resolve("fr") == ("en", True)
    system = packs.compose_system("persona", "fr")
    after = obs_metrics.get_registry().counter_value(
        obs_metrics.LANGUAGE_FALLBACKS, labels={"requested": "fr", "resolved": "en"}
    )
    assert "Reply in English." in system
    assert "Antworte" not in system  # never mixed languages in one deliverable (LANG-R4)
    assert after == before + 1


def test_unset_locale_is_the_presentation_default_not_a_fallback_event() -> None:
    """A NULL/unset locale resolves to the default silently (LANG-R4 presentation default)."""
    packs = _packs()
    before = obs_metrics.get_registry().counter_value(obs_metrics.LANGUAGE_FALLBACKS)
    assert packs.resolve(None) == ("en", False)
    assert packs.resolve("") == ("en", False)
    after = obs_metrics.get_registry().counter_value(obs_metrics.LANGUAGE_FALLBACKS)
    assert after == before


def test_compose_system_layers_exactly_one_localized_variant() -> None:
    """The compose prompt layers the resolved language's directive after the persona (LANG-R3)."""
    system = _packs().compose_system("persona body", "de")
    assert system.startswith("persona body")
    assert "Antworte auf Deutsch." in system
    assert "Reply in English." not in system


def test_limitation_copy_comes_from_the_loaded_config_pack() -> None:
    """The abstain limitation is the CONFIG pack's copy for the resolved language (LANG-R1)."""
    packs = _packs()
    assert limitation_text({"locale": "de"}, packs) == "Zu wenig Daten."
    assert limitation_text({"locale": "fr"}, packs) == "Not enough data."


def test_no_bundle_limitation_floor_preserves_localized_behaviour() -> None:
    """With no loaded packs the deterministic floor keys on the requested locale (GROUND-R6)."""
    assert "gesicherte Daten" in limitation_text({"locale": "de"})
    assert limitation_text({"locale": "xx"}).startswith("I don't have enough confirmed data")


def test_default_language_without_a_pack_fails_closed_at_load() -> None:
    """A configured default_language with no loaded pack is rejected (LANG-R4 fail-closed)."""
    with pytest.raises(ValueError):
        LocalePolicy.from_config({"en": {"compose_directive": "x", "limitation": "y"}}, "de")


async def test_compose_node_uses_the_run_locale_variant() -> None:
    """The graph compose node selects the localized variant from the run state (LANG-R3)."""
    model = _RecordingModel()
    node = make_compose(object(), model, "persona", SingleModelRoutingPolicy(), _packs(), None)
    await node({"athlete_id": "ath-1", "request_text": "wie geht's", "locale": "de"})
    assert model.compose_calls and "Antworte auf Deutsch." in model.compose_calls[0]
