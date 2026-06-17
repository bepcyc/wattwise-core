"""VOICE-R2/-R2a: the presentation strip realizes the allow-list as a closed-vocabulary deny pass.

#98 (v0.0.1-banister). After metric-code translation, ``enforce_presentation`` REMOVES (a) the
engine-emittable internal enum/jargon SCHEMA (``INTERNAL_ENUM_TOKENS``), (b) any deployment-
configured ``forbidden_terms``, and (c) internal-identifier-SHAPED tokens (snake_case + the
requirement-id form ONLY). The single most important guard, proven here: it MUST NOT mangle real
EN/DE/RU coach prose — no ALL_CAPS/camelCase shape rule (so HRV/FTP/ПАНО survive) and NO hardcoded
ambiguous homograph ("weekly cap", German "pro Woche" = per week, "the tool that makes you faster").
"""
# ruff: noqa: RUF001 — this file intentionally carries Cyrillic coach prose as test data; the
# "ambiguous unicode" lint would flag every legitimate RU character.

from __future__ import annotations

import pytest

from wattwise_core.agent.voice import (
    INTERNAL_ENUM_TOKENS,
    VoicePresentation,
    enforce_presentation,
)

pytestmark = pytest.mark.unit

_EMPTY = VoicePresentation()  # OSS default: no labels, empty forbidden_terms


def _scrub(text: str, presentation: VoicePresentation = _EMPTY) -> str:
    """Run the TEXT body through the full presentation pass; return the scrubbed text."""
    _html, out = enforce_presentation(
        f"<p>{text}</p>", text, response_length="standard", presentation=presentation
    )
    return out


# --- (a) the engine enum/jargon SCHEMA is scrubbed -----------------------------------------


def test_internal_enum_jargon_is_scrubbed_from_prose() -> None:
    """VOICE-R2: bare engine jargon (api/database/mcp) never reaches the athlete."""
    out = _scrub("The api hit the database via the mcp adapter.")
    for leaked in ("api", "database", "mcp", "adapter"):
        assert leaked not in out.lower()
    # the connective prose around the removals survives and is tidied (no double spaces / orphan
    # space-before-period).
    assert "  " not in out
    assert " ." not in out


def test_enum_plural_and_spaced_variants_scrubbed() -> None:
    """The deny alternation catches an English plural so 'tokens' is removed, not just 'token'."""
    assert "token" not in _scrub("You have no tokens left for that.").lower()


def test_full_voice_r2_vocabulary_incl_multiword_and_hyphen_scrubbed() -> None:
    """Every VOICE-R2 forbidden term the engine can emit is covered — incl. spaced/hyphen forms.

    The schema alternation preserves spaces/hyphens (``re.escape``), so the human-readable
    two-word jargon the model is likeliest to write is caught, not just the snake_case form.
    """
    out = _scrub(
        "Per the gbo and vector store and canonical-store and source descriptor, we scrub it."
    ).lower()
    for leaked in ("gbo", "vector store", "canonical-store", "source descriptor", "scrub"):
        assert leaked not in out


# --- (b) deployment-configured forbidden terms (per-language extension) ---------------------


def test_configured_forbidden_terms_scrubbed_incl_de_and_ru() -> None:
    """A deployment's ``forbidden_terms`` (incl. DE/RU + a homograph) are scrubbed only when set.

    The match is the configured form + an English plural; non-English INFLECTIONS (RU
    "токенов", DE "Modelle") need explicit enumeration in the config list (documented residual
    limitation) — so the fixture uses the exact configured surface forms.
    """
    pres = VoicePresentation.from_aliases(
        {}, forbidden_terms=["budget", "schnittstelle", "модель", "токен"]
    )
    out = _scrub("Your budget, die Schnittstelle, твоя модель, твой токен.", pres)
    for leaked in ("budget", "schnittstelle", "модель", "токен"):
        assert leaked not in out.lower()


def test_homograph_scrubbed_ONLY_when_configured() -> None:
    """'budget' survives by DEFAULT (a real coach word); scrubbed ONLY when a deployment opts in."""
    assert "budget" in _scrub("Keep your weekly budget of easy hours.").lower()
    pres = VoicePresentation.from_aliases({}, forbidden_terms=["budget"])
    assert "budget" not in _scrub("Keep your weekly budget of easy hours.", pres).lower()


# --- (c) THE critical false-positive guard: real coach prose is preserved -------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "You hit your weekly cap of 8 hours.",
        "Recovery is the tool that makes you faster.",
        "That ramp rate is a model of consistency.",
        "Go pro this season like a pro.",
        "A flash of speed on that final sprint.",
        "That surge cost you on the climb.",
        "Push past the next frontier of your fitness.",
        "Keep the thread of consistency going.",
        "The endpoint of your final interval was strong.",
    ],
)
def test_real_english_homographs_preserved_by_default(sentence: str) -> None:
    """Ambiguous VOICE-R2 words that are also everyday coach prose pass through unmangled."""
    assert _scrub(sentence) == sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "Your FTP test went well.",
        "Compare to your PR from March.",
        "HRV looks solid today.",
        "SST today, then some HIT later.",
    ],
)
def test_sport_abbreviations_preserved_no_allcaps_rule(sentence: str) -> None:
    """VOICE-R4 surfaces 'your HRV'/FTP: NO ALL_CAPS shape rule, so abbreviations survive."""
    assert _scrub(sentence) == sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "Pro Woche zwei harte Einheiten.",
        "Pro Tag eine Stunde locker, dein HFV ist gut.",
        "GA1-Block diese Woche im EN-Bereich.",
    ],
)
def test_real_german_prose_preserved(sentence: str) -> None:
    """German 'pro' (=per) and DE abbreviations (HFV/GA1/EN) are not codes — left intact."""
    assert _scrub(sentence) == sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "Сегодня ты свежий и готов работать.",
        "Работай в зоне ПАНО, следи за ЧСС и МПК.",
        "На этой неделе больше ОФП.",
    ],
)
def test_real_russian_prose_preserved(sentence: str) -> None:
    """ASCII-anchored shape rules are invisible to Cyrillic; RU prose + abbreviations survive."""
    assert _scrub(sentence) == sentence


# --- identifier-shape pass: snake_case + requirement-id ONLY --------------------------------


def test_internal_identifier_shapes_scrubbed() -> None:
    """snake_case identifiers and requirement-ids leaking into prose are removed (VOICE-R2a)."""
    out = _scrub("Per readiness_form_assessment and weekly_digest, see GROUND-R3 and VOICE-R7.")
    for leaked in ("readiness_form_assessment", "weekly_digest", "GROUND-R3", "VOICE-R7"):
        assert leaked not in out


@pytest.mark.parametrize(
    "sentence",
    [
        "A solid warm-up before the long-run today.",
        "Your best block was 2026-06-17.",
        "Easy 7:30 pace felt smooth.",
        "Your form sits around 3.14 right now.",
        "That athlete's engine is strong.",
    ],
)
def test_shape_pass_negative_guards(sentence: str) -> None:
    """Hyphen-compounds, dates, times, decimals, possessives are not identifier-shaped — kept."""
    assert _scrub(sentence) == sentence


# --- lead fallthrough + observability hook --------------------------------------------------


def test_lead_emptied_by_scrub_leaves_no_leak_and_leads_with_state() -> None:
    """A lead gutted by a scrub never leaks and never leads with orphaned punctuation (fail-closed).

    ``_repair_lead`` then either prepends the warm ``fallback_lead`` OR promotes the next genuine
    state sentence to lead — both are acceptable; the invariants are: zero leak + a clean,
    state-first opener + the real state content preserved.
    """
    pres = VoicePresentation()
    out = _scrub("The vector_store database api. You're fresh and ready to work.", pres)
    assert "vector_store" not in out and "database" not in out and "api" not in out.lower()
    assert "fresh" in out  # the genuine state sentence survives
    assert out[:1].isalpha() or out.startswith(
        pres.fallback_lead[:5]
    )  # clean opener, no orphan punct
    leads_fallback = out.startswith(pres.fallback_lead[:12])
    leads_state = out.lstrip().lower().startswith("you're fresh")
    assert leads_fallback or leads_state


def test_on_scrub_hook_records_token_and_reason() -> None:
    """The optional on_scrub callback records (token, reason) per removal; default needs no hook."""
    hits: list[tuple[str, str]] = []
    pres = VoicePresentation.from_aliases({}, forbidden_terms=["budget"])
    enforce_presentation(
        "<p>x</p>",
        "the api budget and weekly_digest",
        response_length="standard",
        presentation=pres,
        on_scrub=lambda token, reason: hits.append((token, reason)),
    )
    reasons = {r for _t, r in hits}
    tokens = {t.lower() for t, _r in hits}
    assert "voice-r2-forbidden" in reasons  # api + budget
    assert "internal-identifier-shape" in reasons  # weekly_digest
    assert "api" in tokens and "budget" in tokens and "weekly_digest" in tokens


def test_schema_set_is_lowercase_and_nonempty() -> None:
    """INTERNAL_ENUM_TOKENS is a non-empty, lowercase schema set (matched case-folded)."""
    assert INTERNAL_ENUM_TOKENS
    assert all(t == t.lower() for t in INTERNAL_ENUM_TOKENS)
    # the ambiguous homographs are DELIBERATELY excluded from the hardcoded schema (#98) — incl.
    # 'endpoint' ("the endpoint of your interval"), moved to the config-only carve-out on review.
    for homograph in (
        "cap",
        "tool",
        "model",
        "pro",
        "flash",
        "budget",
        "cost",
        "tier",
        "thread",
        "endpoint",
    ):
        assert homograph not in INTERNAL_ENUM_TOKENS
