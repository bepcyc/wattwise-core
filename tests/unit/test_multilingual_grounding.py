"""Directive-driven any-language coach + STRUCTURAL workout grounding (issues #17/#18).

These pin the owner's design ruling that the coach answers in ANY language the model speaks via a
config-templated DIRECTIVE — never an enumerated per-language pack/string table — and that a
prescribed workout grounds by its TYPED canonical prescription (a language-independent enum), not by
re-matching a translated surface name:

- the readiness narrator's system prompt is composed through ``LocalePolicy.compose_system`` (the
  SAME any-language directive the free-form answer path uses), so a non-en/de/ru locale still drives
  the requested language with NO language enumeration (issue #17);
- a NAME claim carrying a structured ``workout_type`` grounds by that enum even when its surface
  name is non-English and not in the English canonical-name set, and yields the SAME stable
  canonical id (issue #18); a claim with no type still uses the legacy surface match (back-compat);
- a DETAILED grounded answer surfaces >=1 citation by backfilling from grounded observations (#18);
- an output-language regression over a LOCALE-AGNOSTIC sample (incl. a non-en/de/ru language) proves
  the directive — not a hardcoded supported-set — carries the language.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wattwise_core.agent.contracts import Claim, ClaimKind, GroundDecision, GroundVerdict, RunStatus
from wattwise_core.agent.deliverables import _detailed_citation_floor
from wattwise_core.agent.grounding import ground
from wattwise_core.agent.grounding_evidence import (
    CANONICAL_WORKOUT_NAMES,
    CanonicalWorkoutType,
    _SnapshotEvidence,
)
from wattwise_core.agent.locale import LocalePolicy
from wattwise_core.agent.voice import Citation, Observation
from wattwise_core.api.routers.agent_request import resolve_locale, scan_header_locale
from wattwise_core.api.routers.agent_schemas import AgentAskRequest

pytestmark = pytest.mark.unit


# --- #17: the API BOUNDARY accepts any well-formed language tag, not an enum -------------


@pytest.mark.parametrize("tag", ["en", "de", "ru", "fr", "pt-BR", "zh-Hant", "es-419"])
def test_request_accepts_any_well_formed_bcp47_language_tag(tag: str) -> None:
    """The ``language`` body field is a validated free tag, NOT an enum (issue #17, API-R11).

    A well-formed non-en/de/ru tag is accepted at the boundary so it can drive the directive;
    the field never allow-lists a language. No tag is hardcoded as "the supported set" here.
    """
    assert AgentAskRequest(question="?", language=tag).language == tag


@pytest.mark.parametrize("tag", ["", "1", "e", "english!", "de_DE", "../etc", "x" * 40, "en;rm"])
def test_request_rejects_malformed_language_tag_with_422(tag: str) -> None:
    """A malformed/garbage tag fails closed as a validation error (INJECT-R1, fail-closed).

    The shape gate keeps an injection/garbage tag out of the prompt directive without ever
    enumerating the allowed languages — only the BCP-47 SHAPE is enforced.
    """
    with pytest.raises(ValidationError):
        AgentAskRequest(question="?", language=tag)


def test_request_language_override_drives_resolve_for_an_unenumerated_locale() -> None:
    """A non-en/de/ru body override flows verbatim through ``resolve_locale`` (API-R37/#17)."""
    body = AgentAskRequest(question="?", language="fr")
    assert resolve_locale(body, accept_language="de", persisted="ru") == "fr"


@pytest.mark.parametrize(
    ("header", "expected"),
    [("fr", "fr"), ("pt-BR,en;q=0.5", "pt"), ("ja", "ja"), ("de", "de"), ("en-US", "en")],
)
def test_header_locale_passes_through_any_well_formed_tag(header: str, expected: str) -> None:
    """The ``Accept-Language`` scan is NOT clamped to en/de/ru; any shaped tag drives it (#17)."""
    assert scan_header_locale(header) == expected


@pytest.mark.parametrize("header", ["", "***", "1", ";q=0.9", "  "])
def test_header_locale_skips_garbage_tags(header: str) -> None:
    """A malformed header tag is ignored (falls through to persisted/baseline), not passed on."""
    assert scan_header_locale(header) is None


# A config-templated any-language pass-through policy mirroring the shipped config (NOT an
# enumerated table): only "en" is a loaded pack; every OTHER language is served by the directive
# TEMPLATE interpolated with the requested tag. This is the structure the owner ruled for.
def _passthrough_policy() -> LocalePolicy:
    return LocalePolicy.from_config(
        {"en": {"compose_directive": "Reply in English.", "limitation": "Not enough data."}},
        "en",
        passthrough_enabled=True,
        passthrough_directive=(
            "Answer entirely in the language whose IETF language tag is '{language_tag}' "
            "(the athlete requested the locale '{locale}')."
        ),
    )


# --- #17: readiness narration speaks the requested language via the DIRECTIVE, not packs ---


@pytest.mark.parametrize("locale", ["es", "it", "pt", "ja", "fr-CA"])
def test_compose_system_directive_carries_any_requested_language(locale: str) -> None:
    """An UNENUMERATED language is served by the directive TEMPLATE, never a pack (issue #17).

    The composed system prompt names the EXACT requested primary subtag — proving the directive
    carries the language for any locale the policy never enumerated. No language is hardcoded here:
    the assertion is derived from the requested locale itself, so adding a language needs no code.
    """
    policy = _passthrough_policy()
    primary = locale.split("-", 1)[0]
    system = policy.compose_system("readiness persona", locale)
    assert system.startswith("readiness persona")
    assert f"'{primary}'" in system  # the directive names the requested language tag
    assert "Reply in English." not in system  # NOT the default pack's variant (no enumeration)


def test_compose_system_is_the_same_seam_for_readiness_and_freeform() -> None:
    """Readiness and the free-form answer compose through the SAME directive seam (issue #17).

    Threading ``readiness_system`` vs the free-form persona through the identical
    ``compose_system`` call yields the identical directive tail — proving readiness is
    directive-driven, not served by a separate readiness-language pack.
    """
    policy = _passthrough_policy()
    readiness_system = policy.compose_system("READINESS PERSONA", "it")
    answer_system = policy.compose_system("ANSWER PERSONA", "it")
    readiness_tail = readiness_system.removeprefix("READINESS PERSONA")
    answer_tail = answer_system.removeprefix("ANSWER PERSONA")
    assert readiness_tail == answer_tail
    assert "'it'" in readiness_tail


# --- #18: workout grounding by STRUCTURE, not translated names ----------------------------


def _name_claim(text: str, *, workout_type: str | None = None) -> Claim:
    return Claim(kind=ClaimKind.NAME, text=text, workout_type=workout_type)


def _plan_evidence() -> _SnapshotEvidence:
    """Snapshot evidence wired with the PLAN canonical workout library (COACH-R2)."""

    class _Bare:
        async def metric_value(self, metric: str, as_of: str | None) -> float | None:
            return None

        def url_allowed(self, url: str) -> bool:
            return False

    return _SnapshotEvidence(_Bare(), {}, allow_names=CANONICAL_WORKOUT_NAMES)  # type: ignore[arg-type]


def test_non_english_workout_name_grounds_by_structured_type() -> None:
    """A plan in ANY language grounds because the STRUCTURED type is checked (issue #18).

    The surface name is German ("Schwellenintervalle") and is NOT in the English canonical-name
    set, so the legacy surface-name path would scrub it and the plan would degrade. With the typed
    ``workout_type`` the claim grounds — language-independently — and cites the canonical id.
    """
    evidence = _plan_evidence()
    draft = "Morgen: Schwellenintervalle."
    claim = _name_claim(
        "Schwellenintervalle", workout_type=CanonicalWorkoutType.THRESHOLD_INTERVALS.value
    )
    result = ground(draft, [claim], evidence, allow_urls=[])
    assert result.decision is GroundDecision.PROCEED
    grounded = [c for c in result.claims if c.verdict is GroundVerdict.GROUNDED]
    assert grounded, "the non-English prescription must ground by its structured type"
    assert grounded[0].citation == {
        "kind": "name",
        "record": "workout",
        "canonical_id": "workout:threshold intervals",
    }


def test_structured_type_and_english_name_yield_the_same_canonical_id() -> None:
    """Structural grounding cites the SAME canonical id as the legacy English-name path (#18)."""
    evidence = _plan_evidence()
    by_type = ground(
        "x",
        [_name_claim("irgendwas", workout_type=CanonicalWorkoutType.VO2MAX_INTERVALS.value)],
        evidence,
        allow_urls=[],
    )
    by_name = ground("x", [_name_claim("VO2max intervals")], evidence, allow_urls=[])
    id_by_type = by_type.claims[0].citation["canonical_id"]  # type: ignore[index]
    id_by_name = by_name.claims[0].citation["canonical_id"]  # type: ignore[index]
    assert id_by_type == id_by_name == "workout:vo2max intervals"


def test_name_without_structured_type_still_uses_surface_match() -> None:
    """A NAME claim with no structured type keeps the legacy surface-name behaviour (back-compat).

    Both directions: a recognised English name still grounds; an invented name still scrubs.
    """
    evidence = _plan_evidence()
    good = ground("x", [_name_claim("Rest day")], evidence, allow_urls=[])
    bad = ground("x", [_name_claim("magic super workout")], evidence, allow_urls=[])
    assert good.claims[0].verdict is GroundVerdict.GROUNDED
    assert bad.claims and bad.claims[0].verdict is GroundVerdict.UNGROUNDED


def test_out_of_vocabulary_structured_type_fails_closed() -> None:
    """An unknown structured type resolves to None and the claim scrubs (STRUCT-R3, fail-closed)."""
    evidence = _plan_evidence()
    assert evidence.canonical_workout_type("not_a_real_type") is None
    result = ground("x", [_name_claim("whatever", workout_type="not_a_real_type")], evidence, [])
    assert result.claims and result.claims[0].verdict is GroundVerdict.UNGROUNDED


def test_structured_type_fails_closed_without_a_library() -> None:
    """With no canonical library (the free-form default) a structured type never grounds (#18)."""

    class _Bare:
        async def metric_value(self, metric: str, as_of: str | None) -> float | None:
            return None

        def url_allowed(self, url: str) -> bool:
            return False

    free_form = _SnapshotEvidence(_Bare(), {}, allow_names=frozenset())  # type: ignore[arg-type]
    assert free_form.canonical_workout_type(CanonicalWorkoutType.REST_DAY.value) is None


def test_every_canonical_type_maps_to_an_allowed_name() -> None:
    """Each enum member resolves to a name in the shipped library (1:1, no drift, GROUND-R5)."""
    evidence = _plan_evidence()
    for member in CanonicalWorkoutType:
        cid = evidence.canonical_workout_type(member.value)
        assert cid is not None and cid.removeprefix("workout:") in CANONICAL_WORKOUT_NAMES


# --- #18: detailed-length citation floor on grounded survivors ----------------------------


def _obs_with_citation() -> Observation:
    return Observation(
        observation_id="o1",
        text="grounded line",
        citations=(Citation(record_id="form@2026-06-10", metric="form", value=-12.0),),
    )


def test_detailed_answer_backfills_a_citation_from_grounded_observations() -> None:
    """A DETAILED grounded answer with empty citations surfaces >=1 from observations (#18)."""
    out = _detailed_citation_floor(
        (),
        [_obs_with_citation()],
        status=RunStatus.COMPLETED,
        response_length="detailed",
    )
    assert len(out) == 1 and out[0].record_id == "form@2026-06-10"


def test_detailed_floor_never_fabricates_without_grounded_evidence() -> None:
    """With no grounded observation the floor leaves the answer honestly citation-free (#18)."""
    out = _detailed_citation_floor(
        (),
        [Observation(observation_id="o", text="no numbers", citations=())],
        status=RunStatus.COMPLETED,
        response_length="detailed",
    )
    assert out == ()


def test_detailed_floor_is_inert_for_short_standard_or_unsuccessful_runs() -> None:
    """The floor applies ONLY to a successful detailed turn (length/status guarded, #18)."""
    obs = [_obs_with_citation()]
    assert (
        _detailed_citation_floor((), obs, status=RunStatus.COMPLETED, response_length="standard")
        == ()
    )
    assert (
        _detailed_citation_floor((), obs, status=RunStatus.DEGRADED, response_length="detailed")
        == ()
    )


def test_detailed_floor_preserves_existing_citations_untouched() -> None:
    """When the answer already has citations the floor is a no-op (no duplication, #18)."""
    existing = (Citation(record_id="ctl@2026-06-10", metric="ctl", value=80.0),)
    out = _detailed_citation_floor(
        existing, [_obs_with_citation()], status=RunStatus.COMPLETED, response_length="detailed"
    )
    assert out == existing
