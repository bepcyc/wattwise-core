"""Request-shaping helpers for the agent ``/ask`` surface (API-R11/R11e/R11f/R37).

Factored out of :mod:`wattwise_core.api.routers.agent_routes` (QUAL-R9 module-size split) so the
router file holds the seam/endpoint logic and these pure request-shaping helpers live in one
focused place. Everything here is a PURE function of the inbound request body / header — no engine,
no router state, no I/O: body-invariant validation (``question`` required unless a follow-up,
API-R11e), the ``Accept-Language`` -> locale resolution (API-R37), and the response-length default
(API-R11f). They depend only on the request schemas + the uniform error type.
"""

from __future__ import annotations

import re
from typing import Final

from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.api.routers.agent_schemas import AgentAskRequest, ResponseLength


def validate_request(body: AgentAskRequest) -> None:
    """Enforce the API-R11/R11e body invariants beyond pydantic types.

    ``question`` is REQUIRED unless a ``follow_up`` is present (API-R11e); a request
    with neither is a semantic ``422`` ``validation-error`` (ERR-R6), not a model call.
    The human copy comes from the catalog title (API-R21); the machine-readable cause
    is the ``errors[]`` code clients branch on (ERR-R3), not an inline sentence.
    """
    if body.question is None and body.follow_up is None:
        raise ProblemError(
            "validation-error",
            errors=[FieldError(code="question_required", message="", pointer="/question")],
        )


#: The locales for which this surface ships model-free fail-closed COPY (degraded reason
#: gloss / limitation floor, API-R37). NOT an allow-list for the coach's output language —
#: the coach answers in ANY language via the directive (LANG-R1/-R3). An unenumerated locale
#: still drives the directive; only the model-free sentences fall back to the English floor.
SUPPORTED_LOCALES: Final[frozenset[str]] = frozenset({"en", "de", "ru"})

#: BCP-47-shaped gate for a header language tag (mirrors the body field + INJECT-R1): a primary
#: language subtag + optional subtags only, so a malformed/garbage header tag is ignored rather
#: than passed through to the directive.
_HEADER_TAG_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")


def scan_header_locale(accept_language: str | None) -> str | None:
    """The first well-formed ``Accept-Language`` language tag, else ``None`` (API-R37).

    The single header-scan the locale resolvers share. Per the any-language ruling
    (LANG-R1/-R3) it is NOT clamped to an enumerated set: it returns the first BCP-47-shaped
    primary language subtag from the comma-separated header (quality weights ignored), so a
    ``fr`` / ``pt-BR`` header drives the directive in that language just like en/de/ru. A
    malformed/garbage tag is skipped. Returning ``None`` (not ``en``) lets the caller fall
    through to the PERSISTED setting before the engine ``en`` baseline (the API-R37 chain).
    """
    if accept_language:
        for part in accept_language.split(","):
            raw = part.split(";", 1)[0].strip().lower()
            if _HEADER_TAG_RE.match(raw):
                return raw.split("-", 1)[0]
    return None


def header_locale(accept_language: str | None) -> str:
    """The first well-formed ``Accept-Language`` language tag, else the default ``en`` (API-R37)."""
    return scan_header_locale(accept_language) or "en"


def resolve_locale(
    body: AgentAskRequest, accept_language: str | None, persisted: str | None = None
) -> str:
    """Resolve the response language per the API-R37 precedence chain.

    ``body.language`` (the per-call override) -> ``Accept-Language`` -> the PERSISTED
    setting (the language subtag of ``athlete.primary_locale``, loaded server-side and
    passed as ``persisted``) -> the engine ``en`` baseline. The persisted default is the
    one applied to every athlete-facing agent answer when no per-request value is given
    (API-R37); a per-request value never mutates it. The resolved locale drives the
    any-language compose DIRECTIVE (LANG-R1/-R3): ``body.language`` and the header both
    pass through ANY well-formed BCP-47 tag (validated, never allow-listed); only the
    persisted-fallback rung is restricted to the locales with model-free fail-closed copy.
    """
    if body.language is not None:
        return body.language
    header = scan_header_locale(accept_language)
    if header is not None:
        return header
    if persisted in SUPPORTED_LOCALES:
        return persisted
    return "en"


def resolve_response_length(body: AgentAskRequest) -> ResponseLength | None:
    """The per-request response-length OVERRIDE, or ``None`` for the persisted default (API-R11f).

    Returns the body's per-request ``response_length`` VERBATIM, or ``None`` when omitted. ``None``
    is NOT collapsed to ``standard`` here: the engine applies the athlete's PERSISTED verbosity
    preference (MEM-R1 / VOICE-R8 §382, held in the agent-state store) as the default for a run with
    no per-request value, falling back to ``standard`` only when no preference is stored. A given
    value overrides for this one call WITHOUT mutating the stored default (VOICE-R8).
    """
    return body.response_length


__all__ = [
    "SUPPORTED_LOCALES",
    "header_locale",
    "resolve_locale",
    "resolve_response_length",
    "scan_header_locale",
    "validate_request",
]
