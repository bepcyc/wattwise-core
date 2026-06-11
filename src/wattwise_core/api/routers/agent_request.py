"""Request-shaping helpers for the agent ``/ask`` surface (API-R11/R11e/R11f/R37).

Factored out of :mod:`wattwise_core.api.routers.agent_routes` (QUAL-R9 module-size split) so the
router file holds the seam/endpoint logic and these pure request-shaping helpers live in one
focused place. Everything here is a PURE function of the inbound request body / header — no engine,
no router state, no I/O: body-invariant validation (``question`` required unless a follow-up,
API-R11e), the ``Accept-Language`` -> locale resolution (API-R37), and the response-length default
(API-R11f). They depend only on the request schemas + the uniform error type.
"""

from __future__ import annotations

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


#: The languages this surface localizes athlete-facing copy into (API-R37).
SUPPORTED_LOCALES: Final[frozenset[str]] = frozenset({"en", "de", "ru"})


def scan_header_locale(accept_language: str | None) -> str | None:
    """The first SUPPORTED ``Accept-Language`` tag (en/de/ru), else ``None`` (API-R37).

    The single header-scan the locale resolvers share: it reads the first supported
    two-letter language tag from the comma-separated header, ignoring quality weights.
    Returning ``None`` (not ``en``) lets the caller fall through to the PERSISTED
    setting before the engine ``en`` baseline (the API-R37 precedence chain).
    """
    if accept_language:
        for part in accept_language.split(","):
            tag = part.split(";", 1)[0].strip().lower()[:2]
            if tag in SUPPORTED_LOCALES:
                return tag
    return None


def header_locale(accept_language: str | None) -> str:
    """The first supported ``Accept-Language`` tag (en/de/ru), else the default ``en``."""
    return scan_header_locale(accept_language) or "en"


def resolve_locale(
    body: AgentAskRequest, accept_language: str | None, persisted: str | None = None
) -> str:
    """Resolve the response language per the API-R37 precedence chain.

    ``body.language`` (the per-call override) -> ``Accept-Language`` -> the PERSISTED
    setting (the language subtag of ``athlete.primary_locale``, loaded server-side and
    passed as ``persisted``) -> the engine ``en`` baseline. The persisted default is the
    one applied to every athlete-facing agent answer when no per-request value is given
    (API-R37); a per-request value never mutates it.
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
