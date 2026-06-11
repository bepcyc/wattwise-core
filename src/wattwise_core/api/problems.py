"""Shared catalog-driven problem builders for the read surfaces (ERR-R1/R6/R9).

The performance and activities routers raise the SAME closed-catalog problems for
the same conditions (a reversed range, a bad query parameter, an unmet analytics
precondition). Centralizing the builders here means every raise site routes through
:class:`~wattwise_core.api.errors.ProblemError` (rendered as one
``application/problem+json`` document with a populated ``errors[]``), never a raw
framework ``HTTPException`` whose structured ``detail`` the status-only handler would
discard.

Requirement IDs: ERR-R1 (uniform problem document), ERR-R3 (closed ``type``), ERR-R6
(``422 validation-error`` carries ``errors[]`` with a ``parameter``/``pointer``), ERR-R9
(``422 analytics-precondition-unmet`` carries the machine ``errors[].code``), API-R30.
"""

from __future__ import annotations

from wattwise_core.api.copy import message as _copy
from wattwise_core.api.errors import FieldError, ProblemError


def parameter_invalid(parameter: str, message: str | None = None) -> ProblemError:
    """A ``422 validation-error`` for a bad query/path ``parameter`` (ERR-R6).

    The offending ``parameter`` locator survives into the rendered ``errors[]`` so a
    client can point at the field, unlike a framework ``HTTPException`` whose detail
    the status-only handler drops.
    """
    text = message if message is not None else _copy("validation.check_value")
    return ProblemError(
        "validation-error",
        errors=[FieldError(code="invalid", message=text, parameter=parameter)],
    )


def range_reversed(parameter: str = "from") -> ProblemError:
    """A ``422 validation-error`` for a reversed ``from > to`` range (ERR-R6/PAGE-R8)."""
    return ProblemError(
        "validation-error",
        errors=[
            FieldError(
                code="out_of_range",
                message=_copy("validation.range_reversed"),
                parameter=parameter,
            )
        ],
    )


def precondition_unmet(code: str, detail: str) -> ProblemError:
    """A ``422 analytics-precondition-unmet`` carrying the machine code (ERR-R9).

    The closed-catalog slug AND the machine ``errors[].code`` (e.g.
    ``cp_insufficient_points``/``hrv_dsp_unavailable``) reach the client so it can
    branch on the fail-closed analytics contract (ERR-R3/API-R30); ``detail`` is the
    jargon-free, non-leaking reason.
    """
    return ProblemError(
        "analytics-precondition-unmet",
        detail=detail,
        errors=[FieldError(code=code, message=detail)],
    )


def not_found() -> ProblemError:
    """A ``404 not-found`` for an absent/unowned resource (API-R51)."""
    return ProblemError("not-found")


__all__ = ["not_found", "parameter_invalid", "precondition_unmet", "range_reversed"]
