"""Unit regressions for the API read-surface building blocks (PAGE/ERR/API-R48/R49).

Pure-function tests for the modules the convergence panel flagged: the signed opaque
cursor (PAGE-R5/R6), the extrema-preserving + RDP decimation (API-R48/R49), the
catalog-driven problem builders (ERR-R6/R9), and the problem-document redaction wiring
(ERR-R5/API-R19). These run with no app/DB — they assert the contract at the unit seam.

Tier: T-UNIT (offline, pure functions).
"""

from __future__ import annotations

import datetime as _dt

import pytest
from starlette.requests import Request

from wattwise_core.api.decimate import minmax_index, rdp_simplify, uniform_index
from wattwise_core.api.errors import (
    CATALOG,
    FieldError,
    ProblemError,
    _assemble,
)
from wattwise_core.api.pagination import (
    MAX_PAGE_LIMIT,
    clamp_limit,
    decode_cursor,
    encode_cursor,
)
from wattwise_core.api.problems import (
    parameter_invalid,
    precondition_unmet,
    range_reversed,
)

pytestmark = pytest.mark.unit

_KEY = "unit-cursor-signing-key"
_T = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=_dt.UTC)


def _request(path: str = "/v1/x") -> Request:
    """A minimal Starlette request for the problem-assembly unit (url.path + headers)."""
    return Request({"type": "http", "path": path, "headers": [], "query_string": b""})


# --- PAGE-R5 / PAGE-R6: signed, filter-bound cursor ------------------------------


def test_cursor_roundtrips_and_is_opaque() -> None:
    """A signed cursor round-trips to its keyset and carries no readable plaintext (PAGE-R5)."""
    params = {"sport": "cycling", "sort": "start_time", "order": "desc"}
    token = encode_cursor(_T, "01ACT", params=params, key=_KEY)
    assert "01ACT" not in token  # opaque: the id is not plainly embedded
    when, item_id = decode_cursor(token, params=params, key=_KEY)
    assert when == _T
    assert item_id == "01ACT"


def test_tampered_cursor_is_invalid_cursor() -> None:
    """A forged/edited cursor fails signature verification -> invalid-cursor (PAGE-R5)."""
    params = {"sport": "cycling"}
    token = encode_cursor(_T, "01ACT", params=params, key=_KEY)
    tampered = token[:-2] + ("AA" if not token.endswith("AA") else "BB")
    with pytest.raises(ProblemError) as exc:
        decode_cursor(tampered, params=params, key=_KEY)
    assert exc.value.problem_type.slug == "invalid-cursor"


def test_cursor_bound_to_filters_mismatch_is_rejected() -> None:
    """A cursor replayed against changed filters -> cursor-parameter-mismatch (PAGE-R6)."""
    token = encode_cursor(_T, "01ACT", params={"sport": "cycling"}, key=_KEY)
    with pytest.raises(ProblemError) as exc:
        decode_cursor(token, params={"sport": "running"}, key=_KEY)
    assert exc.value.problem_type.slug == "cursor-parameter-mismatch"


def test_cursor_wrong_key_is_invalid() -> None:
    """A cursor signed with another key cannot be forged in (PAGE-R5, server-issued)."""
    token = encode_cursor(_T, "01ACT", params={}, key=_KEY)
    with pytest.raises(ProblemError) as exc:
        decode_cursor(token, params={}, key="different-key")
    assert exc.value.problem_type.slug == "invalid-cursor"


def test_limit_is_clamped() -> None:
    """limit > 200 is CLAMPED; limit < 1 is REJECTED 422 — never defaulted (PAGE-R3)."""
    assert clamp_limit(10_000) == MAX_PAGE_LIMIT
    assert clamp_limit(25) == 25
    with pytest.raises(ProblemError) as exc:
        clamp_limit(0)
    assert exc.value.problem_type.slug == "validation-error"


# --- API-R48 / API-R49: extrema-preserving + RDP decimation ----------------------


def test_minmax_index_preserves_global_extrema() -> None:
    """Decimation keeps each channel's global min and max sample (API-R48)."""
    length = 1000
    channel = [float(i) for i in range(length)]
    channel[500] = 9999.0  # a spike the uniform stride would otherwise drop
    channel[750] = -9999.0  # a trough
    idx = minmax_index(length, 50, [channel])
    assert 500 in idx and 750 in idx
    assert len(idx) <= 52  # ~budget + the two forced extrema


def test_uniform_index_respects_budget() -> None:
    """A uniform index never exceeds the point budget (plus the final sample)."""
    idx = uniform_index(1000, 100)
    assert len(idx) <= 101
    assert idx[0] == 0 and idx[-1] == 999


def test_rdp_keeps_corners_within_budget() -> None:
    """RDP simplification keeps the turn of an L-shaped track and fits the budget (API-R49)."""
    track = [(0.0, float(i)) for i in range(50)] + [(float(i), 50.0) for i in range(50)]
    out = rdp_simplify(track, 10)
    assert len(out) <= 10
    # the corner (the (0,49)->(0,50) inflection region) must survive
    assert any(abs(p[0]) < 1e-6 and p[1] >= 49.0 for p in out)


# --- ERR-R6 / ERR-R9: catalog problem builders -----------------------------------


def test_parameter_invalid_carries_parameter_locator() -> None:
    """A bad query param -> validation-error with the offending parameter (ERR-R6)."""
    err = parameter_invalid("base")
    assert err.problem_type.slug == "validation-error"
    assert err.errors[0].parameter == "base"


def test_range_reversed_is_validation_error() -> None:
    """A reversed range -> validation-error with an out_of_range code (ERR-R6/PAGE-R8)."""
    err = range_reversed("from")
    assert err.problem_type.slug == "validation-error"
    assert err.errors[0].code == "out_of_range"
    assert err.errors[0].parameter == "from"


def test_precondition_unmet_carries_machine_code() -> None:
    """An analytics precondition -> the catalog slug + machine errors[].code (ERR-R9)."""
    err = precondition_unmet("cp_insufficient_points", "need more points")
    assert err.problem_type.slug == "analytics-precondition-unmet"
    assert err.errors[0].code == "cp_insufficient_points"


# --- ERR-R5 / API-R19: redaction is wired into the rendered problem ---------------


def test_problem_body_redacts_detail_and_field_messages() -> None:
    """A secret/email in detail or a field message is masked in the rendered body (ERR-R5)."""
    request = _request()
    problem = _assemble(
        request,
        CATALOG["internal-error"],
        detail="failed for rider@example.com using sk-abcdef0123456789ABCDEF",
        errors=(FieldError(code="x", message="contact rider@example.com"),),
    )
    body = problem.to_body()
    assert "rider@example.com" not in body["detail"]
    assert "sk-abcdef0123456789ABCDEF" not in body["detail"]
    assert "rider@example.com" not in body["errors"][0]["message"]
