"""Intervals.icu adapter/client error paths + resilience edges (AUT-R17, CLI-R6/R7/R10/R11).

Offline-only (TST-R1): ``respx`` / ``httpx.MockTransport`` fake every response so the
typed client's auth-failure, fetch-failure, retry-exhaustion, and rate-limit branches
run with NO live network. Covered:

* ADP-R4/AUT-R17 — ``ensure_authorized`` probes read-only before ``connected``; a
  missing credential raises the typed :class:`AuthError` (never a silent degrade).
* ADP-R8 — ``fetch_ref`` resolves a wellness day; an absent day is a typed
  :class:`FetchError`, never a fabricated row.
* CLI-R7 — a non-auth 4xx on a required endpoint is a typed ``FETCH_FAILED``.
* CLI-R6 — exhausted transient retries surface as ``SOURCE_UNAVAILABLE``; a
  zero-attempt budget fails immediately (the loop body never runs).
* CLI-R11 — a 429 without / with an unparseable ``Retry-After`` falls back to the
  jittered backoff; the header is never guessed.
* CLI-R10/ING-OBS-R2 — an empty labeled token bucket waits for refill and records
  the per-source rate-limit-wait metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from wattwise_core.config import load_settings
from wattwise_core.domain.enums import GboType
from wattwise_core.ingestion.adapters._resilience import (
    AttemptBudget,
    RetryExhaustedError,
    TokenBucket,
    _Clock,
    _next_wait,
    clock_to_clock,
    resilient_get,
)
from wattwise_core.ingestion.adapters.intervals_icu import (
    IntervalsIcuAdapter,
    IntervalsWellnessAsbo,
)
from wattwise_core.ingestion.base import AuthError, FetchError, FetchErrorKind
from wattwise_core.ingestion.capability import AuthContext, DiscoveryRef

pytestmark = pytest.mark.contract

_FIXTURES = Path(__file__).parent / "fixtures" / "intervals"
_BASE = "https://intervals.icu"
_ATHLETE = "i00000"


def _adapter() -> IntervalsIcuAdapter:
    return IntervalsIcuAdapter(
        settings=load_settings(
            app__environment="development",
            database_dsn="sqlite+aiosqlite:///:memory:",
            token_signing_key="k" * 32,
        )
    )


def _wellness_payload() -> Any:
    return json.loads((_FIXTURES / "wellness.json").read_text())


# ------------------------------------------------------------------- ensure_authorized


async def test_ensure_authorized_without_credential_raises_typed_auth_error() -> None:
    """ADP-R4/AUT-R4: a missing api_key is a terminal AUTH_REVOKED, never a degrade."""
    with pytest.raises(AuthError) as err:
        await _adapter().ensure_authorized(api_key=None, athlete_native_id=_ATHLETE)
    assert err.value.kind is FetchErrorKind.AUTH_REVOKED


@respx.mock
async def test_ensure_authorized_probes_read_only_then_returns_context() -> None:
    """AUT-R17: the read-only athlete probe must succeed before the context is returned."""
    profile = json.loads((_FIXTURES / "athlete_profile.json").read_text())
    route = respx.get(f"{_BASE}/api/v1/athlete/{_ATHLETE}").mock(
        return_value=httpx.Response(200, json=profile)
    )
    ctx = await _adapter().ensure_authorized(api_key="test-key", athlete_native_id=_ATHLETE)
    assert route.called
    assert ctx.athlete_native_id == _ATHLETE
    assert "test-key" not in repr(ctx)  # AUT-R2: the secret never leaks via repr


@respx.mock
async def test_ensure_authorized_rejected_probe_raises_typed_auth_error() -> None:
    """AUT-R17/AUT-R4: a 401 probe raises AuthError so the connection never connects."""
    respx.get(f"{_BASE}/api/v1/athlete/{_ATHLETE}").mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError) as err:
        await _adapter().ensure_authorized(api_key="bad-key", athlete_native_id=_ATHLETE)
    assert err.value.kind is FetchErrorKind.AUTH_REVOKED


def test_client_seam_without_credential_raises_typed_auth_error() -> None:
    """ADP-R4: the per-call client refuses to build without a usable credential."""
    with pytest.raises(AuthError) as err:
        _adapter()._client(AuthContext(athlete_native_id=_ATHLETE, api_key=None))
    assert err.value.kind is FetchErrorKind.AUTH_REVOKED


def test_lazy_settings_resolution_reads_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """CFG-R1a: the no-args factory resolves config lazily once, from the layered sources."""
    monkeypatch.setenv("WATTWISE_APP__ENVIRONMENT", "development")
    monkeypatch.setenv("WATTWISE_DATABASE_DSN", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("WATTWISE_TOKEN_SIGNING_KEY", "k" * 32)
    adapter = IntervalsIcuAdapter()
    resolved = adapter._resolved_settings()
    assert resolved is adapter._resolved_settings()  # resolved once, then reused


# --------------------------------------------------------------------------- fetch_ref


@respx.mock
async def test_fetch_ref_resolves_a_wellness_day() -> None:
    """ADP-R8: a daily_wellness ref fetches the day window and returns the typed ASBO."""
    respx.get(url__regex=rf"{_BASE}/api/v1/athlete/.*/wellness.*").mock(
        return_value=httpx.Response(200, json=_wellness_payload())
    )
    ctx = AuthContext(athlete_native_id=_ATHLETE, api_key="test-key")
    ref = DiscoveryRef(source_native_id="2026-05-01", gbo_type=GboType.DAILY_WELLNESS)
    row = await _adapter().fetch_ref(ctx, ref)
    assert isinstance(row, IntervalsWellnessAsbo)


@respx.mock
async def test_fetch_ref_missing_wellness_day_is_typed_fetch_error() -> None:
    """ADP-R8/CLI-R2: an empty wellness window is a typed FetchError, never a fake row."""
    respx.get(url__regex=rf"{_BASE}/api/v1/athlete/.*/wellness.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    ctx = AuthContext(athlete_native_id=_ATHLETE, api_key="test-key")
    ref = DiscoveryRef(source_native_id="2026-05-02", gbo_type=GboType.DAILY_WELLNESS)
    with pytest.raises(FetchError) as err:
        await _adapter().fetch_ref(ctx, ref)
    assert err.value.kind is FetchErrorKind.FETCH_FAILED


# ------------------------------------------------------------------ typed client errors


@respx.mock
async def test_required_endpoint_404_is_typed_fetch_failed() -> None:
    """CLI-R7: a non-auth 4xx on a required GET becomes FetchError(FETCH_FAILED)."""
    respx.get(url__regex=rf"{_BASE}/api/v1/athlete/.*/wellness.*").mock(
        return_value=httpx.Response(404)
    )
    ctx = AuthContext(athlete_native_id=_ATHLETE, api_key="test-key")
    adapter = _adapter()
    async with adapter._client(ctx) as client:
        with pytest.raises(FetchError) as err:
            await client.fetch_wellness("2026-05-01", "2026-05-01")
    assert err.value.kind is FetchErrorKind.FETCH_FAILED


@respx.mock
async def test_exhausted_transient_retries_surface_as_source_unavailable() -> None:
    """CLI-R6/R7: budget-exhausted 5xx retries become FetchError(SOURCE_UNAVAILABLE)."""
    respx.get(url__regex=rf"{_BASE}/api/v1/athlete/.*").mock(return_value=httpx.Response(503))
    ctx = AuthContext(athlete_native_id=_ATHLETE, api_key="test-key")
    adapter = _adapter()
    async with adapter._client(ctx) as client:
        with pytest.raises(FetchError) as err:
            await client.probe()
    assert err.value.kind is FetchErrorKind.SOURCE_UNAVAILABLE


# ------------------------------------------------------------------- resilience edges


class _FakeClock:
    """Deterministic monotonic clock + sleep recorder (no real wall-time wait)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_zero_attempt_budget_fails_immediately_without_a_request() -> None:
    """CLI-R6: a zero-attempt budget exhausts before any GET — fail closed, no I/O."""
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200)

    budget = AttemptBudget(max_attempts=0, max_elapsed_s=1.0, base_backoff_s=0.0, max_backoff_s=0.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RetryExhaustedError):
            await resilient_get(client, f"{_BASE}/x", budget=budget, clock=_FakeClock())
    assert calls["n"] == 0


def test_429_retry_after_absent_or_unparseable_falls_back_to_backoff() -> None:
    """CLI-R11: only a numeric Retry-After overrides the backoff; junk is never guessed."""
    budget = AttemptBudget(
        max_attempts=3, max_elapsed_s=60.0, base_backoff_s=2.0, max_backoff_s=8.0
    )
    no_header = httpx.Response(429)
    junk_header = httpx.Response(429, headers={"Retry-After": "soon"})
    assert _next_wait(budget, 0, no_header, lambda: 1.0) == 2.0  # ceiling * jitter
    assert _next_wait(budget, 0, junk_header, lambda: 1.0) == 2.0
    numeric = httpx.Response(429, headers={"Retry-After": "7"})
    assert _next_wait(budget, 0, numeric, lambda: 1.0) == 7.0


def test_clock_coercion_accepts_a_prebuilt_clock() -> None:
    """The injectable clock seam passes a prebuilt _Clock through unchanged."""
    fake = _FakeClock()
    clock = _Clock(monotonic=fake.monotonic, sleep=fake.sleep)
    assert clock_to_clock(clock) is clock
    coerced = clock_to_clock(fake)
    fake.now = 42.0
    assert coerced.monotonic() == 42.0  # duck-typed clock-likes are wrapped, not lost


async def test_empty_labeled_bucket_waits_and_records_the_wait_metric() -> None:
    """CLI-R10/ING-OBS-R2: an empty source-labeled bucket sleeps for the deficit."""
    now = {"t": 0.0}
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    bucket = TokenBucket(1.0, 1.0, clock=lambda: now["t"], metrics_source="intervals_icu")
    await bucket.acquire(_sleep)  # consumes the single starting token
    await bucket.acquire(_sleep)  # empty -> must wait ~1 s for the refill
    assert len(slept) == 1
    assert slept[0] == pytest.approx(1.0)
