"""Outbound-client resilience fault-injection: the §4.2/§4.4 transient + drift path (TST-R5).

Offline-only: a deterministic ``httpx.MockTransport`` injects the §4.2 transient faults
(429 / 5xx / connection-reset / timeout) and the schema-drift case, so the typed client's
retry/backoff/limiter paths run with NO live network (TST-R1). The clock and sleep are
injected so backoff is asserted deterministically with no real wall-time wait.

Covered (the §4.2 / §4.4 contract the OSS Intervals client MUST satisfy):

* CLI-R6 — a transient 5xx/429/reset/timeout is retried with exponential backoff +
  FULL JITTER, bounded by a per-source max attempt count AND a max elapsed budget.
* CLI-R7 — a non-transient 4xx (e.g. 404) is NOT retried; it surfaces once.
* CLI-R2 — a response that fails validation raises ``FetchError(kind=SCHEMA_MISMATCH)``,
  NOT a raw ``pydantic.ValidationError``, and is never partially coerced into a GBO.
* CLI-R10 — the client enforces the source rate limit client-side via a token bucket
  keyed per source+credential, so it never relies on the source to reject excess.
* CLI-R11 — on a 429 the client respects ``Retry-After`` AND adaptively (persistently)
  reduces the shared limiter's issue rate so subsequent requests are slower.
* CLI-R5 — a cancellation mid-retry propagates cleanly (the client raises, never returns
  a half-built object) and never fabricates a value.

This file covers the TST-R5 fault categories that live at the CLIENT boundary: the §4.2
transient retry/backoff path, the §4.4 limiter (incl. CLI-R11 adaptive reduction), schema
drift (CLI-R2), and client-level cancellation (CLI-R5) — asserting in every case that no
value is fabricated. The remaining TST-R5 categories that live at the ORCHESTRATOR /
multi-source boundary — mid-run cancellation across the store transaction, source-removed
isolation, and the two-source "other sources keep working" assertion — are covered in
``tests/integration/test_sync_faults.py``. (The TST-R5 token-refresh-race [AUT-R3] and
circuit-breaker [SCH-R6] categories are scheduler/credentials machinery the spec scopes to
the COMMERCIAL orchestration layer — §8.4 / CLI-R13 — and are NOT shipped in the OSS engine,
so they have no OSS code path to fault-inject here; see the slice gate / report.)
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pydantic
import pytest

from wattwise_core.ingestion.adapters._resilience import (
    AttemptBudget,
    RetryExhaustedError,
    TokenBucket,
    resilient_get,
    validate_or_fetch_error,
)
from wattwise_core.ingestion.adapters.intervals_icu import (
    IntervalsActivityAsbo,
    IntervalsIcuClient,
)
from wattwise_core.ingestion.base import AuthError, FetchError, FetchErrorKind

pytestmark = pytest.mark.contract

_URL = "https://intervals.icu/api/v1/athlete/i0/activities"


class _FakeClock:
    """A monotonic clock + sleep recorder so backoff is asserted with no real wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _budget(*, max_attempts: int = 5, max_elapsed_s: float = 60.0) -> AttemptBudget:
    return AttemptBudget(
        max_attempts=max_attempts,
        max_elapsed_s=max_elapsed_s,
        base_backoff_s=1.0,
        max_backoff_s=8.0,
    )


# --------------------------------------------------------------- CLI-R6 retry/backoff


async def test_transient_5xx_is_retried_then_succeeds() -> None:
    """A 503 then 200 succeeds after exactly one backoff sleep (CLI-R6 retry transient)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=[{"id": "a1"}])

    clock = _FakeClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await resilient_get(client, _URL, budget=_budget(), clock=clock)
    assert resp.status_code == 200
    assert calls["n"] == 2
    assert len(clock.slept) == 1  # exactly one backoff between the two attempts


async def test_backoff_is_exponential_with_full_jitter() -> None:
    """Each backoff is full-jitter in [0, min(base*2**i, cap)] and grows (CLI-R6)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    clock = _FakeClock()
    # Full jitter draws the FRACTION of the ceiling; 1.0 == draw the whole ceiling
    # (deterministic), so the sleeps expose the doubling-then-capping ceiling itself.
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RetryExhaustedError):
            await resilient_get(
                client,
                _URL,
                budget=_budget(max_attempts=4),
                clock=clock,
                jitter=lambda: 1.0,
            )
    # Three backoffs between four attempts; ceiling = min(1*2**i, 8) = 1, 2, 4.
    assert clock.slept == [1.0, 2.0, 4.0]


async def test_full_jitter_draws_a_fraction_of_the_ceiling() -> None:
    """A jitter draw of 0.5 yields half the exponential ceiling (CLI-R6 full jitter)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    clock = _FakeClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RetryExhaustedError):
            await resilient_get(
                client,
                _URL,
                budget=_budget(max_attempts=3),
                clock=clock,
                jitter=lambda: 0.5,
            )
    # Ceilings 1, 2 -> half each: 0.5, 1.0 (full jitter in [0, ceiling]).
    assert clock.slept == [0.5, 1.0]


async def test_retry_stops_at_max_attempts() -> None:
    """A permanently-failing transient is retried at most ``max_attempts`` times (CLI-R6)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(502)

    clock = _FakeClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RetryExhaustedError):
            await resilient_get(client, _URL, budget=_budget(max_attempts=3), clock=clock)
    assert calls["n"] == 3  # bounded — not infinite


async def test_retry_stops_at_max_elapsed_budget() -> None:
    """Retries stop once the per-source elapsed budget is exceeded (CLI-R6 budget)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    clock = _FakeClock()
    budget = AttemptBudget(
        max_attempts=100, max_elapsed_s=3.0, base_backoff_s=1.0, max_backoff_s=8.0
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RetryExhaustedError):
            await resilient_get(client, _URL, budget=budget, clock=clock, jitter=lambda: 1.0)
    # Elapsed budget (3s), not the 100-attempt cap, bounds the loop.
    assert clock.now <= 3.0 + 1e-9
    assert calls["n"] < 100


async def test_timeout_and_connection_reset_are_retried() -> None:
    """A timeout then a connect error then 200 succeeds (CLI-R6 reset/timeout transient)."""
    seq: list[object] = [
        httpx.TimeoutException("read timeout"),
        httpx.ConnectError("connection reset"),
        httpx.Response(200, json=[]),
    ]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        item = seq[calls["n"]]
        calls["n"] += 1
        if isinstance(item, httpx.Response):
            return item
        raise item

    clock = _FakeClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await resilient_get(client, _URL, budget=_budget(), clock=clock)
    assert resp.status_code == 200
    assert calls["n"] == 3


# --------------------------------------------------------------- CLI-R7 non-transient


async def test_non_transient_4xx_is_not_retried() -> None:
    """A 404 is NOT retried; it surfaces once (CLI-R7 non-transient)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    clock = _FakeClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await resilient_get(client, _URL, budget=_budget(), clock=clock)
    assert resp.status_code == 404
    assert calls["n"] == 1  # exactly one — non-transient is never retried
    assert clock.slept == []


# --------------------------------------------------------------- CLI-R11 Retry-After


async def test_429_respects_retry_after_header() -> None:
    """A 429 with ``Retry-After: 5`` waits exactly 5s before the retry (CLI-R11)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "5"})
        return httpx.Response(200, json=[])

    clock = _FakeClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await resilient_get(client, _URL, budget=_budget(), clock=clock, jitter=lambda: 1.0)
    assert resp.status_code == 200
    # Retry-After (5s) overrides the computed backoff (which would be ~1s).
    assert clock.slept == [5.0]


# --------------------------------------------------------------- CLI-R10 token bucket


async def test_token_bucket_waits_when_exhausted() -> None:
    """The client-side bucket throttles before the request, not via source rejection (CLI-R10)."""
    clock = _FakeClock()
    bucket = TokenBucket(rate_per_s=1.0, capacity=1.0, clock=clock.monotonic)
    await bucket.acquire(clock.sleep)  # first token is free (full bucket)
    assert clock.slept == []
    await bucket.acquire(clock.sleep)  # second must wait ~1/rate for a refill
    assert clock.slept and clock.slept[-1] == pytest.approx(1.0, abs=1e-6)


def test_token_bucket_key_is_source_plus_credential() -> None:
    """The limiter is keyed per source+credential so swapping the key is a new bucket (CLI-R10)."""
    assert TokenBucket.key("intervals_icu", "cred-A") != TokenBucket.key("intervals_icu", "cred-B")
    assert TokenBucket.key("intervals_icu", "cred-A") == TokenBucket.key("intervals_icu", "cred-A")


# -------------------------------------------------- CLI-R11 adaptive issue-rate reduction


def test_reduce_rate_persistently_lowers_the_issue_rate() -> None:
    """A 429 quota signal persistently lowers the bucket's refill rate (CLI-R11 adaptive)."""
    bucket = TokenBucket(rate_per_s=4.0, capacity=4.0, reduce_factor=0.5, min_rate=0.5)
    assert bucket.rate_per_s == pytest.approx(4.0)
    assert bucket.reduce_rate() == pytest.approx(2.0)  # one quota signal halves the rate
    assert bucket.rate_per_s == pytest.approx(2.0)  # ...and the reduction PERSISTS
    assert bucket.reduce_rate() == pytest.approx(1.0)  # a second signal halves again
    assert bucket.rate_per_s == pytest.approx(1.0)


def test_reduce_rate_is_floored_at_the_configured_minimum() -> None:
    """Repeated reductions never drop below the configured floor (CLI-R11 bounded)."""
    bucket = TokenBucket(rate_per_s=2.0, capacity=2.0, reduce_factor=0.1, min_rate=0.5)
    bucket.reduce_rate()  # 2.0 * 0.1 = 0.2 -> floored to 0.5
    assert bucket.rate_per_s == pytest.approx(0.5)
    bucket.reduce_rate()  # already at the floor; stays
    assert bucket.rate_per_s == pytest.approx(0.5)


def test_reduce_rate_fails_closed_without_config() -> None:
    """Without a configured factor+floor, reduce_rate raises — it never guesses (CFG-R1a)."""
    bucket = TokenBucket(rate_per_s=4.0, capacity=4.0)  # no reduce_factor/min_rate
    assert bucket.adaptive_reduction_enabled is False
    with pytest.raises(ValueError, match="reduce_factor and min_rate"):
        bucket.reduce_rate()
    assert bucket.rate_per_s == pytest.approx(4.0)  # unchanged — no silent no-op either


def test_reduce_rate_rejects_a_non_fraction_factor() -> None:
    """A factor outside (0, 1) is rejected — reduction must actually reduce (CLI-R11)."""
    bucket = TokenBucket(rate_per_s=4.0, capacity=4.0, reduce_factor=0.5, min_rate=0.5)
    with pytest.raises(ValueError, match="fraction in"):
        bucket.reduce_rate(1.5)
    with pytest.raises(ValueError, match="fraction in"):
        bucket.reduce_rate(0.0)


async def test_429_in_resilient_get_triggers_adaptive_reduction() -> None:
    """A 429 observed during retry fires on_quota_signal so the shared rate drops (CLI-R11)."""
    bucket = TokenBucket(rate_per_s=4.0, capacity=4.0, reduce_factor=0.5, min_rate=0.5)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=[])

    clock = _FakeClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await resilient_get(
            client, _URL, budget=_budget(), clock=clock, on_quota_signal=bucket.reduce_rate
        )
    assert resp.status_code == 200
    # The single 429 both waited Retry-After (the one-shot pause) AND persistently
    # reduced the SHARED bucket's issue rate for subsequent requests (CLI-R11).
    assert clock.slept == [1.0]
    assert bucket.rate_per_s == pytest.approx(2.0)


async def test_no_quota_signal_leaves_the_rate_untouched() -> None:
    """A non-429 transient (503) never reduces the rate — only a quota signal does (CLI-R11)."""
    bucket = TokenBucket(rate_per_s=4.0, capacity=4.0, reduce_factor=0.5, min_rate=0.5)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=[])

    clock = _FakeClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await resilient_get(
            client, _URL, budget=_budget(), clock=clock, on_quota_signal=bucket.reduce_rate
        )
    assert bucket.rate_per_s == pytest.approx(4.0)  # untouched — a 5xx is not a quota signal


# -------------------------------------------------------------- CLI-R5 client cancellation


async def test_cancellation_mid_retry_propagates_cleanly() -> None:
    """A CancelledError raised mid-retry propagates — never a half-built/returned value (CLI-R5)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)  # always transient -> the loop will sleep then retry

    class _CancelClock:
        """A clock whose sleep raises CancelledError, modelling a cancel during backoff."""

        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(asyncio.CancelledError):
            await resilient_get(client, _URL, budget=_budget(), clock=_CancelClock())
    # The cancel surfaced as CancelledError (NOT swallowed into a fabricated 200/empty
    # list, NOT converted to RetryExhaustedError): the cancellation is clean (CLI-R5).
    assert calls["n"] == 1  # cancelled during the first backoff, before a second attempt


# --------------------------------------------------------------- CLI-R2 schema drift


def test_schema_mismatch_raises_typed_fetch_error() -> None:
    """Validating drifted JSON raises FetchError(kind=SCHEMA_MISMATCH), not pydantic (CLI-R2)."""
    # The required ``id`` is absent -> drift. Must raise the TYPED engine error,
    # not a raw pydantic.ValidationError, and never a partially-coerced object.
    with pytest.raises(FetchError) as ei:
        validate_or_fetch_error(IntervalsActivityAsbo, {"type": "Ride"})
    assert ei.value.kind is FetchErrorKind.SCHEMA_MISMATCH
    assert ei.value.kind.value == "schema_mismatch"
    assert not isinstance(ei.value, pydantic.ValidationError)


def test_valid_payload_passes_validation() -> None:
    """A well-formed payload validates to the typed ASBO (CLI-R2 non-vacuous boundary)."""
    asbo = validate_or_fetch_error(IntervalsActivityAsbo, {"id": "a1", "type": "Ride"})
    assert asbo.id == "a1"


# ---------------------------------------------- CLI-R2/R6/AUT-R4 wired into the client


def _intervals_client(handler: Any) -> IntervalsIcuClient:
    """An IntervalsIcuClient whose budget retries fast (zero backoff) over a mock transport."""
    return IntervalsIcuClient(
        "k",
        "i0",
        base_url="https://intervals.icu",
        transport=httpx.MockTransport(handler),
        budget=AttemptBudget(
            max_attempts=3, max_elapsed_s=5.0, base_backoff_s=0.0, max_backoff_s=0.0
        ),
    )


async def test_client_probe_401_raises_typed_auth_error() -> None:
    """A revoked key (401) on probe raises AuthError, not httpx.HTTPStatusError (AUT-R4/CLI-R7)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    async with _intervals_client(handler) as client:
        with pytest.raises(AuthError) as ei:
            await client.probe()
    assert ei.value.kind in {FetchErrorKind.AUTH_REVOKED, FetchErrorKind.INSUFFICIENT_SCOPE}


async def test_client_probe_403_raises_typed_auth_error() -> None:
    """A 403 (insufficient scope) on probe surfaces as AuthError (AUT-R4/CLI-R7)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    async with _intervals_client(handler) as client:
        with pytest.raises(AuthError):
            await client.probe()


async def test_client_retries_transient_5xx_then_succeeds() -> None:
    """The client retries a 503 on a read GET then succeeds (CLI-R6 wired into the client)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=[{"id": "a1"}])

    async with _intervals_client(handler) as client:
        rows = await client.discover_activities("2026-01-01", "2026-12-31")
    assert rows == [{"id": "a1"}]
    assert calls["n"] == 2  # retried once


async def test_client_schema_drift_raises_typed_fetch_error() -> None:
    """A wellness row missing the required id raises FetchError(schema_mismatch) (CLI-R2)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"restingHR": 50}])  # the required ``id`` is absent

    async with _intervals_client(handler) as client:
        with pytest.raises(FetchError) as ei:
            await client.fetch_wellness("2026-05-01", "2026-06-06")
    assert ei.value.kind is FetchErrorKind.SCHEMA_MISMATCH
