"""Intervals.icu thin typed HTTP client (CLI-R1/R3/R4, CLI-R13) — impure I/O only.

A focused split of the adapter's client side (QUAL-R9): the resilient, token-bucketed
``httpx.AsyncClient`` wrapper that owns probe/discover/fetch network calls and the
typed-error conversion (AuthError on 401/403, FetchError otherwise). It owns NO
mapping logic; the pure map and the five-phase contract live on the adapter.
Construction takes an injectable ``transport`` so every call is exercisable offline
against recorded fixtures (CLI-R3, TST-R1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

import httpx

from wattwise_core.config import Settings
from wattwise_core.ingestion.adapters import _intervals_map as _im
from wattwise_core.ingestion.adapters._intervals_asbo import (
    ActivityWithStreams,
    IntervalsActivityAsbo,
    IntervalsStreamAsbo,
    IntervalsWellnessAsbo,
)
from wattwise_core.ingestion.adapters._resilience import (
    AttemptBudget,
    RetryExhaustedError,
    TokenBucket,
    intervals_icu_bucket,
    intervals_icu_budget,
    resilient_get,
    validate_or_fetch_error,
)
from wattwise_core.ingestion.base import (  # noqa: import-direction
    AuthError,
    FetchError,
    FetchErrorKind,
)
from wattwise_core.observability import metrics as _metrics

_BASE_URL: Final = "https://intervals.icu"
_BASIC_USERNAME: Final = "API_KEY"  # literal username per CLI-R13 (NOT a secret)


class IntervalsIcuClient:
    """Thin typed ``httpx.AsyncClient`` for Intervals.icu (impure I/O; CLI-R1/R3).

    Construction takes an injectable ``transport`` so tests substitute a fixture
    transport with no live network (CLI-R3, TST-R1). All calls are bounded by a
    connect + total timeout (CLI-R4). This object owns NO mapping logic.
    """

    def __init__(
        self,
        api_key: str,
        athlete_id: str,
        *,
        base_url: str = _BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
        budget: AttemptBudget,
        bucket: TokenBucket | None = None,
    ) -> None:
        self._athlete_id = athlete_id
        self._budget = budget
        self._bucket = bucket
        self._client = httpx.AsyncClient(
            base_url=base_url,
            auth=httpx.BasicAuth(_BASIC_USERNAME, api_key),
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    @classmethod
    def from_settings(
        cls,
        api_key: str,
        athlete_id: str,
        settings: Settings,
        *,
        base_url: str = _BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        bucket: TokenBucket | None = None,
    ) -> IntervalsIcuClient:
        """Build the production client with its resilience read FROM config (CFG-R1a).

        The per-source retry budget (CLI-R6) + the request timeout (CLI-R4) come from the
        ``adapters.intervals_icu`` config section — NO resilience literal is baked into code
        (CFG-R1a: a value absent from every layer fails closed at load, never a code default).
        When no shared ``bucket`` is supplied, the client-side token-bucket limiter (CLI-R10/R11)
        is also built from config, with the adaptive-429 ``reduce_factor`` / ``min_rate`` set so a
        quota signal persistently lowers the issue rate rather than failing closed. The
        budget/bucket assembly lives in :mod:`_resilience` (its primitives' home).
        """
        return cls(
            api_key,
            athlete_id,
            base_url=base_url,
            transport=transport,
            timeout=settings.adapters__intervals_icu__http_timeout_s,
            budget=intervals_icu_budget(settings),
            bucket=bucket if bucket is not None else intervals_icu_bucket(settings),
        )

    async def __aenter__(self) -> IntervalsIcuClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """Issue a resilient read GET: token-bucket throttle + retry/backoff + typed errors.

        The per-source token bucket (CLI-R10) gates issue rate client-side; the GET is
        retried on a transient 429/5xx/reset/timeout with exponential backoff + full
        jitter, bounded by the per-source budget (CLI-R6), honoring ``Retry-After`` on a
        429 (CLI-R11). A 401/403 becomes a typed :class:`AuthError` (AUT-R4); any other
        non-2xx (incl. a budget-exhausted transient) becomes a typed :class:`FetchError`
        (CLI-R7) — never a raw ``httpx`` exception leaking to the engine.
        """
        resp = await self._raw_get(path, params=params)
        if resp.status_code >= 400:
            raise FetchError(FetchErrorKind.FETCH_FAILED, f"source returned {resp.status_code}")
        return resp

    async def _raw_get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """Resilient GET that still converts auth/transient faults but RETURNS a non-auth 4xx.

        Used for the optional streams endpoint, where a ``404`` legitimately means "no
        streams" rather than a failure: the auth (AUT-R4) and budget-exhaustion (CLI-R7)
        conversions still apply, but a non-auth 4xx is returned for the caller to treat as
        empty. The token bucket (CLI-R10) and retry/backoff budget (CLI-R6) gate the call;
        on a 429 the shared bucket's issue rate is adaptively reduced (CLI-R11).
        """
        # ING-OBS-R2: outbound request count (cost) is observable per source.
        _metrics.get_registry().increment(
            _metrics.INGEST_OUTBOUND_REQUESTS, labels={"source_key": "intervals_icu"}
        )
        if self._bucket is not None:
            await self._bucket.acquire()
        bucket = self._bucket
        on_quota: Callable[[], object] | None = (
            bucket.reduce_rate  # CLI-R11: a 429 lowers the shared bucket's issue rate
            if bucket is not None and bucket.adaptive_reduction_enabled
            else None
        )
        try:
            resp = await resilient_get(
                self._client, path, budget=self._budget, params=params, on_quota_signal=on_quota
            )
        except RetryExhaustedError as exc:
            raise FetchError(
                FetchErrorKind.SOURCE_UNAVAILABLE, "source unavailable after retries"
            ) from exc
        if resp.status_code in (401, 403):
            raise AuthError(
                FetchErrorKind.AUTH_REVOKED
                if resp.status_code == 401
                else FetchErrorKind.INSUFFICIENT_SCOPE,
                "source credential rejected",
            )
        return resp

    async def probe(self) -> Mapping[str, Any]:
        """Mandatory read-only credential probe (AUT-R17): GET the athlete profile.

        Returns the profile mapping on success; raises a typed :class:`AuthError` on a
        401/403 (AUT-R4) so the caller never marks the connection ``connected``.
        """
        resp = await self._get(f"/api/v1/athlete/{self._athlete_id}")
        result: Mapping[str, Any] = resp.json()
        return result

    async def discover_activities(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        """List activity summaries in an ISO-date window (ADP-R5; oldest/newest)."""
        resp = await self._get(
            f"/api/v1/athlete/{self._athlete_id}/activities",
            params={"oldest": oldest, "newest": newest},
        )
        payload: list[dict[str, Any]] = resp.json()
        return payload

    async def fetch_activity(self, activity_id: str) -> ActivityWithStreams:
        """Fetch one activity detail + its streams as a validated ASBO (ADP-R8, CLI-R2)."""
        detail = await self._get(f"/api/v1/activity/{activity_id}")
        streams = await self._raw_get(
            f"/api/v1/activity/{activity_id}/streams",
            params={"types": ",".join(_im.STREAM_CHANNELS)},
        )
        stream_rows = streams.json() if streams.status_code == httpx.codes.OK else []
        return ActivityWithStreams(
            activity=validate_or_fetch_error(IntervalsActivityAsbo, detail.json()),
            streams=[validate_or_fetch_error(IntervalsStreamAsbo, s) for s in stream_rows],
        )

    async def fetch_wellness(self, oldest: str, newest: str) -> list[IntervalsWellnessAsbo]:
        """Fetch daily-wellness rows in an ISO-date window (ADP-R8, CLI-R2)."""
        resp = await self._get(
            f"/api/v1/athlete/{self._athlete_id}/wellness",
            params={"oldest": oldest, "newest": newest},
        )
        rows: list[dict[str, Any]] = resp.json()
        return [validate_or_fetch_error(IntervalsWellnessAsbo, r) for r in rows]


__all__ = ["IntervalsIcuClient"]
