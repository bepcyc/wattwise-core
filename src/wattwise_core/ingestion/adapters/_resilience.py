"""Outbound-client resilience primitives: retry/backoff + token-bucket (CLI-R6/R7/R10/R11, CLI-R2).

The typed-client contract (§4.2/§4.4) the OSS Intervals client wraps its GETs in:

* :func:`resilient_get` — issue a read GET with **exponential backoff + full jitter**,
  bounded by a per-source max attempt count AND a max elapsed budget (CLI-R6). Only
  TRANSIENT failures (429, 5xx, connection-reset, timeout) are retried; a non-transient
  4xx is returned as-is for the caller to convert to a typed error (CLI-R7). On a 429 the
  ``Retry-After`` header is honored over the computed backoff (CLI-R11). Retries are
  idempotency-safe: this wraps read-style GETs only (CLI-R8).
* :class:`TokenBucket` — a client-side rate limiter keyed per source+credential so the
  engine never relies on the source to reject excess traffic (CLI-R10); its state is
  shared across concurrent fetches for the same source+credential (CLI-R11).
* :func:`validate_or_fetch_error` — validate an ingress payload, converting a
  ``pydantic.ValidationError`` into the typed ``FetchError(kind=schema_mismatch)`` and
  never partially coercing a drifted payload into a GBO (CLI-R2).

This is adapter-private machinery (rank-0 L2): it imports only the rankless adapter seam
(:mod:`wattwise_core.ingestion.base`) plus stdlib/``httpx``/``pydantic`` — never the store,
analytics, the agent, or another adapter (ARCH-R8/ADP-R16).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pydantic

from wattwise_core.config import Settings
from wattwise_core.ingestion.base import FetchError, FetchErrorKind
from wattwise_core.observability import metrics as _metrics

#: The HTTP status the §4.2/§4.4 contract treats specially (transient + Retry-After).
_TooManyRequests = 429


class RetryExhaustedError(Exception):
    """Every retry attempt of a transient failure failed within the budget (CLI-R6).

    Raised when :func:`resilient_get` exhausts its per-source attempt count or elapsed
    budget on a transient failure (429/5xx/reset/timeout). The caller converts this to a
    typed gap / DEGRADED outcome (it is the recoverable, self-healing case — never an
    auth break, which surfaces as :class:`~wattwise_core.ingestion.base.AuthError`).
    """


@dataclass(frozen=True, slots=True)
class AttemptBudget:
    """The per-source retry budget (CLI-R6): bounded attempts AND bounded elapsed time.

    ``max_attempts`` caps the number of tries; ``max_elapsed_s`` caps the total wall time
    spent (including backoff sleeps). Backoff grows ``base_backoff_s * 2**i`` capped at
    ``max_backoff_s``; the actual sleep is full-jitter in ``[0, ceiling]``. All four are
    declared per source (loaded from config, CFG-R1a) — there is no value default here.
    """

    max_attempts: int
    max_elapsed_s: float
    base_backoff_s: float
    max_backoff_s: float


@dataclass(slots=True)
class _Clock:
    """The injectable monotonic clock + async sleep (so backoff is deterministic in tests)."""

    monotonic: Callable[[], float]
    sleep: Callable[[float], Awaitable[None]]


def _default_clock() -> _Clock:
    return _Clock(monotonic=time.monotonic, sleep=asyncio.sleep)


def _is_transient_status(status_code: int) -> bool:
    """A 429 or any 5xx is transient (retryable per §4.2); a 4xx<429 is not (CLI-R6/R7)."""
    return status_code == _TooManyRequests or 500 <= status_code <= 599


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a numeric-seconds ``Retry-After`` header, ``None`` if absent/unparseable (CLI-R11)."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _backoff_ceiling(budget: AttemptBudget, attempt_index: int) -> float:
    """The exponential backoff ceiling for ``attempt_index`` (0-based), capped (CLI-R6)."""
    return min(budget.base_backoff_s * (2.0**attempt_index), budget.max_backoff_s)


async def resilient_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    budget: AttemptBudget,
    params: dict[str, Any] | None = None,
    clock: Any = None,
    jitter: Callable[[], float] | None = None,
    on_quota_signal: Callable[[], object] | None = None,
) -> httpx.Response:
    """GET ``url`` with exponential-backoff-+-full-jitter retry on transient faults (CLI-R6/R7/R11).

    Retries ONLY transient failures — HTTP 429, 5xx, connection reset, and timeout —
    bounded by ``budget`` (attempts AND elapsed). A non-transient response (a 4xx other
    than 429) is returned immediately for the caller to convert to a typed error (CLI-R7);
    it is never retried. On a 429, a numeric ``Retry-After`` overrides the computed
    backoff (CLI-R11). Raises :class:`RetryExhaustedError` if the budget is exhausted on a
    transient failure. ``jitter`` returns a fraction in ``[0, 1]`` of the backoff ceiling
    (full jitter); it defaults to :func:`random.random`.

    ``on_quota_signal`` (when supplied) is invoked ONCE for each observed ``429`` so the
    caller can adaptively reduce the SHARED limiter's issue rate (CLI-R11) — this is the
    persistent issue-rate reduction, distinct from the one-shot ``Retry-After`` pause.
    """
    clk: _Clock = clock_to_clock(clock)
    draw = jitter or random.random
    started = clk.monotonic()
    last_exc: Exception | None = None
    for attempt in range(budget.max_attempts):
        response: httpx.Response | None = None
        try:
            response = await client.get(url, params=params)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc  # transient transport fault (reset/timeout) -> retry
        else:
            if response.status_code == _TooManyRequests and on_quota_signal is not None:
                on_quota_signal()  # adaptively reduce the shared limiter's rate (CLI-R11)
            if not _is_transient_status(response.status_code):
                return response  # success or non-transient 4xx — surface as-is (CLI-R7)
        if attempt + 1 >= budget.max_attempts:
            break
        wait = _next_wait(budget, attempt, response, draw)
        if clk.monotonic() - started + wait > budget.max_elapsed_s:
            break  # the elapsed budget would be exceeded — stop (CLI-R6)
        await clk.sleep(wait)
    raise RetryExhaustedError(
        f"transient failure not resolved within budget "
        f"(attempts<={budget.max_attempts}, elapsed<={budget.max_elapsed_s}s)"
    ) from last_exc


def _next_wait(
    budget: AttemptBudget,
    attempt: int,
    response: httpx.Response | None,
    draw: Callable[[], float],
) -> float:
    """The wait before the next attempt: ``Retry-After`` on a 429 else full-jitter backoff."""
    if response is not None and response.status_code == _TooManyRequests:
        retry_after = _retry_after_seconds(response)
        if retry_after is not None:
            return retry_after
    return _backoff_ceiling(budget, attempt) * draw()


def clock_to_clock(clock: Any) -> _Clock:
    """Coerce an injected clock-like object (``.monotonic``/``.sleep``) to a :class:`_Clock`."""
    if clock is None:
        return _default_clock()
    if isinstance(clock, _Clock):
        return clock
    return _Clock(monotonic=clock.monotonic, sleep=clock.sleep)


class TokenBucket:
    """A client-side token-bucket rate limiter keyed per source+credential (CLI-R10/R11).

    Tokens refill continuously at ``rate_per_s`` up to ``capacity``; :meth:`acquire`
    waits (using the injected sleep) when empty rather than relying on the source to
    reject excess (CLI-R10). One bucket instance is SHARED across concurrent fetches for
    the same source+credential, so its state coordinates the combined issue rate
    (CLI-R11). ``rate_per_s`` / ``capacity`` are declared per source (config, CFG-R1a).

    On a ``429`` / quota signal the client calls :meth:`reduce_rate`, which PERSISTENTLY
    lowers the ongoing refill ``rate_per_s`` by a configured factor (floored at a
    configured minimum) so SUBSEQUENT requests through the shared bucket are issued more
    slowly — the adaptive issue-rate reduction CLI-R11 mandates, distinct from the
    one-shot ``Retry-After`` pause of a single retry. The reduction factor and floor are
    config (CFG-R1a) injected at construction; if they are absent, :meth:`reduce_rate`
    fails closed (raises) rather than guessing a value.
    """

    __slots__ = (
        "_capacity",
        "_clock",
        "_lock",
        "_metrics_source",
        "_min_rate",
        "_rate",
        "_reduce_factor",
        "_tokens",
        "_updated",
    )

    def __init__(
        self,
        rate_per_s: float,
        capacity: float,
        *,
        clock: Callable[[], float] | None = None,
        reduce_factor: float | None = None,
        min_rate: float | None = None,
        metrics_source: str | None = None,
    ) -> None:
        # ING-OBS-R2: when a source label is given, limiter waits are recorded as the
        # per-source rate-limit-utilization metric (an unlabeled bucket records nothing).
        self._metrics_source = metrics_source
        self._rate = rate_per_s
        self._capacity = capacity
        self._clock = clock or time.monotonic
        self._tokens = capacity  # start full
        self._updated = self._clock()
        self._lock = asyncio.Lock()
        # CLI-R11 adaptive-reduction config (CFG-R1a): no code-default value — absent
        # means reduce_rate fails closed (the limiter still throttles at its base rate).
        self._reduce_factor = reduce_factor
        self._min_rate = min_rate

    @staticmethod
    def key(source_key: str, credential_ref: str | None) -> tuple[str, str | None]:
        """The shared-state key: per source AND per credential (CLI-R10)."""
        return (source_key, credential_ref)

    @property
    def rate_per_s(self) -> float:
        """The CURRENT refill rate (lowered by any prior :meth:`reduce_rate`; CLI-R11)."""
        return self._rate

    @property
    def adaptive_reduction_enabled(self) -> bool:
        """True when adaptive 429 rate-reduction is configured (CLI-R11/CFG-R1a).

        When the per-source ``reduce_factor`` + ``min_rate`` are present a 429 lowers the
        issue rate; when absent the limiter still throttles at its base rate and a quota
        signal does NOT attempt a reduction (so the unconfigured path never raises).
        """
        return self._reduce_factor is not None and self._min_rate is not None

    def reduce_rate(self, factor: float | None = None) -> float:
        """Adaptively lower the ongoing issue rate after a 429/quota signal (CLI-R11).

        Multiplies the live refill ``rate_per_s`` by ``factor`` (a fraction in ``(0, 1)``
        — config CFG-R1a; falls back to the construction-time ``reduce_factor`` when
        omitted) and floors the result at the configured ``min_rate`` so subsequent
        acquires through this SHARED bucket throttle harder. The reduction is persistent
        for the bucket's lifetime — it does not auto-restore. Returns the new rate.

        Fails closed (``ValueError``) if neither a ``factor`` argument nor a configured
        ``reduce_factor`` is present, or if ``min_rate`` was not configured, or if the
        factor is not in ``(0, 1)`` — never silently no-ops on a quota signal.
        """
        chosen = factor if factor is not None else self._reduce_factor
        if chosen is None or self._min_rate is None:
            raise ValueError("reduce_rate requires a configured reduce_factor and min_rate")
        if not 0.0 < chosen < 1.0:
            raise ValueError("reduce_rate factor must be a fraction in (0, 1)")
        self._rate = max(self._min_rate, self._rate * chosen)
        return self._rate

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._updated = now

    async def acquire(self, sleep: Callable[[float], Awaitable[None]] | None = None) -> None:
        """Acquire one token, waiting (via ``sleep``) for a refill if empty (CLI-R10)."""
        wait_for = asyncio.sleep if sleep is None else sleep
        async with self._lock:
            self._refill()
            if self._tokens < 1.0:
                deficit = 1.0 - self._tokens
                wait = deficit / self._rate if self._rate > 0 else 0.0
                # ING-OBS-R2 / CLI-R12: rate-limit waits are observable per source.
                if self._metrics_source is not None:
                    _metrics.get_registry().observe(
                        _metrics.INGEST_RATE_LIMIT_WAIT,
                        wait,
                        labels={"source_key": self._metrics_source},
                    )
                await wait_for(wait)
                self._refill()
            self._tokens -= 1.0


def intervals_icu_budget(settings: Settings) -> AttemptBudget:
    """Build the Intervals.icu retry budget FROM config (CLI-R6, CFG-R1a).

    Every value comes from the ``adapters.intervals_icu`` config section (defaults.toml,
    overridable per layer) — NO attempt/backoff literal is baked into code (CFG-R1a).
    """
    return AttemptBudget(
        max_attempts=settings.adapters__intervals_icu__budget_max_attempts,
        max_elapsed_s=settings.adapters__intervals_icu__budget_max_elapsed_s,
        base_backoff_s=settings.adapters__intervals_icu__budget_base_backoff_s,
        max_backoff_s=settings.adapters__intervals_icu__budget_max_backoff_s,
    )


def intervals_icu_bucket(settings: Settings) -> TokenBucket:
    """Build the Intervals.icu client-side token bucket FROM config (CLI-R10/R11, CFG-R1a).

    The base issue rate + capacity and the adaptive-429 ``reduce_factor`` / ``min_rate``
    (CLI-R11) all come from config — NO rate literal is baked into code (CFG-R1a). With the
    reduction params configured, a quota signal persistently lowers the rate rather than
    failing closed.
    """
    return TokenBucket(
        settings.adapters__intervals_icu__bucket_rate_per_s,
        settings.adapters__intervals_icu__bucket_capacity,
        reduce_factor=settings.adapters__intervals_icu__bucket_reduce_factor,
        min_rate=settings.adapters__intervals_icu__bucket_min_rate,
        metrics_source="intervals_icu",
    )


def validate_or_fetch_error[M: pydantic.BaseModel](model: type[M], payload: Any) -> M:
    """Validate ``payload`` into ``model`` or raise ``FetchError(kind=schema_mismatch)`` (CLI-R2).

    A response that fails validation MUST raise the typed engine error rather than a raw
    ``pydantic.ValidationError``, and MUST NOT be partially coerced into a GBO. The raw
    validation message is NOT carried (it may echo response content); a generic
    non-sensitive detail is used (AUT-R2/ING-SEC-R3).
    """
    try:
        return model.model_validate(payload)
    except pydantic.ValidationError as exc:
        raise FetchError(
            FetchErrorKind.SCHEMA_MISMATCH,
            f"{model.__name__} failed ingress validation",
        ) from exc


__all__ = [
    "AttemptBudget",
    "RetryExhaustedError",
    "TokenBucket",
    "intervals_icu_bucket",
    "intervals_icu_budget",
    "resilient_get",
    "validate_or_fetch_error",
]
