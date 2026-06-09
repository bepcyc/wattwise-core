"""Per-athlete request-rate limiting — the OSS token-bucket baseline (LIMIT-R*).

Every endpoint is rate-limited per athlete/owner (LIMIT-R1). This module owns the
OSS request-rate baseline: a per-(athlete, class) token bucket whose refill rate is
the per-minute ceiling for that endpoint class (LIMIT-R2 — general read ``120/min``,
mutating ``30/min``, ``agent`` ``20/min``). Exceeding the rate yields ``429``
``rate-limited`` (ERR-R8) carrying ``Retry-After`` and the IETF ``RateLimit-Limit`` /
``RateLimit-Remaining`` / ``RateLimit-Reset`` headers (LIMIT-R3).

The bucket key is the SERVER-DERIVED athlete id (AUTH-R3) — never a client header —
so the limit cannot be bypassed by omitting or forging a header (LIMIT-R6). This is
the OSS baseline ONLY: it enforces the request rate and carries NO monetary cost
budget and never raises ``cost-limit-exceeded`` (that reserve-then-settle cost gate
is commercial, doc 90 COMM-R20 / API-R11b; it never fires in OSS — LIMIT-R1/R2).

A token bucket (not a fixed window) gives a smooth limit with a small burst
allowance equal to the per-minute ceiling, and a deterministic ``RateLimit-Reset``
(seconds until one token is available again). The store is in-process and bounded by
the active athlete set; the commercial layer swaps this seam for a shared store
without changing the contract.

Requirement IDs: LIMIT-R1 (per-athlete rate limit, OSS baseline), LIMIT-R2 (the
read/mutating/agent ceilings), LIMIT-R3 (``429`` + ``Retry-After`` + ``RateLimit-*``
headers), LIMIT-R6 (server-side, header-forge-proof), AUTH-R3 (server-derived key),
ERR-R8 (``rate-limited`` problem type).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from wattwise_core.api.errors import ProblemError

#: Seconds in the rolling rate window the per-minute ceilings are expressed over.
_WINDOW_SECONDS: Final = 60.0


class LimitClass(StrEnum):
    """The endpoint classes with distinct per-minute request ceilings (LIMIT-R2)."""

    READ = "read"
    MUTATING = "mutating"
    AGENT = "agent"


#: Fallback per-athlete per-minute REQUEST ceilings for an isolated ``RateLimiter()`` constructed
#: with NO settings (a pure unit test that exercises the bucket mechanics, not the config wiring).
#: This is NOT the PRODUCTION source of the ceilings (that would violate CFG-R1a): the app factory
#: builds the limiter from the layered config — ``read``/``mutating`` from the ``[ratelimit]`` table
#: and ``agent`` from the entitlement-governed ``entitlement.request_rate_per_minute`` — so no
#: production rate value is a code literal. The bucket capacity (max burst) equals the per-minute
#: rate for each class.
DEFAULT_LIMITS: Final[dict[LimitClass, int]] = {
    LimitClass.READ: 120,
    LimitClass.MUTATING: 30,
    LimitClass.AGENT: 20,
}


@dataclass(slots=True)
class _Bucket:
    """A single athlete/class token bucket: ``tokens`` available, last refill time."""

    tokens: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class RateLimitHeaders:
    """The IETF ``RateLimit-*`` + ``Retry-After`` header set for a decision (LIMIT-R3).

    Emitted on the ``429`` problem (and available for a success response). Values are
    integers per the IETF RateLimit header fields; ``retry_after`` and ``reset`` are
    whole seconds (ceil) so a client never under-waits.
    """

    limit: int
    remaining: int
    reset: int
    retry_after: int | None = None

    def to_dict(self) -> dict[str, str]:
        """Render to the HTTP header mapping (string values, LIMIT-R3)."""
        headers = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(self.remaining),
            "RateLimit-Reset": str(self.reset),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class RateLimiter:
    """In-process per-(athlete, class) token-bucket rate limiter (LIMIT-R1/R2/R6).

    Thread-safe: a single lock guards the bucket map so concurrent requests for one
    athlete debit the same bucket atomically (LIMIT-R6 — not bypassable by racing).
    Keyed by the server-derived athlete id (AUTH-R3). ``check`` debits one token and
    either returns the success headers or raises ``429`` ``rate-limited``.
    """

    def __init__(
        self,
        limits: dict[LimitClass, int] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build a limiter with the given per-class ceilings (default LIMIT-R2).

        ``clock`` is injectable so tests advance time deterministically rather than
        sleeping. The bucket map starts empty and fills lazily per active athlete.
        """
        self._limits = dict(limits or DEFAULT_LIMITS)
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, LimitClass], _Bucket] = {}

    def _rate_per_second(self, limit_class: LimitClass) -> float:
        """Tokens regenerated per second for a class (its per-minute ceiling / 60)."""
        return self._limits[limit_class] / _WINDOW_SECONDS

    def _refill(self, bucket: _Bucket, limit_class: LimitClass, now: float) -> None:
        """Refill a bucket up to capacity for elapsed time (token-bucket refill)."""
        capacity = float(self._limits[limit_class])
        elapsed = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(capacity, bucket.tokens + elapsed * self._rate_per_second(limit_class))
        bucket.updated_at = now

    def check(self, athlete_id: str, limit_class: LimitClass) -> RateLimitHeaders:
        """Debit one token for ``(athlete_id, limit_class)``; raise ``429`` if empty.

        On success returns the ``RateLimit-*`` headers for the post-debit state. When
        the bucket is empty it raises ``ProblemError("rate-limited")`` carrying
        ``Retry-After`` + ``RateLimit-*`` (LIMIT-R3); the athlete-facing ``title``/
        ``detail`` come from the catalog and are warm/jargon-free (API-R21). Enforced
        server-side under the lock so it cannot be raced or header-forged (LIMIT-R6).
        """
        limit = self._limits[limit_class]
        with self._lock:
            now = self._clock()
            key = (athlete_id, limit_class)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(limit), updated_at=now)
                self._buckets[key] = bucket
            self._refill(bucket, limit_class, now)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return self._success_headers(bucket, limit_class)
            raise self._limited(bucket, limit_class)

    def _success_headers(self, bucket: _Bucket, limit_class: LimitClass) -> RateLimitHeaders:
        """Build the success ``RateLimit-*`` headers from the post-debit bucket."""
        limit = self._limits[limit_class]
        remaining = math.floor(bucket.tokens)
        return RateLimitHeaders(
            limit=limit,
            remaining=remaining,
            reset=self._seconds_to_full_token(bucket, limit_class),
        )

    def _limited(self, bucket: _Bucket, limit_class: LimitClass) -> ProblemError:
        """Build the ``429 rate-limited`` problem with retry/RateLimit headers (LIMIT-R3)."""
        wait = self._seconds_to_full_token(bucket, limit_class)
        headers = RateLimitHeaders(
            limit=self._limits[limit_class],
            remaining=0,
            reset=wait,
            retry_after=wait,
        )
        return ProblemError("rate-limited", headers=headers.to_dict())

    def _seconds_to_full_token(self, bucket: _Bucket, limit_class: LimitClass) -> int:
        """Whole seconds until at least one token is available (ceil, never < 1)."""
        if bucket.tokens >= 1.0:
            return 0
        deficit = 1.0 - bucket.tokens
        seconds = deficit / self._rate_per_second(limit_class)
        return max(1, math.ceil(seconds))


__all__ = [
    "DEFAULT_LIMITS",
    "LimitClass",
    "RateLimitHeaders",
    "RateLimiter",
]
