"""Canonical local-date projection — the single owner of instant→local-day (doc 20 §3.8).

WattWise turns UTC instants into calendar days by projecting the instant into the athlete's
**reference timezone** (``athlete.reference_timezone``), using the timezone in effect at that
instant per the as-of metadata ``reference_timezone_effective_from`` (GBO-R33). The projection
is reproducible: the same UTC instant plus the same as-of tz metadata always yields the same
``local_date`` (GBO-R34) — the UTC instant stays the source of truth, never a stored local
instant (GBO-R32). The reference timezone, not the device/source timezone, is authoritative,
so a single trip abroad does not scatter a day across two buckets.

Fail-closed (CFG-R1a / CFG-R6): the reference timezone is DATA (the athlete profile), never a
literal baked into code. A missing/blank/unresolvable zone raises :class:`MissingReferenceTimezone`
— the engine NEVER silently falls back to a code-baked ``UTC`` default. Stdlib ``zoneinfo`` is
the only tz source (full IANA history incl. DST transitions and historical offset changes).
"""

from __future__ import annotations

import datetime as _dt
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class MissingReferenceTimezone(ValueError):
    """The athlete has no resolvable reference timezone, so day-bucketing fails closed.

    Raised when ``reference_timezone`` is absent/blank or not a resolvable IANA zone. Per
    CFG-R1a/CFG-R6 the engine must refuse to bucket rather than assume a code-baked default
    (e.g. ``UTC``); the caller isolates the record / surfaces the gap (GBO-R7/GAP-R1).
    """


class _TzAthlete(Protocol):
    """The minimal as-of tz surface a projection target must expose (GBO-R13/R33/R34).

    ``reference_timezone`` mirrors the ``Athlete`` column (non-nullable ``str``); a blank /
    unresolvable value still fails closed at the seam below (CFG-R6), so the absence path is
    handled at runtime rather than by an optional type.
    """

    reference_timezone: str
    reference_timezone_effective_from: _dt.datetime | None


def _as_utc(instant: _dt.datetime) -> _dt.datetime:
    """Read a stored instant as tz-aware UTC (GBO-R32: every stored instant is UTC)."""
    return instant if instant.tzinfo else instant.replace(tzinfo=_dt.UTC)


def _resolve_zone(athlete: _TzAthlete) -> ZoneInfo:
    """Resolve the athlete's reference timezone, failing closed when unusable (CFG-R6).

    The reference timezone is athlete DATA, never a code literal (CFG-R1a). A ``None``/blank
    value or an unresolvable IANA key raises :class:`MissingReferenceTimezone` — there is NO
    silent ``UTC`` fallback.
    """
    name = athlete.reference_timezone
    if name is None or not name.strip():
        raise MissingReferenceTimezone("athlete has no reference timezone configured")
    try:
        return ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError) as exc:  # unresolvable / malformed key
        raise MissingReferenceTimezone(f"unresolvable reference timezone: {name!r}") from exc


def project_local_wall_clock(instant: _dt.datetime, athlete: _TzAthlete) -> _dt.datetime:
    """Project a UTC instant to the athlete-LOCAL wall-clock datetime (``start_time_local``).

    Returns the athlete's LOCAL wall-clock fields carried on a tz-aware datetime. The
    ``start_time_local`` column is a ``timestamptz`` that normalizes any stored value to UTC
    (``UtcDateTime``), so a value tagged with the reference zone's offset would be re-coerced
    back to the UTC instant and lose its local-ness on read. To keep ``start_time_local`` an
    honest "derived display" of the LOCAL time (GBO-R13, §3.8) across all three backends, the
    local wall-clock numbers are carried with a UTC tzinfo: the stored value then reads back
    with the athlete's local hour/day (and ``.date()`` equals ``local_date``), never the UTC
    instant. The UTC instant of record stays ``start_time`` (GBO-R32). Fails closed without a
    resolvable tz (CFG-R6).
    """
    local = _as_utc(instant).astimezone(_resolve_zone(athlete))
    return local.replace(tzinfo=_dt.UTC)


def project_local_date(
    instant: _dt.datetime,
    athlete: _TzAthlete,
    *,
    prior_local_date: _dt.date | None = None,
) -> _dt.date:
    """The athlete-LOCAL calendar date of a UTC instant (GBO-R33/R34) — fail-closed.

    Projects ``instant`` into the reference timezone in effect at that instant and returns
    its calendar date (GBO-R33). The reference timezone is effective-dated: an instant at or
    after ``reference_timezone_effective_from`` uses the current zone; an instant BEFORE a
    non-NULL ``reference_timezone_effective_from`` predates the current zone, so the current
    zone is NOT authoritative for it. To honour GBO-R34 ("a later relocation MUST NOT
    retroactively re-bucket prior days") a ``prior_local_date`` — the bucket the record
    already carries from when it was first projected under the then-current zone — is returned
    unchanged for such pre-relocation instants. When no prior projection is supplied the
    current zone is used (the first-ingest case, where the instant is at/after the current
    effective_from). A missing/unresolvable tz fails closed (CFG-R6).
    """
    utc_instant = _as_utc(instant)
    zone = _resolve_zone(athlete)  # fail-closed BEFORE any as-of branch (no silent default)
    eff = athlete.reference_timezone_effective_from
    if eff is not None and prior_local_date is not None and utc_instant < _as_utc(eff):
        # Pre-relocation instant with a persisted projection: keep it (GBO-R34, no re-bucket).
        return prior_local_date
    return utc_instant.astimezone(zone).date()


__all__ = [
    "MissingReferenceTimezone",
    "project_local_date",
    "project_local_wall_clock",
]
