"""The fitness-signature write seam (GBO-R27/R28).

The ONLY sanctioned way to record a new effective-dated signature. It enforces the
GBO-R27 interval invariants at write time, portably across all three backends (no
backend-specific exclusion constraint is required):

* intervals for a ``(athlete_id, signature_type)`` scope never overlap — a new
  signature must be dated AFTER every existing one in its scope; an out-of-order
  write is REFUSED loudly (fail-closed), never silently reordered or overwritten;
* at most one open interval (``effective_to IS NULL``) exists per scope — recording a
  new signature CLOSES the prior open interval at the new effective date rather than
  overwriting it, so threshold history is preserved;
* a MODELED signature MUST carry ``fit_quality`` (GBO-R28) — recording one without
  it is refused, since the analytics layer could never fit-gate it.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import SignatureOrigin
from wattwise_core.persistence.models import FitnessSignature


class SignatureIntervalError(ValueError):
    """A write violating the GBO-R27/R28 signature invariants (refused, fail-closed)."""


def _refuse_unfit_modeled(origin: SignatureOrigin, fit_quality: dict[str, object] | None) -> None:
    """Refuse a MODELED signature carrying no ``fit_quality`` (GBO-R28, fail-closed)."""
    if origin == SignatureOrigin.MODELED and not fit_quality:
        raise SignatureIntervalError(
            "a MODELED signature MUST carry fit_quality (GBO-R28); refusing to record one"
            " the analytics layer could never fit-gate"
        )


async def _refuse_overlapping(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    signature_type: str,
    effective_date: _dt.date,
) -> None:
    """Refuse a write not dated strictly AFTER every existing one in scope (GBO-R27)."""
    latest = (
        await session.execute(
            select(FitnessSignature)
            .where(
                FitnessSignature.athlete_id == athlete_id,
                FitnessSignature.signature_type == signature_type,
            )
            .order_by(FitnessSignature.effective_date.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is not None and latest.effective_date >= effective_date:
        raise SignatureIntervalError(
            f"signature for {signature_type!r} effective {effective_date} would overlap "
            f"the existing interval starting {latest.effective_date} (GBO-R27: intervals "
            "must not overlap; out-of-order writes are refused, never reordered)"
        )


async def _close_open_intervals(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    signature_type: str,
    effective_date: _dt.date,
) -> None:
    """CLOSE the prior open interval at the new effective date, never overwrite (GBO-R27)."""
    open_rows = (
        (
            await session.execute(
                select(FitnessSignature).where(
                    FitnessSignature.athlete_id == athlete_id,
                    FitnessSignature.signature_type == signature_type,
                    FitnessSignature.effective_to.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    closure = _dt.datetime.combine(effective_date, _dt.time.min, tzinfo=_dt.UTC)
    for row in open_rows:
        row.effective_to = closure  # close, never overwrite (GBO-R27)


async def record_signature(
    session: AsyncSession,
    *,
    athlete_id: uuid.UUID,
    signature_type: str,
    effective_date: _dt.date,
    origin: SignatureOrigin,
    cp_w: float | None = None,
    w_prime_j: float | None = None,
    ftp_w: float | None = None,
    threshold_hr_bpm: int | None = None,
    max_hr_bpm: int | None = None,
    resting_hr_bpm: int | None = None,
    vo2max: float | None = None,
    fit_quality: dict[str, object] | None = None,
) -> FitnessSignature:
    """Record a new effective-dated signature, closing the prior open interval (GBO-R27).

    The new signature must be dated strictly AFTER every existing signature in its
    ``(athlete_id, signature_type)`` scope — given each prior write held the invariant,
    this keeps the intervals non-overlapping with at most one open row. The prior open
    interval is CLOSED (``effective_to`` = the new effective date's UTC midnight), never
    overwritten. A modeled signature without ``fit_quality`` is refused (GBO-R28).
    """
    _refuse_unfit_modeled(origin, fit_quality)
    await _refuse_overlapping(session, athlete_id, signature_type, effective_date)
    await _close_open_intervals(session, athlete_id, signature_type, effective_date)
    signature = FitnessSignature(
        athlete_id=athlete_id,
        signature_type=signature_type,
        effective_date=effective_date,
        effective_to=None,
        origin=origin,
        cp_w=cp_w,
        w_prime_j=w_prime_j,
        ftp_w=ftp_w,
        threshold_hr_bpm=threshold_hr_bpm,
        max_hr_bpm=max_hr_bpm,
        resting_hr_bpm=resting_hr_bpm,
        vo2max=vo2max,
        fit_quality=fit_quality,
    )
    session.add(signature)
    await session.flush()
    return signature


__all__ = ["SignatureIntervalError", "record_signature"]
