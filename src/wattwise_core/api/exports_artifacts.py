"""Export artifact generation + the signed-download-URL primitives (§8.15, API-R34).

The focused sibling of :mod:`wattwise_core.api.routers.exports` (QUAL-R9 size split)
that owns the deterministic artifact builders (the athlete's canonical data rendered to
``json``/``csv``/``zip``) and the signed-URL mint/verify pair.

Artifacts are generated ON DEMAND from the stored job parameters — verbatim canonical
values, never recomputed/fabricated (GROUND-R7-adjacent honesty) and never duplicated
into the operational job row. The signed URL encodes the OWNING athlete, the job id, an
expiry, and the job's one-time nonce (API-R34); verification is constant-time and the
nonce single-use claim is the caller's atomic guarded UPDATE.

Requirement IDs: API-R34, API-R19 (portability export), AUTH-R15 (no provider name).
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import hmac
import io
import json
import uuid
import zipfile
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.persistence.models import Activity, FitnessSignature

#: Content types per export format (API-R34 download `Content-Type` per format).
CONTENT_TYPES: Final[dict[str, str]] = {
    "json": "application/json",
    "csv": "text/csv",
    "zip": "application/zip",
}

#: The flat per-activity columns the csv/json artifact carries (canonical names only).
_ACTIVITY_FIELDS: Final = (
    "activity_id",
    "sport",
    "start_time",
    "elapsed_time_s",
    "distance_m",
    "avg_power_w",
    "avg_hr_bpm",
)


def sign_download(key: str, *, athlete_id: str, job_id: str, exp: int, nonce: str) -> str:
    """The HMAC-SHA256 signature binding athlete + job + expiry + nonce (API-R34)."""
    payload = f"{athlete_id}|{job_id}|{exp}|{nonce}".encode()
    return hmac.new(key.encode(), payload, hashlib.sha256).hexdigest()


def verify_download(
    key: str,
    *,
    athlete_id: str,
    job_id: str,
    exp: int,
    nonce: str,
    sig: str,
    now: _dt.datetime,
) -> bool:
    """Constant-time verify a presented signed-URL tuple; expired/tampered -> ``False``.

    The caller separately enforces the nonce's SINGLE-USE property with an atomic
    guarded UPDATE on the job row — this check covers binding (owner/job/expiry/nonce)
    and freshness only.
    """
    if int(now.timestamp()) > exp:
        return False
    expected = sign_download(key, athlete_id=athlete_id, job_id=job_id, exp=exp, nonce=nonce)
    return hmac.compare_digest(expected, sig)


def _activity_row(activity: Activity) -> dict[str, object]:
    """One activity's flat export row — canonical fields only, no provider shape."""
    return {
        "activity_id": str(activity.activity_id),
        "sport": activity.sport,
        "start_time": activity.start_time.isoformat(),
        "elapsed_time_s": activity.elapsed_time_s,
        "distance_m": activity.distance_m,
        "avg_power_w": activity.avg_power_w,
        "avg_hr_bpm": activity.avg_hr_bpm,
    }


async def _activities(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    frm: str | None,
    to: str | None,
) -> list[dict[str, object]]:
    """The athlete's canonical activities in the window, oldest first (deterministic)."""
    stmt = (
        select(Activity)
        .where(Activity.athlete_id == athlete_id)
        .order_by(Activity.start_time.asc())
    )
    if frm:
        start = _dt.datetime.combine(_dt.date.fromisoformat(frm), _dt.time.min, _dt.UTC)
        stmt = stmt.where(Activity.start_time >= start)
    if to:
        end = _dt.datetime.combine(_dt.date.fromisoformat(to), _dt.time.max, _dt.UTC)
        stmt = stmt.where(Activity.start_time <= end)
    rows = (await session.execute(stmt)).scalars().all()
    return [_activity_row(a) for a in rows]


async def _signatures(
    session: AsyncSession, athlete_id: uuid.UUID
) -> list[dict[str, object]]:
    """The athlete's effective-dated fitness signatures, oldest first (analytics scope)."""
    stmt = (
        select(FitnessSignature)
        .where(FitnessSignature.athlete_id == athlete_id)
        .order_by(FitnessSignature.effective_date.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "signature_id": str(r.signature_id),
            "signature_type": str(r.signature_type),
            "effective_date": r.effective_date.isoformat(),
            "ftp_w": r.ftp_w,
            "cp_w": r.cp_w,
            "w_prime_j": r.w_prime_j,
            "threshold_hr_bpm": r.threshold_hr_bpm,
        }
        for r in rows
    ]


async def build_artifact(
    session: AsyncSession,
    *,
    athlete_id: uuid.UUID,
    scope: str,
    fmt: str,
    frm: str | None,
    to: str | None,
) -> bytes:
    """Render the export artifact for a job's stored parameters (API-R34 / API-R19).

    ``json`` is the canonical portability shape; ``csv`` flattens the activity rows;
    ``zip`` wraps the json document. Every value is the verbatim canonical one.
    """
    activities: list[dict[str, object]] = []
    payload: dict[str, object] = {}
    if scope in ("activities", "all"):
        activities = await _activities(session, athlete_id, frm, to)
        payload["activities"] = activities
    if scope in ("analytics", "all"):
        payload["analytics"] = {"fitness_signatures": await _signatures(session, athlete_id)}
    if fmt == "json":
        return json.dumps(payload, separators=(",", ":")).encode()
    if fmt == "csv":
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=_ACTIVITY_FIELDS)
        writer.writeheader()
        for row in activities:
            writer.writerow(row)
        return out.getvalue().encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("export.json", json.dumps(payload, separators=(",", ":")))
    return buffer.getvalue()


__all__ = [
    "CONTENT_TYPES",
    "build_artifact",
    "sign_download",
    "verify_download",
]
