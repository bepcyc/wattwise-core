"""Data-health router — the coverage/issues audit surface (``/v1/data-health``, §8.3).

The audit/operational surface of API-R10: unlike every consumer analytics endpoint it
MAY name a source (AUTH-R15 lists ``/v1/data-health/*`` as a documented exception),
because the athlete is auditing where their data comes from. All three reads derive
DETERMINISTICALLY from the same fail-closed canonical coverage diagnosis the agent
narrates (API-R15) — never an LLM, never a fabricated number:

- ``GET /v1/data-health/summary`` — completeness score (the real present-fraction of
  the closed check set) + counts + an athlete-native ``headline_text`` (API-R21).
- ``GET /v1/data-health/coverage-matrix`` — per-canonical-domain coverage with the
  connected sources listed; a domain whose % is not computable carries a typed
  ``null`` (ANL-R3/R4), never an invented percentage.
- ``GET /v1/data-health/issues`` — typed issues (closed ``severity``, SCHEMA-R3) with
  jargon-free ``message_text``.

``POST /v1/data-health/activity-linking`` (§8.3) — resolving/confirming a canonical
merge of overlapping observations — is NOT yet implemented: it requires the canonical
merge executor (doc 20 MAP-R9..R11) and is deliberately absent rather than shipped as
a destructive stub (accepted deviation; the read surfaces above are complete).

Requirement IDs: API-R10 (§8.3), API-R15 (shared checks), API-R21, AUTH-R3, AUTH-R11,
AUTH-R15 (documented exception), SCHEMA-R3 (severity), PAGE-R3/R4, LIMIT-R1.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from wattwise_core.agent.diagnose_deliverable import (
    DIAGNOSIS_WINDOW_DAYS,
    AgentDiagnosis,
    InputStatus,
    diagnose_coverage,
)
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.api.deps import DbSession, RateLimit
from wattwise_core.api.pagination import clamp_limit
from wattwise_core.api.routers.performance import (
    analytics_service,
    current_athlete_id,
    require_read_scope,
)
from wattwise_core.persistence.models import Connection, SourceDescriptor

router = APIRouter(prefix="/v1/data-health", tags=["data-health"], dependencies=[RateLimit])

_Read = Depends(require_read_scope)
Service = Annotated[AnalyticsService, Depends(analytics_service)]
AthleteId = Annotated[str, Depends(current_athlete_id)]


class DataHealthSummary(BaseModel):
    """``GET /v1/data-health/summary``: the overall completeness read (§8.3)."""

    model_config = ConfigDict(extra="forbid")

    completeness_score: float
    present_count: int
    stale_count: int
    missing_count: int
    headline_text: str
    as_of: str


class CoverageCell(BaseModel):
    """One canonical domain's coverage row in the matrix (§8.3).

    ``coverage_pct`` is computed ONLY where a real per-day basis exists (the training
    load domain's resolved daily series); otherwise it is a typed ``null`` — the matrix
    never invents a percentage (ANL-R3/R4). ``sources`` lists the connected source keys
    (this audit surface is the documented AUTH-R15 exception).
    """

    domain: str
    label: str
    status: Literal["present", "stale", "missing"]
    coverage_pct: float | None
    sources: list[str]


class CoverageMatrix(BaseModel):
    """``GET /v1/data-health/coverage-matrix``: per-domain coverage (§8.3)."""

    model_config = ConfigDict(extra="forbid")

    domains: list[CoverageCell]
    as_of: str


class DataHealthIssue(BaseModel):
    """One typed data-health issue (§8.3): closed ``severity`` + athlete copy."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str
    kind: str
    severity: Literal["info", "warning", "critical"]
    message_text: str


class IssuePage(BaseModel):
    """The PAGE-R4 page block of the issue list."""

    limit: int
    next_cursor: str | None = None
    has_more: bool


class DataHealthIssueList(BaseModel):
    """``GET /v1/data-health/issues``: the bounded issue page (PAGE-R3/R4)."""

    data: list[DataHealthIssue]
    page: IssuePage


@router.get(
    "/summary",
    response_model=DataHealthSummary,
    operation_id="getDataHealthSummary",
    dependencies=[_Read],
)
async def data_health_summary(svc: Service, athlete_id: AthleteId) -> DataHealthSummary:
    """The completeness summary: a REAL present-fraction over the closed check set.

    The score is ``present / total`` of the deterministic coverage checks (the same
    fail-closed probes API-R15 narrates) — a measured fraction, never a fabricated
    health number. ``headline_text`` is warm, jargon-free copy (API-R21).
    """
    diagnosis = await diagnose_coverage(svc, athlete_id)
    statuses = [inp.status for inp in diagnosis.inputs]
    present = sum(1 for s in statuses if s is InputStatus.PRESENT)
    stale = sum(1 for s in statuses if s is InputStatus.STALE)
    missing = sum(1 for s in statuses if s is InputStatus.MISSING)
    total = len(statuses)
    return DataHealthSummary(
        completeness_score=round(present / total, 4) if total else 0.0,
        present_count=present,
        stale_count=stale,
        missing_count=missing,
        headline_text=_headline(present, total),
        as_of=diagnosis.as_of,
    )


@router.get(
    "/coverage-matrix",
    response_model=CoverageMatrix,
    operation_id="getDataHealthCoverageMatrix",
    dependencies=[_Read],
)
async def coverage_matrix(
    svc: Service, athlete_id: AthleteId, session: DbSession
) -> CoverageMatrix:
    """Per-canonical-domain coverage across the connected sources (§8.3).

    Each domain row carries its typed status; ``coverage_pct`` is computed only for
    the training-load domain (the fraction of recent days with a resolved load — a
    real measurement) and is a typed ``null`` elsewhere (ANL-R3/R4). Source keys are
    visible here by design (the AUTH-R15 documented exception).
    """
    diagnosis = await diagnose_coverage(svc, athlete_id)
    sources = await _connected_sources(session, athlete_id)
    load_pct = await _load_coverage_pct(svc, athlete_id)
    cells = [
        CoverageCell(
            domain=inp.key,
            label=inp.label,
            status=inp.status.value,
            coverage_pct=load_pct if inp.key == "training_load" else None,
            sources=sources,
        )
        for inp in diagnosis.inputs
    ]
    return CoverageMatrix(domains=cells, as_of=diagnosis.as_of)


@router.get(
    "/issues",
    response_model=DataHealthIssueList,
    operation_id="listDataHealthIssues",
    dependencies=[_Read],
)
async def data_health_issues(
    svc: Service,
    athlete_id: AthleteId,
    limit: Annotated[int, Query(ge=1, json_schema_extra={"maximum": 200})] = 50,
) -> DataHealthIssueList:
    """Typed data-health issues from the deterministic coverage checks (§8.3).

    A MISSING canonical input is a ``warning``, a STALE one ``info`` — each with the
    typed analytics reason as its machine ``kind`` and athlete-native copy (API-R21).
    The set is naturally bounded by the closed check set; ``limit`` is still
    clamped/rejected per PAGE-R3.
    """
    bounded = clamp_limit(int(limit))
    diagnosis = await diagnose_coverage(svc, athlete_id)
    issues = _issues_of(diagnosis)
    return DataHealthIssueList(
        data=issues[:bounded],
        page=IssuePage(limit=bounded, next_cursor=None, has_more=len(issues) > bounded),
    )


def _issues_of(diagnosis: AgentDiagnosis) -> list[DataHealthIssue]:
    """Project non-present coverage inputs onto typed, athlete-native issues."""
    issues: list[DataHealthIssue] = []
    for inp in diagnosis.inputs:
        if inp.status is InputStatus.PRESENT:
            continue
        missing = inp.status is InputStatus.MISSING
        issues.append(
            DataHealthIssue(
                issue_id=f"coverage:{inp.key}",
                kind=inp.reason or ("missing_input" if missing else "stale_input"),
                severity="warning" if missing else "info",
                message_text=(
                    f"Your {inp.label.lower()} "
                    + ("isn't available yet." if missing else "is out of date.")
                ),
            )
        )
    return issues


def _headline(present: int, total: int) -> str:
    """The athlete-native one-liner for the summary (API-R21)."""
    if total and present == total:
        return "Your data looks complete — everything we need is here."
    if present:
        return "Most of your data is here — a few inputs could be richer."
    return "We don't have enough data yet — connect a source or upload a workout."


async def _connected_sources(session: DbSession, athlete_id: str) -> list[str]:
    """The owner's distinct connected source keys (the AUTH-R15 audit exception)."""
    try:
        owner = uuid.UUID(athlete_id)
    except (ValueError, AttributeError):
        return []
    rows = (
        await session.execute(
            select(SourceDescriptor.source_key)
            .join(
                Connection,
                Connection.source_descriptor_id == SourceDescriptor.source_descriptor_id,
            )
            .where(Connection.athlete_id == owner)
            .distinct()
        )
    ).scalars()
    return sorted(rows)


async def _load_coverage_pct(svc: AnalyticsService, athlete_id: str) -> float | None:
    """The measured fraction of recent days with a resolved load, as a percent."""
    today = _dt.datetime.now(_dt.UTC).date()
    frm = today - _dt.timedelta(days=DIAGNOSIS_WINDOW_DAYS - 1)
    loads = await svc.daily_load_series(athlete_id, frm, today)
    if not loads:
        return None
    covered = sum(1 for v in loads.values() if v is not None)
    return round(100.0 * covered / len(loads), 1)


__all__ = ["CoverageMatrix", "DataHealthIssue", "DataHealthSummary", "router"]
