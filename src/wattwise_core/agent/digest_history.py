"""Stored weekly-review (digest) history on the agent-state store (API-R14 / ARCH-R13).

``GET /v1/agent/digest/list`` is the spec's *paginated history of weekly reviews* (doc 60
API-R14) — not the standing subscription CRUD. The history is OPERATIONAL deliverable
state, so its rows live on the dedicated agent-state store (:class:`AgentStateBase`,
ARCH-R13 — never the canonical GBO store): one row per ``(athlete, week_end)``, upserted
when a grounded weekly review is generated, read newest-first behind a signed keyset
cursor (PAGE-R1/R5).

Only a COMPLETED (grounded) review is recorded — a ``degraded`` abstention is a visible
non-answer (OUTCOME-R3), not a review the athlete should page back through. The stored
body is the deliverable's own fields (status/thread/week/body/observations/citations/
caveat), serialized to portable JSON and re-hydrated VERBATIM on read — this layer never
recomputes or invents a value (GROUND-R7).

Requirement IDs: API-R14 (paginated weekly-review history), ARCH-R13 (operational state on
the agent-state store), PAGE-R1/R5 (keyset behind a signed cursor), GROUND-R7.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import Digest
from wattwise_core.agent.state_store import AGENT_STATE_PREFIX, AgentStateBase

if TYPE_CHECKING:  # imported lazily: state_db imports THIS module to register the table
    from wattwise_core.agent.state_db import AgentStateDatabase
from wattwise_core.agent.voice import Citation, Observation
from wattwise_core.persistence.types import UtcDateTime, utcnow, uuid7


class AgentDigestRecord(AgentStateBase):
    """One stored weekly review (digest) for ``(athlete, week_end)`` (API-R14 history).

    Upserted when a grounded weekly review is generated so the history surface replays
    the SAME deliverable the athlete saw (no recomputation). Operational agent-state —
    never canonical master data (ARCH-R13); joins the per-athlete erasure target set.
    """

    __tablename__ = AGENT_STATE_PREFIX + "digest_record"
    __table_args__ = (
        UniqueConstraint("athlete_id", "week_end", name="uq_agent_digest_athlete_week"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    athlete_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    week_end: Mapped[str] = mapped_column(String(10), nullable=False)
    body: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[_dt.datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )


def _serialize(digest: Digest) -> dict[str, Any]:
    """Project a :class:`Digest` deliverable into the portable JSON row body (verbatim)."""
    return {
        "status": digest.status.value,
        "thread_id": digest.thread_id,
        "week_end": digest.week_end,
        "digest_html": digest.digest_html,
        "digest_text": digest.digest_text,
        "observations": [
            {"observation_id": o.observation_id, "text": o.text} for o in digest.observations
        ],
        "citations": [
            {"record_id": c.record_id, "metric": c.metric, "value": c.value, "as_of": c.as_of}
            for c in digest.citations
        ],
        "suggested_followups": list(digest.suggested_followups),
        "coverage_caveat": dict(digest.coverage_caveat)
        if digest.coverage_caveat is not None
        else None,
    }


def _deserialize(body: dict[str, Any]) -> Digest:
    """Re-hydrate the stored JSON body into the :class:`Digest` deliverable (verbatim)."""
    return Digest(
        status=RunStatus(body["status"]),
        thread_id=str(body["thread_id"]),
        week_end=str(body["week_end"]),
        digest_html=str(body["digest_html"]),
        digest_text=str(body["digest_text"]),
        observations=tuple(
            Observation(observation_id=str(o["observation_id"]), text=str(o["text"]))
            for o in body.get("observations", ())
        ),
        citations=tuple(
            Citation(
                record_id=str(c["record_id"]),
                metric=c.get("metric"),
                value=c.get("value"),
                as_of=c.get("as_of"),
            )
            for c in body.get("citations", ())
        ),
        suggested_followups=tuple(str(s) for s in body.get("suggested_followups", ())),
        coverage_caveat=body.get("coverage_caveat"),
    )


async def record_digest(
    state_db: AgentStateDatabase, *, athlete_id: str, digest: Digest
) -> None:
    """Upsert the grounded weekly review into the history (one row per athlete+week).

    A non-``completed`` digest is NOT recorded (an abstention is not a review). The
    upsert keys on ``(athlete_id, week_end)`` so a re-generated week replaces its row
    rather than duplicating history.
    """
    if digest.status is not RunStatus.COMPLETED:
        return
    owner = uuid.UUID(athlete_id)
    async with state_db.session() as session:
        stmt = select(AgentDigestRecord).where(
            AgentDigestRecord.athlete_id == owner,
            AgentDigestRecord.week_end == digest.week_end,
        )
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            row = AgentDigestRecord(
                athlete_id=owner, week_end=digest.week_end, body=_serialize(digest)
            )
            session.add(row)
        else:
            row.body = _serialize(digest)
        await session.commit()


async def digest_history(
    state_db: AgentStateDatabase,
    *,
    athlete_id: str,
    limit: int,
    before_week_end: str | None = None,
) -> list[Digest]:
    """The athlete's stored weekly reviews, newest week first, keyset-paged (API-R14).

    ``before_week_end`` is the exclusive keyset bound (the page resumes strictly BEFORE
    that ISO week-end), carried in the router's signed opaque cursor (PAGE-R5). Scoped
    STRICTLY to the server-derived owner — never another athlete's rows.
    """
    owner = uuid.UUID(athlete_id)
    stmt = (
        select(AgentDigestRecord)
        .where(AgentDigestRecord.athlete_id == owner)
        .order_by(AgentDigestRecord.week_end.desc())
        .limit(limit)
    )
    if before_week_end is not None:
        stmt = stmt.where(AgentDigestRecord.week_end < before_week_end)
    async with state_db.session() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [_deserialize(dict(r.body)) for r in rows]


__all__ = [
    "AgentDigestRecord",
    "digest_history",
    "record_digest",
]
