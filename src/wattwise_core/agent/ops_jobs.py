"""Import-job + export-job rows on the agent-state store (API-R33 / API-R34, ARCH-R13).

OPERATIONAL API state per the amended ARCH-R13: export-job rows (and their single-use
signed-download-URL nonces) and the import-job rows backing the ``GET /v1/imports``
read surface live on the dedicated agent-state store — NEVER the canonical GBO store.
The canonical activities an import lands remain canonical; only the JOB bookkeeping is
operational state here.

Requirement IDs: API-R33 (import-job list/detail), API-R34 (export jobs + signed-URL
nonce single-use), ARCH-R13 (operational state on the agent-state store).
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.agent.state_store import AGENT_STATE_PREFIX, AgentStateBase
from wattwise_core.persistence.types import UtcDateTime, utcnow, uuid7


class ImportJobRecord(AgentStateBase):
    """One accepted upload job (API-R33): the ``GET /v1/imports`` read surface's row.

    Written when ``POST /v1/imports`` accepts an upload; ``status`` is the canonical
    import-job status (``queued|processing|done|failed``) and MUST reach a TERMINAL value
    reflecting the real ingest outcome — ``done`` on success, ``failed`` on a post-acceptance
    ingest error — never stranded at ``queued`` (API-R33a). Carries NO file bytes and no
    source/provider shape — the landed activity lives canonically; this row is only the
    job bookkeeping (operational state, ARCH-R13).
    """

    __tablename__ = AGENT_STATE_PREFIX + "import_job"

    # The processor-issued job handle (an activity/ingest-run id) — opaque on the wire,
    # stored verbatim so the GET surface returns exactly the id the 202 carried.
    import_job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    athlete_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status_text: Mapped[str] = mapped_column(String(256), nullable=False)
    received_at: Mapped[_dt.datetime] = mapped_column(UtcDateTime(), default=utcnow, nullable=False)


class ExportJobRecord(AgentStateBase):
    """One export job (API-R34): parameters, status, and the signed-URL nonce state.

    The job's ``nonce`` seeds the single-use signed download URL minted when the job is
    ready; ``nonce_used_at`` is the one-time-use marker (an atomic guarded UPDATE claims
    it). The artifact itself is generated deterministically from the stored parameters
    at download time — no athlete data is duplicated into this operational row.
    """

    __tablename__ = AGENT_STATE_PREFIX + "export_job"

    export_job_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    athlete_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    format: Mapped[str] = mapped_column(String(8), nullable=False)
    from_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    to_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ready")
    nonce: Mapped[str] = mapped_column(String(64), nullable=False)
    nonce_used_at: Mapped[_dt.datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(UtcDateTime(), default=utcnow, nullable=False)


__all__ = ["ExportJobRecord", "ImportJobRecord"]
