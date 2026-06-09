"""Source provenance: descriptors, connections, and the per-source candidate store.

Owning requirements:

* ``source_descriptor`` — global read-only source registration config
  (LIN-R1 / TEN-R4); the built-in ``file_import`` descriptor is seeded by migration
  (LIN-R1.1).
* ``connection`` — ONE athlete -> source authorization (GBO-R43/R44/R45/R48); UNIQUE
  ``(athlete_id, source_descriptor_id)``; opaque ``credential_ref``.
* ``source_candidate`` — the per-source lineage envelope (LIN-R2); candidate key
  ``(athlete_id, source_descriptor_id, source_native_id, gbo_type)`` — the ONLY key
  in which source identity appears (UPS-R1). Carries resolved-*-id back-pointers,
  ``is_superseded``, ``content_hash``, observed/fetched clocks, adapter/mapping
  versions, trust profile, confidence, ingest run id, untrusted-content flag.

* ``ingestion_watermark`` — the idempotent incremental cursor per
  ``(athlete_id, source_descriptor_id, gbo_type, stream)`` (SYN-R2); advanced
  transactionally with the batch it represents (SYN-R3) and honored by discover
  (ADP-R6).
* ``ingestion_gap`` — the first-class typed gap recording a partial failure
  (ING-GAP-R1..R6): typed ``reason`` taxonomy, open/closed state, range-precision.

Tier-2 store (GBO-R8c): NEVER read by consumers (LIN-R4). The watermark and gap entities
are source-derived canonical data, written ONLY by the Ingestion/Sync service (ARCH-R3
canonical-write partition), never by the master-data-write or agent-state-write roles.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Boolean, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, validates

from wattwise_core.domain.enums import (
    AuthArchetype,
    ConnectionStatus,
    GapReason,
    GapState,
    GboType,
    Severity,
    SourceKind,
)
from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.models.athlete_preference import ensure_ranked_tier
from wattwise_core.persistence.types import (
    enum_column,
    fk_uuid_column,
    json_column,
    numeric_column,
    pk_column,
    timestamptz_column,
)


class SourceDescriptor(Base, TimestampMixin):
    """Global read-only source registration (LIN-R1 / TEN-R4).

    Registration is *data*, not code (GBO-R4). The built-in OSS file importer
    registers exactly one descriptor (``source_key="file_import"``, LIN-R1.1),
    seeded by the initial migration.
    """

    __tablename__ = "source_descriptor"
    __table_args__ = (
        UniqueConstraint("source_key", name="uq_source_descriptor_source_key"),
    )

    source_descriptor_id: Mapped[uuid.UUID] = pk_column()
    source_key: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[SourceKind] = enum_column(SourceKind, nullable=False)
    # default trust profile: per-channel/metric base trust_tier ordering +
    # default_fidelity (LIN-R1). Stored as portable JSON config.
    trust_profile: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)
    default_fidelity: Mapped[str | None] = mapped_column(String(64), nullable=True)

    @validates("default_fidelity")
    def _validate_default_fidelity(self, _key: str, value: object) -> str | None:
        """Reject a non-ranked-tier ``default_fidelity`` (bare String, NO CHECK) (CONF-R2).

        ``default_fidelity`` is the whole-source declared base tier; only the 5 ranked
        tiers are valid. ``None`` is allowed (no declared base). A non-tier token would
        otherwise flow into the effective tier and abort ingest, so it is rejected here.
        """
        if value is None:
            return None
        return ensure_ranked_tier(value, field="default_fidelity").value

    @validates("trust_profile")
    def _validate_trust_profile(self, _key: str, value: object) -> dict[str, object]:
        """Reject any non-ranked-tier token among a ``trust_profile``'s tier values.

        ``trust_profile`` maps channel (or ``"*"``) -> tier token; every value must be one
        of the 5 ranked tiers (CONF-R2). A ``None``/empty profile is the seeded default.
        """
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"trust_profile must be a dict, got {type(value).__name__}")
        for channel, token in value.items():
            ensure_ranked_tier(token, field=f"trust_profile[{channel!r}]")
        return dict(value)


class Connection(Base, TimestampMixin):
    """ONE authorization athlete -> source_descriptor (GBO-R43/R45/R48).

    Surrogate PK ``connection_id`` is the canonical upsert key; UNIQUE
    ``(athlete_id, source_descriptor_id)`` enforces at most one auth per source.
    Stores ONLY an opaque ``credential_ref`` (secret owned by doc 70). Not
    source-derived (no candidate key). Ad-hoc file upload creates NO row.
    """

    __tablename__ = "connection"
    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "source_descriptor_id",
            name="uq_connection_athlete_source",
        ),
        Index("ix_connection_athlete_source", "athlete_id", "source_descriptor_id"),
    )

    connection_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    source_descriptor_id: Mapped[uuid.UUID] = fk_uuid_column(
        "source_descriptor.source_descriptor_id", nullable=False
    )
    status: Mapped[ConnectionStatus] = enum_column(ConnectionStatus, nullable=False)
    credential_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    scopes: Mapped[list[str]] = json_column(nullable=False, default=list)
    connected_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    last_synced_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    # set at creation, immutable; consumers branch on archetype not source name.
    auth_archetype: Mapped[AuthArchetype] = enum_column(AuthArchetype, nullable=False)


class SourceCandidate(Base, TimestampMixin):
    """Per-source mapped observation + the SINGLE lineage envelope (LIN-R2).

    Candidate key ``(athlete_id, source_descriptor_id, source_native_id, gbo_type)``
    — the ONLY key carrying source identity (UPS-R1). Carries resolved-*-id
    back-pointers into the canonical store (nullable, since a candidate may be
    quarantined/unresolved), supersession + content-hash for idempotent re-ingest
    (UPS-R3/R5), and the trust/confidence inputs to ``resolve_field`` (CONF-R2).
    """

    __tablename__ = "source_candidate"
    __table_args__ = (
        # UNIQUE on the candidate key PLUS content_hash: re-ingesting byte-identical
        # content is an idempotent no-op (UPS-R3), while a changed restatement (same
        # candidate key, new content_hash) lands as a NEW retained version that
        # supersedes the prior one (UPS-R5/PRV-R2) — versioning the bare candidate key
        # could not (only one current version is non-superseded at a time).
        UniqueConstraint(
            "athlete_id",
            "source_descriptor_id",
            "source_native_id",
            "gbo_type",
            "content_hash",
            name="uq_source_candidate_candidate_key",
        ),
        Index(
            "ix_source_candidate_candidate_key",
            "athlete_id",
            "source_descriptor_id",
            "source_native_id",
            "gbo_type",
        ),
        Index("ix_source_candidate_resolved_activity_id", "resolved_activity_id"),
        Index("ix_source_candidate_resolved_daily_wellness_id", "resolved_daily_wellness_id"),
        Index(
            "ix_source_candidate_resolved_stream_set_id", "resolved_stream_set_id"
        ),
        Index(
            "ix_source_candidate_resolved_wellness_stream_set_id",
            "resolved_wellness_stream_set_id",
        ),
        Index("ix_source_candidate_resolved_signature_id", "resolved_signature_id"),
        Index("ix_source_candidate_content_hash", "content_hash"),
    )

    source_candidate_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    source_descriptor_id: Mapped[uuid.UUID] = fk_uuid_column(
        "source_descriptor.source_descriptor_id", nullable=False
    )
    # NULL for connectionless file-import (LIN-R1.1).
    connection_id: Mapped[uuid.UUID | None] = fk_uuid_column(
        "connection.connection_id", nullable=True
    )
    source_native_id: Mapped[str] = mapped_column(String(256), nullable=False)
    gbo_type: Mapped[GboType] = enum_column(GboType, nullable=False)
    observed_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    fetched_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    adapter_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mapping_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trust_profile: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)
    # The adapter's mapped canonical payload (tier-2), retained durably so the
    # conflict resolver can re-resolve a field without re-fetching the source
    # (UPS-R4 / CONF-R6). Canonical fields only — no source-named keys (MAP-R2).
    payload: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)
    confidence: Mapped[float | None] = numeric_column(nullable=True)
    ingest_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    untrusted_content: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_superseded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # nullable canonical back-pointers (no hard FK: the candidate may be unresolved
    # or quarantined, and which canonical type it resolves to depends on gbo_type).
    resolved_activity_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    resolved_activity_lap_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    resolved_activity_file_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    resolved_daily_wellness_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    resolved_stream_set_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    resolved_wellness_stream_set_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    resolved_stream_channel_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    resolved_signature_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)


class IngestionWatermark(Base, TimestampMixin):
    """The idempotent incremental cursor per ingest scope (SYN-R2).

    ONE row per ``(athlete_id, source_descriptor_id, gbo_type, stream)`` — the watermark
    scope SYN-R2 mandates; ``stream`` is the optional sub-key (NULL for record-level GBO
    types, a channel name when a per-stream cursor is needed). It captures BOTH a
    **high-water timestamp/cursor** (``high_water_at`` / ``cursor``) AND a **content
    hint** (``content_hint`` — e.g. the last ``observed_at`` or ``content_hash``) so a
    changed-but-not-new record is re-fetched rather than skipped (SYN-R2). It is the
    source-derived canonical cursor written ONLY by the Ingestion/Sync service in the
    SAME transaction as the batch upsert it represents (SYN-R3 / ING-UPS-R2), so a crash
    mid-run never advances past un-committed data and a re-run resumes from the committed
    cursor (ING-R6). Discover honors it for incremental mode (ADP-R6).

    NULL ``stream`` cannot participate in a portable UNIQUE constraint identically across
    backends, so the empty sentinel ``""`` denotes "no stream sub-key"; the unique key is
    over the four non-null columns.
    """

    __tablename__ = "ingestion_watermark"
    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "source_descriptor_id",
            "gbo_type",
            "stream",
            name="uq_ingestion_watermark_scope",
        ),
        Index(
            "ix_ingestion_watermark_scope",
            "athlete_id",
            "source_descriptor_id",
            "gbo_type",
        ),
    )

    ingestion_watermark_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    source_descriptor_id: Mapped[uuid.UUID] = fk_uuid_column(
        "source_descriptor.source_descriptor_id", nullable=False
    )
    gbo_type: Mapped[GboType] = enum_column(GboType, nullable=False)
    # Optional per-stream sub-key (SYN-R2 ``[, stream]``); ``""`` = no sub-key (so the
    # composite UNIQUE is identical across SQLite/PostgreSQL/MariaDB — no NULL semantics).
    stream: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    # High-water cursor: the most-recent ingested instant (timestamp half of SYN-R2).
    high_water_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    # Opaque source-supplied cursor token (e.g. a discovery ``next_cursor`` watermark).
    cursor: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # Content hint (last observed_at / content_hash) so changed-but-not-new records are
    # re-fetched and not silently skipped by discover (SYN-R2 / ADP-R6).
    content_hint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ingest_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)


class IngestionGap(Base, TimestampMixin):
    """A first-class typed gap: what acquisition/mapping/conflict could not complete.

    The structured, queryable representation of a partial failure (ING-GAP-R1); never
    swallowed nor logged-only. It identifies ``athlete_id``, ``source_descriptor_id``
    (NULL for a source-agnostic gap), the affected ``gbo_type``, the time/record range it
    covers, a typed ``reason``, ``severity``, the ``ingest_run_id``, first/last-seen
    timestamps, and whether it is ``transient`` (auto-retryable) or ``terminal`` (needs
    user/operator action) (ING-GAP-R2). It carries open/closed ``state`` plus a
    ``closed_at`` closure timestamp so a transient gap is self-healing — a later
    successful sync covering the same range closes it (ING-GAP-R4). It is range-precise:
    it covers exactly the un-ingested range, leaving successfully ingested records in the
    same run committed (ING-GAP-R5 / ING-UPS-R3). A genuine source-side absence is NOT a
    gap (ING-GAP-R6).

    The range is carried as an OPEN pair so EITHER a time range (``range_start_at`` /
    ``range_end_at``) OR a discovery record range (``range_start_token`` /
    ``range_end_token``) can be expressed without a second divergent schema.
    """

    __tablename__ = "ingestion_gap"
    __table_args__ = (
        Index(
            "ix_ingestion_gap_athlete_source_gbo",
            "athlete_id",
            "source_descriptor_id",
            "gbo_type",
        ),
        Index("ix_ingestion_gap_state", "state"),
        Index("ix_ingestion_gap_ingest_run_id", "ingest_run_id"),
    )

    ingestion_gap_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    # NULL for a source-agnostic gap (ING-GAP-R2 explicitly allows it).
    source_descriptor_id: Mapped[uuid.UUID | None] = fk_uuid_column(
        "source_descriptor.source_descriptor_id", nullable=True
    )
    gbo_type: Mapped[GboType] = enum_column(GboType, nullable=False)
    reason: Mapped[GapReason] = enum_column(GapReason, nullable=False)
    severity: Mapped[Severity] = enum_column(Severity, nullable=False)
    state: Mapped[GapState] = enum_column(GapState, nullable=False, default=GapState.OPEN)
    # transient = auto-retryable / self-healing; terminal = needs user/operator action.
    transient: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # The covered range — a time range and/or a discovery record-token range (ING-GAP-R5).
    range_start_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    range_end_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    range_start_token: Mapped[str | None] = mapped_column(String(256), nullable=True)
    range_end_token: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ingest_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    first_seen_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    last_seen_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    # The closure timestamp ING-GAP-R4 mandates a transient gap records when it heals.
    closed_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)


__all__ = [
    "Connection",
    "IngestionGap",
    "IngestionWatermark",
    "SourceCandidate",
    "SourceDescriptor",
]
