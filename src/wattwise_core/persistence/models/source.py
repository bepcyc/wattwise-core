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

Tier-2 store (GBO-R8c): NEVER read by consumers (LIN-R4).
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Boolean, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.domain.enums import (
    AuthArchetype,
    ConnectionStatus,
    GboType,
    SourceKind,
)
from wattwise_core.persistence.base import Base, TimestampMixin
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
        UniqueConstraint(
            "athlete_id",
            "source_descriptor_id",
            "source_native_id",
            "gbo_type",
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


__all__ = ["Connection", "SourceCandidate", "SourceDescriptor"]
