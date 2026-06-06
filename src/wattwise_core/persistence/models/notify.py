"""Digest subscriptions and per-channel notification routes.

Owning requirements:

* ``digest_subscription`` — ONE standing digest schedule (GBO-R46/R46b/R46c/R47);
  ``hour_local`` is athlete-LOCAL (NEVER UTC); ``weekday`` token is identical on the
  wire; closing sets a terminal status.
* ``notification_route`` — ONE per-channel delivery binding (GBO-R49); UNIQUE
  ``(athlete_id, channel)``; ``web`` is always-on.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.domain.enums import (
    DeliveryChannel,
    DigestCadence,
    DigestStatus,
    Weekday,
)
from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.types import (
    enum_column,
    fk_uuid_column,
    json_column,
    pk_column,
    smallint_column,
)


class DigestSubscription(Base, TimestampMixin):
    """ONE standing digest schedule for the athlete (GBO-R46).

    Surrogate ``subscription_id`` PK is the canonical upsert key. The firing instant
    is derived from ``hour_local`` (+ ``weekday``/``cadence``) projected through the
    athlete reference timezone — NEVER stored as a UTC hour (GBO-R47).
    """

    __tablename__ = "digest_subscription"
    __table_args__ = (
        Index("ix_digest_subscription_athlete_status", "athlete_id", "status"),
    )

    subscription_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    cadence: Mapped[DigestCadence] = enum_column(DigestCadence, nullable=False)
    weekday: Mapped[Weekday | None] = enum_column(Weekday, nullable=True)
    hour_local: Mapped[int] = smallint_column(nullable=False)
    # ordered set of delivery channels (GBO-R46c) as a portable JSON list.
    channels: Mapped[list[str]] = json_column(nullable=False, default=list)
    status: Mapped[DigestStatus] = enum_column(DigestStatus, nullable=False)


class NotificationRoute(Base, TimestampMixin):
    """ONE per-channel delivery binding (GBO-R49).

    UNIQUE ``(athlete_id, channel)``. ``web`` is always-on (no ``address_ref``,
    verified by construction); ``email``/``telegram`` deliver only when
    ``verified AND enabled``.
    """

    __tablename__ = "notification_route"
    __table_args__ = (
        UniqueConstraint("athlete_id", "channel", name="uq_notification_route_athlete_channel"),
    )

    route_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    channel: Mapped[DeliveryChannel] = enum_column(DeliveryChannel, nullable=False)
    address_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


__all__ = ["DigestSubscription", "NotificationRoute"]
