"""Unit tests for the connection sync-health predicate (the issue #12 MNAR disambiguator).

Pins the pure :func:`wattwise_core.agent.engine_readiness.connection_is_suspect` contract: a gap in
observed training data only implies MISSING data (rather than a legitimate taper/rest) when a
connector that SHOULD be delivering is broken or silently stalled. A broken/reauth connector is
suspect outright; a "connected" source that never synced or whose last sync is itself stale is
silently failing; a deliberately disconnected source is never suspect.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from wattwise_core.agent.engine_readiness import connection_is_suspect
from wattwise_core.domain.enums import ConnectionStatus

pytestmark = pytest.mark.unit

_REF = _dt.date(2026, 6, 10)
_STALE_AFTER = 3


def _suspect(status: ConnectionStatus, last_synced: _dt.datetime | None) -> bool:
    return connection_is_suspect(
        status, last_synced, reference_date=_REF, sync_stale_after_days=_STALE_AFTER
    )


def _synced_days_ago(days: int) -> _dt.datetime:
    return _dt.datetime(2026, 6, 10, 12, 0, tzinfo=_dt.UTC) - _dt.timedelta(days=days)


def test_reauth_required_is_suspect() -> None:
    """A connector needing re-auth cannot deliver -> any data gap behind it is suspect."""
    assert _suspect(ConnectionStatus.REAUTH_REQUIRED, _synced_days_ago(1)) is True


def test_error_is_suspect() -> None:
    """An errored connector is overtly broken -> suspect."""
    assert _suspect(ConnectionStatus.ERROR, _synced_days_ago(0)) is True


def test_connected_and_recently_synced_is_not_suspect() -> None:
    """A healthy, recently-synced connector proves a data gap is real rest, not missing data."""
    assert _suspect(ConnectionStatus.CONNECTED, _synced_days_ago(1)) is False


def test_connected_but_silently_stalled_is_suspect() -> None:
    """A "connected" source whose last sync is itself stale is silently failing (bad credential)."""
    assert _suspect(ConnectionStatus.CONNECTED, _synced_days_ago(10)) is True


def test_connected_but_never_synced_is_suspect() -> None:
    """Connected with no successful sync yet is treated as not-yet-delivering -> suspect."""
    assert _suspect(ConnectionStatus.CONNECTED, None) is True


def test_disconnected_is_never_suspect() -> None:
    """A deliberate disconnect expects no data; its gap must not manufacture a sync alarm."""
    assert _suspect(ConnectionStatus.DISCONNECTED, None) is False
    assert _suspect(ConnectionStatus.DISCONNECTED, _synced_days_ago(99)) is False
