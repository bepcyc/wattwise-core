"""Round-trip contract for the time-relative FIT forge (issue #29).

The forge (:mod:`tools.fit_forge`) hand-encodes the FIT binary protocol (the official
Python SDK is decode-only), so this contract pins its output to the PRODUCTION decode
path: every forged file must decode through the PRIMARY ``garmin-fit-sdk`` decoder —
never the corrupt-file recovery fallback — and yield exactly the telemetry the E2E
smoke relies on (recent timestamps, power/HR/cadence, a cycling session summary, and
a DISTINCT strong fingerprint per ride so dedup can never merge the batch). If the
encoder drifts from what ingest accepts, this fails before the smoke ever runs.
All inputs are forged in-test bytes — no fixtures, no network (TST-R1).
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import pytest

# The repo root is not an installed package; tools/ modules are imported the same way
# the contract suite imports tools.client_gen (sys.path bootstrap, any import-mode).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.fit_forge import (  # noqa: E402  (after the sys.path bootstrap)
    DEFAULT_DAYS_BACK,
    forge_recent_batch,
    forge_ride,
)

from wattwise_core.ingestion.adapters._decode_fit import (  # noqa: E402
    _decode_with_sdk,
    decode_fit,
)

pytestmark = pytest.mark.unit

_START = _dt.datetime(2026, 6, 1, 10, 0, tzinfo=_dt.UTC)


def test_forged_ride_decodes_via_primary_sdk_decoder() -> None:
    """The forged bytes satisfy the STRICT official decoder, not just the recovery path."""
    payload = forge_ride(start=_START)
    asbo = _decode_with_sdk(payload)
    assert len(asbo.records) == 1200  # 20 min at 1 Hz, like a real head unit
    first = asbo.records[0]
    assert first.timestamp == _START
    assert first.power_w and first.power_w > 0
    assert first.hr_bpm and first.hr_bpm > 0
    assert first.cadence_rpm and first.cadence_rpm > 0
    assert first.speed_mps and first.speed_mps > 0
    assert first.latlng is not None


def test_forged_session_summary_is_cycling_with_load_bearing_fields() -> None:
    """Session decodes as a cycling ride with the aggregates analytics reads."""
    asbo = decode_fit(forge_ride(start=_START))
    assert asbo.session["sport"] == "cycling"
    assert asbo.session["start_time"] == _START
    assert asbo.session["total_timer_time"] == pytest.approx(1200.0)
    assert asbo.session["avg_power"] > 0
    assert asbo.session["max_power"] >= asbo.session["avg_power"]
    assert len(asbo.laps) == 1
    assert asbo.laps[0].duration_s == pytest.approx(1200.0)


def test_forged_ride_is_deterministic_for_a_fixed_start() -> None:
    """Same start instant -> byte-identical file (reproducible failures, no randomness)."""
    assert forge_ride(start=_START) == forge_ride(start=_START)


def test_recent_batch_lands_inside_the_two_week_recency_window() -> None:
    """Every forged start sits in (now - 14d, now] — the window a recency ask reads."""
    now = _dt.datetime(2026, 6, 11, 8, 30, tzinfo=_dt.UTC)
    batch = forge_recent_batch(now=now)
    assert len(batch) == len(DEFAULT_DAYS_BACK)
    for ride, days in zip(batch, DEFAULT_DAYS_BACK, strict=True):
        age = now - ride.start
        assert _dt.timedelta() < age < _dt.timedelta(days=14)
        assert ride.start.date() == (now - _dt.timedelta(days=days)).date()
        decoded = decode_fit(ride.payload)
        assert decoded.records[0].timestamp == ride.start


def test_recent_batch_fingerprints_are_strong_and_distinct() -> None:
    """Each ride carries its own MAP-R10 STRONG fingerprint — dedup can never merge two."""
    batch = forge_recent_batch(now=_dt.datetime(2026, 6, 11, 8, 30, tzinfo=_dt.UTC))
    fingerprints = [decode_fit(r.payload).strong_fingerprint for r in batch]
    assert all(fp is not None for fp in fingerprints)
    assert len(set(fingerprints)) == len(batch)


def test_forge_rejects_naive_start() -> None:
    """A naive datetime fails closed — forged timestamps must be unambiguous UTC."""
    with pytest.raises(ValueError, match="timezone-aware"):
        forge_ride(start=_dt.datetime(2026, 6, 1, 10, 0))
