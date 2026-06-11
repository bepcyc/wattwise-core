"""Property-based contract for the time-relative FIT forge (issue #29 follow-through).

Fuzzes the forge↔production-decoder contract over its whole parameter space (start
instant, duration, sample rate, power level, device serial): EVERY forged file must
decode through the PRIMARY ``garmin-fit-sdk`` path — never the corrupt-file recovery
fallback — with timestamps, record counts, telemetry bounds, session aggregates, and
the STRONG fingerprint all consistent with the requested parameters. This is the seed
of the generator-driven activity testing the forge enables: any function that consumes
decoded activities can be driven off the same strategies.

Property IDs (local to the forge contract):

- **FORGE-T1** — round-trip: primary-SDK decode succeeds for all parameters; record
  count == ``duration // interval``; first/last record timestamps land exactly on
  ``start`` / ``start + (n-1)*interval``.
- **FORGE-T2** — telemetry bounds: every record's power lies in
  ``[base_power_w, base_power_w + 30]``; session ``avg_power`` equals the record mean
  (floor) and ``max_power >= avg_power``.
- **FORGE-T3** — determinism: identical parameters ⇒ byte-identical files.
- **FORGE-T4** — identity: the strong fingerprint is present and injective in
  ``(serial_number, start)`` — distinct serials or starts can never collide, so dedup
  can never merge two distinct forged rides.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# The repo root is not an installed package; tools/ modules are imported the same way
# the contract suite imports tools.client_gen (sys.path bootstrap, any import-mode).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.fit_forge import forge_ride  # noqa: E402  (after the sys.path bootstrap)

from wattwise_core.ingestion.adapters._decode_fit import _decode_with_sdk  # noqa: E402

pytestmark = pytest.mark.property

CI_SETTINGS = settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Aware, second-resolution UTC starts comfortably after the FIT epoch (1989-12-31).
starts = st.datetimes(
    min_value=_dt.datetime(2015, 1, 1),
    max_value=_dt.datetime(2035, 12, 31),
).map(lambda d: d.replace(microsecond=0, tzinfo=_dt.UTC))
durations = st.integers(min_value=60, max_value=1800)
intervals = st.integers(min_value=1, max_value=3)
base_powers = st.integers(min_value=80, max_value=1500)
serials = st.integers(min_value=1, max_value=2**31)


@CI_SETTINGS
@given(start=starts, duration_s=durations, interval_s=intervals, base_power_w=base_powers)
def test_forge_round_trips_through_primary_decoder_for_all_parameters(
    start: _dt.datetime, duration_s: int, interval_s: int, base_power_w: int
) -> None:
    """FORGE-T1 + FORGE-T2: strict-SDK decode, exact timestamps, bounded telemetry."""
    payload = forge_ride(
        start=start,
        duration_s=duration_s,
        sample_interval_s=interval_s,
        base_power_w=base_power_w,
    )
    asbo = _decode_with_sdk(payload)
    n = duration_s // interval_s
    assert len(asbo.records) == n
    assert asbo.records[0].timestamp == start
    assert asbo.records[-1].timestamp == start + _dt.timedelta(seconds=(n - 1) * interval_s)
    record_powers = [r.power_w for r in asbo.records]
    assert all(p is not None and base_power_w <= p <= base_power_w + 30 for p in record_powers)
    mean_floor = int(sum(p for p in record_powers if p is not None) // n)
    assert asbo.session["avg_power"] == mean_floor
    assert asbo.session["max_power"] >= asbo.session["avg_power"]
    assert asbo.session["start_time"] == start


@CI_SETTINGS
@given(start=starts, duration_s=durations, base_power_w=base_powers)
def test_forge_is_deterministic_in_its_parameters(
    start: _dt.datetime, duration_s: int, base_power_w: int
) -> None:
    """FORGE-T3: identical parameters produce byte-identical files (no hidden state)."""
    kwargs = {"start": start, "duration_s": duration_s, "base_power_w": base_power_w}
    assert forge_ride(**kwargs) == forge_ride(**kwargs)


@CI_SETTINGS
@given(start=starts, serial_a=serials, serial_b=serials)
def test_strong_fingerprints_are_injective_in_serial_and_start(
    start: _dt.datetime, serial_a: int, serial_b: int
) -> None:
    """FORGE-T4: fingerprints exist and collide only for identical (serial, start)."""
    fp_a = _decode_with_sdk(
        forge_ride(start=start, duration_s=60, serial_number=serial_a)
    ).strong_fingerprint
    fp_b = _decode_with_sdk(
        forge_ride(start=start, duration_s=60, serial_number=serial_b)
    ).strong_fingerprint
    assert fp_a is not None and fp_b is not None
    assert (fp_a == fp_b) == (serial_a == serial_b)
    shifted = _decode_with_sdk(
        forge_ride(start=start + _dt.timedelta(seconds=1), duration_s=60, serial_number=serial_a)
    ).strong_fingerprint
    assert shifted != fp_a
