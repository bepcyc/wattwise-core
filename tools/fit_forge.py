"""Time-relative FIT activity forge for the E2E smoke (issue #29).

A static FIT fixture frozen in 2024 can never ground a "recent training" question:
the truthful-agent core — correctly — refuses on recency/sufficiency grounds, so the
smoke could only ever exercise the refusal path. This module forges small, valid,
DETERMINISTIC FIT activity files whose timestamps are computed relative to "now" at
smoke time, so the grounded-answer path becomes provable on every run.

Design constraints:

- **Stdlib-only encoder.** The official ``garmin-fit-sdk`` Python package is
  decode-only (Garmin has stated no encoder is planned), so the FIT binary protocol
  (header + definition/data messages + CRC-16) is encoded here by hand. The encoding
  is pinned by a round-trip contract test through the PRODUCTION decoder
  (:func:`wattwise_core.ingestion.adapters._decode_fit.decode_fit`) — the forge can
  never silently drift from what ingest actually accepts.
- **Deterministic content, relative time.** Only the timestamps move with the clock;
  every sample (power/HR/cadence/speed) is a fixed function of the sample index, so a
  failing smoke reproduces byte-identically for the same start instant (no seeds, no
  randomness — flaky-fixture guidance says generated data must be deterministic).
- **Distinct device identities.** Each forged ride carries its own ``file_id``
  ``serial_number`` + ``time_created``, i.e. a distinct MAP-R10 STRONG fingerprint,
  so the dedup resolver can never merge two forged rides into one.

Usage (module)::

    from tools.fit_forge import forge_recent_batch
    for ride in forge_recent_batch():
        upload(ride.filename, ride.payload)

Usage (CLI, writes files for manual poking)::

    uv run python -m tools.fit_forge --out-dir /tmp/forged
"""

from __future__ import annotations

import argparse
import datetime as _dt
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

#: FIT timestamps count seconds since the FIT epoch, 1989-12-31T00:00:00Z.
_FIT_EPOCH_UNIX = 631065600

#: Garmin semicircle unit: degrees * 2**31 / 180.
_DEG_TO_SEMICIRCLE = (2**31) / 180.0

#: The CRC-16 nibble table from the FIT SDK (the file checksum algorithm).
_CRC_TABLE = (
    0x0000,
    0xCC01,
    0xD801,
    0x1400,
    0xF001,
    0x3C00,
    0x2800,
    0xE401,
    0xA001,
    0x6C00,
    0x7800,
    0xB401,
    0x5000,
    0x9C01,
    0x8801,
    0x4400,
)

# FIT base-type ids + their little-endian struct formats (only the ones the forge emits).
_ENUM = 0x00
_SINT8 = 0x01
_UINT8 = 0x02
_UINT16 = 0x84
_SINT32 = 0x85
_UINT32 = 0x86
_UINT32Z = 0x8C

_FMT_BY_BASE_TYPE = {
    _ENUM: "<B",
    _SINT8: "<b",
    _UINT8: "<B",
    _UINT16: "<H",
    _SINT32: "<i",
    _UINT32: "<I",
    _UINT32Z: "<I",
}

#: Default "recent batch" placement: days back from now for each forged ride. The most
#: recent ride sits INSIDE the caveat-free freshness zone (``readiness_fresh_staleness_days
#: = 2``) and the batch spans the two-week window every "recent training" question reads,
#: while staying clear of the ``readiness_max_staleness_days = 14`` hard floor.
DEFAULT_DAYS_BACK: tuple[int, ...] = (1, 4, 7, 11)


def _crc16(data: bytes) -> int:
    """The FIT CRC-16 over ``data`` (nibble-table algorithm from the FIT SDK)."""
    crc = 0
    for byte in data:
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[byte & 0xF]
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[(byte >> 4) & 0xF]
    return crc


def _fit_ts(instant: _dt.datetime) -> int:
    """``instant`` (aware UTC) as a FIT uint32 timestamp."""
    return int(instant.timestamp()) - _FIT_EPOCH_UNIX


@dataclass(frozen=True)
class _MesgType:
    """One FIT message layout: local id, global number, and (field_num, base_type) defs."""

    local_id: int
    global_num: int
    fields: tuple[tuple[int, int], ...]

    def definition(self) -> bytes:
        """The definition record announcing this layout (little-endian architecture)."""
        out = bytearray()
        out += struct.pack("<BBBHB", 0x40 | self.local_id, 0, 0, self.global_num, len(self.fields))
        for field_num, base_type in self.fields:
            size = struct.calcsize(_FMT_BY_BASE_TYPE[base_type])
            out += struct.pack("<BBB", field_num, size, base_type)
        return bytes(out)

    def data(self, *values: int) -> bytes:
        """One data record carrying ``values`` in definition order."""
        if len(values) != len(self.fields):
            raise ValueError(f"expected {len(self.fields)} values, got {len(values)}")
        out = bytearray(struct.pack("<B", self.local_id))
        for (_, base_type), value in zip(self.fields, values, strict=True):
            out += struct.pack(_FMT_BY_BASE_TYPE[base_type], value)
        return bytes(out)


# Message layouts (field numbers per the Garmin FIT global profile).
_FILE_ID = _MesgType(
    local_id=0,
    global_num=0,
    fields=(
        (0, _ENUM),  # type = 4 (activity)
        (1, _UINT16),  # manufacturer = 1 (garmin)
        (2, _UINT16),  # product
        (3, _UINT32Z),  # serial_number
        (4, _UINT32),  # time_created
    ),
)
_RECORD = _MesgType(
    local_id=1,
    global_num=20,
    fields=(
        (253, _UINT32),  # timestamp
        (0, _SINT32),  # position_lat (semicircles)
        (1, _SINT32),  # position_long (semicircles)
        (2, _UINT16),  # altitude ((m + 500) * 5)
        (3, _UINT8),  # heart_rate (bpm)
        (4, _UINT8),  # cadence (rpm)
        (5, _UINT32),  # distance (cm)
        (6, _UINT16),  # speed (mm/s)
        (7, _UINT16),  # power (W)
        (13, _SINT8),  # temperature (degC)
    ),
)
_LAP = _MesgType(
    local_id=2,
    global_num=19,
    fields=(
        (253, _UINT32),  # timestamp (lap end)
        (0, _ENUM),  # event = 9 (lap)
        (1, _ENUM),  # event_type = 1 (stop)
        (2, _UINT32),  # start_time
        (7, _UINT32),  # total_elapsed_time (ms)
        (8, _UINT32),  # total_timer_time (ms)
        (9, _UINT32),  # total_distance (cm)
        (15, _UINT8),  # avg_heart_rate
        (16, _UINT8),  # max_heart_rate
        (17, _UINT8),  # avg_cadence
        (19, _UINT16),  # avg_power
        (20, _UINT16),  # max_power
    ),
)
_SESSION = _MesgType(
    local_id=3,
    global_num=18,
    fields=(
        (253, _UINT32),  # timestamp (session end)
        (0, _ENUM),  # event = 8 (session)
        (1, _ENUM),  # event_type = 1 (stop)
        (2, _UINT32),  # start_time
        (5, _ENUM),  # sport = 2 (cycling)
        (6, _ENUM),  # sub_sport = 0 (generic)
        (7, _UINT32),  # total_elapsed_time (ms)
        (8, _UINT32),  # total_timer_time (ms)
        (9, _UINT32),  # total_distance (cm)
        (16, _UINT8),  # avg_heart_rate
        (17, _UINT8),  # max_heart_rate
        (18, _UINT8),  # avg_cadence
        (20, _UINT16),  # avg_power
        (21, _UINT16),  # max_power
        (25, _UINT16),  # first_lap_index
        (26, _UINT16),  # num_laps
    ),
)
_ACTIVITY = _MesgType(
    local_id=4,
    global_num=34,
    fields=(
        (253, _UINT32),  # timestamp
        (0, _UINT32),  # total_timer_time (ms)
        (1, _UINT16),  # num_sessions
        (2, _ENUM),  # type = 0 (manual)
        (3, _ENUM),  # event = 26 (activity)
        (4, _ENUM),  # event_type = 1 (stop)
    ),
)


def _sample(index: int) -> tuple[int, int, int, int]:
    """Deterministic (power_w, hr_bpm, cadence_rpm, speed_mms) for sample ``index``.

    A fixed tempo ride with a gentle 60-sample undulation — enough variation to look
    like telemetry, zero randomness so the forged bytes are reproducible.
    """
    wave = index % 60
    bump = wave if wave < 30 else 60 - wave  # triangle 0..30..0
    power = 185 + bump  # 185..215 W
    hr = 135 + bump // 3  # 135..145 bpm
    cadence = 88 + bump // 10  # 88..91 rpm
    speed = 8200 + bump * 10  # 8.2..8.5 m/s in mm/s
    return power, hr, cadence, speed


def forge_ride(
    *,
    start: _dt.datetime,
    duration_s: int = 1200,
    sample_interval_s: int = 1,
    serial_number: int = 77_000_001,
) -> bytes:
    """Forge one valid FIT cycling activity starting at ``start`` (aware UTC).

    Returns the complete file bytes: 14-byte header, definition + data messages
    (file_id, records, one lap, one session, one activity), trailing CRC-16.

    ``sample_interval_s`` defaults to 1 Hz like a real head unit: the analytics
    resampler interpolates gaps only up to ``MAX_INTERP_GAP_S`` (3 s, ANL-R8), so a
    sparser forged stream would fail closed out of every power/HR load metric.
    """
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware (UTC)")
    start = start.astimezone(_dt.UTC)
    n_samples = duration_s // sample_interval_s
    start_ts = _fit_ts(start)
    end_ts = start_ts + duration_s

    body = bytearray()
    body += _FILE_ID.definition()
    body += _FILE_ID.data(4, 1, 3121, serial_number, start_ts)

    body += _RECORD.definition()
    distance_cm = 0
    powers: list[int] = []
    hrs: list[int] = []
    cadences: list[int] = []
    lat0 = int(45.0 * _DEG_TO_SEMICIRCLE)
    lon0 = int(7.0 * _DEG_TO_SEMICIRCLE)
    for i in range(n_samples):
        power, hr, cadence, speed_mms = _sample(i)
        powers.append(power)
        hrs.append(hr)
        cadences.append(cadence)
        distance_cm += speed_mms * sample_interval_s // 10  # mm/s * s -> cm
        body += _RECORD.data(
            start_ts + i * sample_interval_s,
            lat0 + i * 20,
            lon0 + i * 20,
            (100 + 500) * 5,  # flat 100 m altitude
            hr,
            cadence,
            distance_cm,
            speed_mms,
            power,
            21,
        )

    avg_power = sum(powers) // len(powers)
    max_power = max(powers)
    avg_hr = sum(hrs) // len(hrs)
    max_hr = max(hrs)
    avg_cadence = sum(cadences) // len(cadences)
    duration_ms = duration_s * 1000

    body += _LAP.definition()
    body += _LAP.data(
        end_ts, 9, 1, start_ts, duration_ms, duration_ms, distance_cm,
        avg_hr, max_hr, avg_cadence, avg_power, max_power,
    )  # fmt: skip
    body += _SESSION.definition()
    body += _SESSION.data(
        end_ts, 8, 1, start_ts, 2, 0, duration_ms, duration_ms, distance_cm,
        avg_hr, max_hr, avg_cadence, avg_power, max_power, 0, 1,
    )  # fmt: skip
    body += _ACTIVITY.definition()
    body += _ACTIVITY.data(end_ts, duration_ms, 1, 0, 26, 1)

    header = struct.pack("<BBHI4s", 14, 0x10, 2132, len(body), b".FIT")
    header += struct.pack("<H", _crc16(header))
    payload = header + bytes(body)
    return payload + struct.pack("<H", _crc16(payload))


@dataclass(frozen=True)
class ForgedRide:
    """One forged activity: upload filename, FIT bytes, and its start instant."""

    filename: str
    payload: bytes
    start: _dt.datetime


def forge_recent_batch(
    *,
    now: _dt.datetime | None = None,
    days_back: tuple[int, ...] = DEFAULT_DAYS_BACK,
) -> list[ForgedRide]:
    """Forge one ride per ``days_back`` entry, timestamps relative to ``now``.

    Each ride starts at 10:00 UTC the given number of days before ``now`` and carries
    a distinct serial_number, so every forged activity has its own STRONG fingerprint.
    """
    anchor = (now or _dt.datetime.now(_dt.UTC)).astimezone(_dt.UTC)
    rides: list[ForgedRide] = []
    for i, days in enumerate(days_back):
        start = (anchor - _dt.timedelta(days=days)).replace(
            hour=10, minute=0, second=0, microsecond=0
        )
        rides.append(
            ForgedRide(
                filename=f"forged_ride_minus_{days}d.fit",
                payload=forge_ride(start=start, serial_number=77_000_001 + i),
                start=start,
            )
        )
    return rides


def main() -> int:
    """CLI: write the forged recent batch to ``--out-dir`` for manual inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ride in forge_recent_batch():
        path = args.out_dir / ride.filename
        path.write_bytes(ride.payload)
        print(f"wrote {path} ({len(ride.payload)} bytes, start={ride.start.isoformat()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
