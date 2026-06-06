"""Pure Intervals.icu source-vocabulary -> canonical mapping tables and helpers.

The single place the Intervals adapter translates source vocabulary into canonical
terms (MAP-R4): sport / sub_sport registry codes, stream-channel names + sampling
basis, and per-sample numeric coercion (ADP-R10/R12). Kept apart from the adapter's
client + dispatch (``intervals_icu.py``) to bound module size (QUAL-R9). Every
function here is pure and deterministic (no clock, no I/O) so the adapter's ``map``
stays golden-testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Protocol

from wattwise_core.domain.enums import DeviceClass, SampleBasis, StreamChannelName

# Source stream ``type`` -> canonical channel + sampling basis (MAP-R4). ``distance``
# is a TIME-sampled cumulative-metres channel (matches the file-upload adapter's
# ``distance_m`` basis), NOT a distance-indexed axis (MAP-R3 cross-adapter consistency).
STREAM_CHANNELS: Final[dict[str, tuple[StreamChannelName, SampleBasis]]] = {
    "watts": (StreamChannelName.POWER_W, SampleBasis.TIME),
    "heartrate": (StreamChannelName.HR_BPM, SampleBasis.TIME),
    "cadence": (StreamChannelName.CADENCE_RPM, SampleBasis.TIME),
    "velocity_smooth": (StreamChannelName.SPEED_MPS, SampleBasis.TIME),
    "distance": (StreamChannelName.DISTANCE_M, SampleBasis.TIME),
    "altitude": (StreamChannelName.ALTITUDE_M, SampleBasis.TIME),
    "latlng": (StreamChannelName.LATLNG, SampleBasis.TIME),
    "temp": (StreamChannelName.TEMP_C, SampleBasis.TIME),
}

# Source ``sub_type`` vocab -> canonical sub_sport registry code (MAP-R4/R2). Values
# are SEEDED ``{sport}_other`` codes (migration 0001 §_seed) so the emitted value is a
# valid FK into the ``sub_sport`` registry and never a raw source token; an unmapped
# token resolves to ``None`` (a typed gap, never a fabricated/un-seeded code).
_SUB_SPORT_CODES: Final[dict[str, str]] = {
    "gravel": "cycling_other",
    "mountainbike": "cycling_other",
    "mtb": "cycling_other",
    "cyclocross": "cycling_other",
    "road": "cycling_other",
    "indoor": "cycling_other",
    "virtual": "cycling_other",
    "track": "running_other",
    "trail": "running_other",
    "treadmill": "running_other",
    "road_run": "running_other",
    "openwater": "swimming_other",
    "pool": "swimming_other",
    "indoor_rowing": "rowing_other",
    "classic": "xc_ski_other",
    "skate": "xc_ski_other",
}

# Source activity ``type`` vocab -> canonical sport registry code (MAP-R4).
# Unknown tokens map to "other" (never a passthrough of the raw source token).
_SPORT_CODES: Final[dict[str, str]] = {
    "ride": "cycling",
    "virtualride": "cycling",
    "gravelride": "cycling",
    "mountainbikeride": "cycling",
    "ebikeride": "cycling",
    "run": "running",
    "virtualrun": "running",
    "trailrun": "running",
    "walk": "other",
    "hike": "other",
    "swim": "swimming",
    "openwaterswim": "swimming",
    "rowing": "rowing",
    "kayaking": "rowing",
    "nordicski": "xc_ski",
    "backcountryski": "xc_ski",
    "weighttraining": "strength",
    "workout": "strength",
}


class _ProvenanceFlags(Protocol):
    """The activity provenance flags ``device_class`` reads (structural, source-blind)."""

    @property
    def trainer(self) -> bool | None: ...
    @property
    def power_meter(self) -> bool | None: ...
    @property
    def device_watts(self) -> bool | None: ...


class _StreamRow(Protocol):
    """One source stream channel as ``build_streams`` consumes it (structural)."""

    @property
    def type(self) -> str: ...
    @property
    def data(self) -> list[Any]: ...


def sport_code(raw_type: str | None) -> str:
    """Map a source activity ``type`` to a canonical sport code (MAP-R4)."""
    if raw_type is None:
        return "other"
    return _SPORT_CODES.get(raw_type.strip().lower(), "other")


def sub_sport_code(raw_sub_type: str | None) -> str | None:
    """Map a source ``sub_type`` to a canonical sub_sport registry code (MAP-R4/R2).

    Translates through ``_SUB_SPORT_CODES`` (seeded registry codes); an unmapped or
    absent token yields ``None`` (a typed gap). NEVER echoes the raw source token.
    """
    if not isinstance(raw_sub_type, str):
        return None
    token = raw_sub_type.strip().lower().replace(" ", "_").replace("-", "_")
    if not token or token in ("generic", "all"):
        return None
    return _SUB_SPORT_CODES.get(token)


def as_float(value: Any) -> float | None:
    """Coerce one stream sample to ``float`` or ``None`` (ADP-R10/TIER-R5).

    A non-numeric element (dict/list/str/bool) becomes ``None`` (a typed gap) so a
    canonical SI channel can never hold an un-typed blob — mirrors the FIT/GPX/TCX
    decoders' ``_as_float`` (CON-R3 cross-adapter consistency).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def device_class(act: _ProvenanceFlags) -> str:
    """Derive a canonical device class from provenance flags (MAP-R2/R12; never a name).

    Only an actual provenance signal assigns a concrete class. Speed/distance alone do
    NOT establish a GPS watch (a trainer or phone produces both), so an absent signal
    stays ``UNKNOWN`` rather than fabricating a plausible class (MAP-R12).
    """
    if act.trainer:
        return DeviceClass.TRAINER.value
    if act.power_meter or act.device_watts:
        return DeviceClass.POWERMETER.value
    return DeviceClass.UNKNOWN.value


def _coerce_sample(channel: StreamChannelName, value: Any) -> Any:
    """Coerce one source stream element to a typed canonical sample (ADP-R10/R12).

    A scalar channel value is coerced to ``float | None`` (a non-numeric blob becomes a
    typed gap, never passed through). ``latlng`` is a paired channel: a clean
    ``[lat, lon]`` float pair survives; anything else (dict/str/short/non-numeric)
    becomes ``None``. This guarantees no canonical SI channel holds a dict/list/str.
    """
    if channel is StreamChannelName.LATLNG:
        if not isinstance(value, list | tuple) or len(value) != 2:
            return None
        lat, lon = as_float(value[0]), as_float(value[1])
        return None if lat is None or lon is None else [lat, lon]
    return as_float(value)


def build_streams(streams: Sequence[_StreamRow]) -> dict[str, dict[str, Any]]:
    """Map source streams to canonical channels with gaps as ``None`` (MAP-R5/ADP-R10)."""
    out: dict[str, dict[str, Any]] = {}
    for row in streams:
        mapped = STREAM_CHANNELS.get(row.type)
        if mapped is None:
            continue
        channel, basis = mapped
        out[channel.value] = {
            "values": [_coerce_sample(channel, v) for v in row.data],
            "sample_basis": basis.value,
            "sample_rate_hz": 1.0,
        }
    return out


__all__ = [
    "STREAM_CHANNELS",
    "as_float",
    "build_streams",
    "device_class",
    "sport_code",
    "sub_sport_code",
]
