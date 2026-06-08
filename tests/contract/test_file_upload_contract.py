"""Contract tests for the FIT/GPX/TCX/PWX file-upload adapter (ADP-R17, TST-R1, FIL-R*).

Offline-only (TIER-R1): every fixture under ``fixtures/file_upload/`` is decoded
(impure I/O) then run through the pure :meth:`FileUploadAdapter.map`, and the
ASBO -> GBO mapping is asserted on:

* canonical activity payload + per-sample streams + laps (MAP-R2/R3): canonical field
  names, SI units, canonical stream channels, FIT semicircle -> WGS84 degrees;
* provenance / trust (PRV-R7): a real per-sample stream -> ``raw_stream``;
* an unknown source sport maps to ``"other"`` (MAP-R4), never a passthrough;
* free text (title/description/notes) tagged untrusted (MAP-R7) for injection
  quarantine — and never interpreted;
* real gaps preserved as ``None`` never ``0`` (MAP-R5);
* ``source_native_id`` per LIN-R1.1 (FIT file_id fingerprint; GPX/TCX/PWX
  start+elapsed+extent fingerprint) and byte-identical re-decode determinism
  (GBO-AC-1, FIL-R3/FIL-R5);
* the ``fitdecode`` corrupt/truncated-file recovery fallback (CLI-R13);
* every connectionless upload lands under the single ``file_import`` descriptor
  (LIN-R1.1) — no per-platform descriptor inferred (Principle A, CLI-R14).

The FIT fixtures are SYNTHESIZED with the official ``garmin-fit-sdk`` encoder and
committed as recorded bytes (see module ``conftest``-free note); GPX/TCX/PWX are
hand-authored XML.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import (
    DeviceClass,
    Fidelity,
    SampleBasis,
    SourceKind,
    StreamChannelName,
)
from wattwise_core.ingestion.adapters._asbo import ActivityAsbo, FileDecodeError
from wattwise_core.ingestion.adapters._decode_pwx import decode_pwx
from wattwise_core.ingestion.adapters.file_upload import (
    FILE_IMPORT_SOURCE_KEY,
    FileUploadAdapter,
    decode,
    detect_format,
    native_id,
)
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef

pytestmark = pytest.mark.contract

_FIXTURES = Path(__file__).parent / "fixtures" / "file_upload"
_FETCHED_AT = _dt.datetime(2026, 6, 6, 12, 0, tzinfo=_dt.UTC)
_EXPECTED_START = _dt.datetime(2024, 1, 2, 10, 0, 0, tzinfo=_dt.UTC)


def _descriptor() -> SourceDescriptorRef:
    return SourceDescriptorRef(
        source_descriptor_id="sd-file-import",
        source_key=FILE_IMPORT_SOURCE_KEY,
        kind=SourceKind.FILE_UPLOAD,
    )


def _ctx() -> FetchContext:
    return FetchContext(ingest_run_id="run-1", fetched_at=_FETCHED_AT, connection_id=None)


def _read(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def _map_file(name: str) -> tuple[bytes, ActivityAsbo, list[GboCandidate]]:
    raw = _read(name)
    asbo = decode(raw, filename=name)
    cands = FileUploadAdapter().map_upload(raw, asbo, _descriptor(), _ctx())
    return raw, asbo, cands


# --------------------------------------------------------------------------- FIT


def test_fit_decodes_and_maps_canonical_activity() -> None:
    _raw, asbo, cands = _map_file("ride.fit")
    assert len(asbo.records) == 3
    assert len(cands) == 1
    cand = cands[0]
    assert cand.gbo_type == "activity"
    assert cand.source_descriptor_id == "sd-file-import"
    assert cand.fetched_at == _FETCHED_AT
    p = cand.payload
    assert p["start_time"] == _EXPECTED_START
    assert p["sport"] == "cycling"  # MAP-R4 known token
    assert p["device_class"] == DeviceClass.POWERMETER.value
    assert p["avg_power_w"] == 210.0
    assert p["max_power_w"] == 220.0
    assert p["elevation_gain_m"] == 2.0
    assert p["elapsed_time_s"] == 3
    assert p["has_power"] is True
    assert p["has_gps"] is True
    assert p["has_hr"] is True


def test_fit_streams_are_canonical_si_with_semicircle_conversion() -> None:
    _raw, _asbo, cands = _map_file("ride.fit")
    streams = cands[0].payload["streams"]
    assert StreamChannelName.POWER_W.value in streams
    power = streams[StreamChannelName.POWER_W.value]
    assert power["values"] == [200.0, 210.0, 220.0]
    assert power["sample_basis"] == SampleBasis.TIME.value
    # FIT semicircles -> WGS84 degrees (MAP-R3): 45 deg lat, ~7 deg lon.
    lat0, lon0 = streams[StreamChannelName.LATLNG.value]["values"][0]
    assert lat0 == pytest.approx(45.0, abs=1e-5)
    assert lon0 == pytest.approx(7.0, abs=1e-5)
    # RR intervals are event-spaced (GBO-R21) and in milliseconds (MAP-R3).
    rr = streams[StreamChannelName.RR_INTERVALS_MS.value]
    assert rr["sample_basis"] == SampleBasis.EVENT.value
    assert rr["values"] == [pytest.approx(789.0), pytest.approx(812.0)]


def test_fit_trust_tier_is_raw_stream_for_real_streams() -> None:
    _raw, _asbo, cands = _map_file("ride.fit")
    assert cands[0].trust_tier is Fidelity.RAW_STREAM


def test_fit_laps_are_contiguous_with_relative_offsets() -> None:
    _raw, _asbo, cands = _map_file("ride.fit")
    laps = cands[0].payload["laps"]
    assert len(laps) == 1
    lap = laps[0]
    assert lap["lap_index"] == 0
    assert lap["start_offset_s"] == 0
    assert lap["duration_s"] == 3
    assert lap["avg_power_w"] == 210.0


def test_fit_source_native_id_is_file_id_fingerprint() -> None:
    raw, asbo, cands = _map_file("ride.fit")
    # LIN-R1.1: manufacturer+product+serial_number+time_created.
    assert asbo.native_fingerprint is not None
    assert "garmin" in asbo.native_fingerprint
    assert "1234567" in asbo.native_fingerprint
    assert cands[0].source_native_id == asbo.native_fingerprint
    assert cands[0].source_native_id == native_id(asbo, raw)


def test_fit_unknown_sport_maps_to_other() -> None:
    raw = _read("unknown_sport.fit")
    asbo = decode(raw, filename="unknown_sport.fit")
    cands = FileUploadAdapter().map_upload(raw, asbo, _descriptor(), _ctx())
    assert cands[0].payload["sport"] == "other"  # MAP-R4: tennis is unmodeled
    # No power/GPS stream -> not a powermeter, HR-only summary fidelity downgrade.
    assert cands[0].payload["has_power"] is False


# --------------------------------------------------------------------------- GPX


def test_gpx_decodes_and_maps_with_extensions() -> None:
    _raw, asbo, cands = _map_file("ride.gpx")
    assert len(asbo.records) == 3
    p = cands[0].payload
    assert p["start_time"] == _EXPECTED_START
    assert p["sport"] == "other"  # "kitesurfing" is unmodeled (MAP-R4)
    streams = p["streams"]
    assert streams[StreamChannelName.POWER_W.value]["values"] == [200.0, 210.0, 220.0]
    assert streams[StreamChannelName.HR_BPM.value]["values"] == [140.0, 142.0, 144.0]
    assert streams[StreamChannelName.LATLNG.value]["values"][0] == [
        pytest.approx(45.0),
        pytest.approx(7.0),
    ]
    assert p["has_gps"] is True
    assert cands[0].trust_tier is Fidelity.RAW_STREAM


def test_gpx_free_text_title_is_untrusted_and_not_interpreted() -> None:
    _raw, _asbo, cands = _map_file("ride.gpx")
    # MAP-R7: a title carrying an injection string is flagged, never acted on; the
    # canonical payload must NOT contain the raw free text at all.
    assert cands[0].untrusted_content is True
    assert "ignore previous instructions" not in str(cands[0].payload)


def test_gpx_missing_temperature_on_last_point_is_none_not_zero() -> None:
    _raw, _asbo, cands = _map_file("ride.gpx")
    temp = cands[0].payload["streams"][StreamChannelName.TEMP_C.value]["values"]
    # Third point has no atemp -> real gap is None, never 0 (MAP-R5).
    assert temp == [21.0, 21.0, None]


# --------------------------------------------------------------------------- TCX


def test_tcx_decodes_and_maps_with_tpx_watts_and_speed() -> None:
    _raw, asbo, cands = _map_file("ride.tcx")
    assert len(asbo.records) == 3
    assert len(asbo.laps) == 1
    p = cands[0].payload
    assert p["start_time"] == _EXPECTED_START
    assert p["sport"] == "cycling"  # "Biking" -> cycling (MAP-R4)
    streams = p["streams"]
    assert streams[StreamChannelName.POWER_W.value]["values"] == [200.0, 210.0, 220.0]
    assert streams[StreamChannelName.SPEED_MPS.value]["values"] == [8.0, 8.2, 8.4]
    assert streams[StreamChannelName.HR_BPM.value]["values"] == [140.0, 142.0, 144.0]
    assert streams[StreamChannelName.DISTANCE_M.value]["values"] == [0.0, 8.0, 16.0]
    assert p["device_class"] == DeviceClass.POWERMETER.value
    assert cands[0].trust_tier is Fidelity.RAW_STREAM


def test_tcx_lap_summary_and_notes_untrusted() -> None:
    _raw, _asbo, cands = _map_file("ride.tcx")
    lap = cands[0].payload["laps"][0]
    assert lap["distance_m"] == 24.0
    assert lap["avg_hr_bpm"] == 142.0
    assert lap["max_hr_bpm"] == 144.0
    assert cands[0].untrusted_content is True  # <Notes> is free text (MAP-R7)


def test_tcx_source_native_id_is_format_fingerprint() -> None:
    raw, asbo, cands = _map_file("ride.tcx")
    # LIN-R1.1 GPX/TCX: first start instant + total elapsed + total distance.
    assert asbo.native_fingerprint is not None
    assert "2024-01-02T10:00:00+00:00" in asbo.native_fingerprint
    assert cands[0].source_native_id == native_id(asbo, raw)


# --------------------------------------------------------------------------- PWX


def test_pwx_decodes_and_maps_canonical_activity() -> None:
    """PWX samples map to canonical SI streams + powermeter device class (MAP-R2/R3)."""
    _raw, asbo, cands = _map_file("ride.pwx")
    assert len(asbo.records) == 3
    assert len(cands) == 1
    cand = cands[0]
    assert cand.gbo_type == "activity"
    assert cand.source_descriptor_id == "sd-file-import"
    assert cand.fetched_at == _FETCHED_AT
    assert cand.observed_at == _EXPECTED_START
    assert cand.trust_tier is Fidelity.RAW_STREAM
    p = cand.payload
    assert p["start_time"] == _EXPECTED_START
    assert p["sport"] == "cycling"  # MAP-R4: PWX "Bike" -> cycling
    assert p["sub_sport"] is None
    assert p["elapsed_time_s"] == 3
    assert p["distance_m"] == 24.0
    assert p["total_work_j"] == 630.0
    assert p["energy_kj"] == 0.63
    assert p["avg_power_w"] == 210.0
    assert p["max_power_w"] == 220.0
    assert p["avg_hr_bpm"] == 142.0
    assert p["max_hr_bpm"] == 144.0
    assert p["avg_cadence_rpm"] == 91.0
    assert p["avg_speed_mps"] == 8.2
    assert p["elevation_gain_m"] is None  # MAP-R5: a real gap stays None, never 0
    assert p["device_class"] == DeviceClass.POWERMETER.value
    assert p["has_power"] is True
    assert p["has_hr"] is True
    assert p["has_gps"] is True
    assert p["has_cadence"] is True
    streams = p["streams"]
    assert streams[StreamChannelName.POWER_W.value]["values"] == [200.0, 210.0, 220.0]
    assert streams[StreamChannelName.HR_BPM.value]["values"] == [140.0, 142.0, 144.0]
    assert streams[StreamChannelName.CADENCE_RPM.value]["values"] == [90.0, 91.0, 92.0]
    assert streams[StreamChannelName.SPEED_MPS.value]["values"] == [8.0, 8.2, 8.4]
    assert streams[StreamChannelName.DISTANCE_M.value]["values"] == [0.0, 8.0, 16.0]
    assert streams[StreamChannelName.ALTITUDE_M.value]["values"] == [100.0, 101.0, 102.0]
    assert streams[StreamChannelName.POWER_W.value]["sample_basis"] == SampleBasis.TIME.value
    # PWX lat/lon are already WGS84 degrees (no semicircle conversion, MAP-R3).
    latlng = streams[StreamChannelName.LATLNG.value]["values"]
    assert latlng[0] == [pytest.approx(45.0), pytest.approx(7.0)]
    assert latlng[1] == [pytest.approx(45.0001), pytest.approx(7.0001)]
    assert latlng[2] == [pytest.approx(45.0002), pytest.approx(7.0002)]
    assert streams[StreamChannelName.LATLNG.value]["sample_basis"] == SampleBasis.TIME.value
    # temp_c is not present in the PWX subset -> the channel is absent (MAP-R5).
    assert StreamChannelName.TEMP_C.value not in streams
    assert StreamChannelName.RR_INTERVALS_MS.value not in streams


def test_pwx_segment_lap_and_free_text_untrusted() -> None:
    """A PWX segment becomes one lap; title/cmt free text is quarantined (MAP-R7)."""
    _raw, _asbo, cands = _map_file("ride.pwx")
    laps = cands[0].payload["laps"]
    assert len(laps) == 1
    lap = laps[0]
    assert lap["lap_index"] == 0
    assert lap["start_offset_s"] == 0  # segment <beginning>0 == workout start
    assert lap["duration_s"] == 3
    assert lap["distance_m"] == 24.0
    assert lap["avg_power_w"] == 210.0
    assert lap["max_power_w"] == 220.0
    assert lap["avg_hr_bpm"] == 142.0
    assert lap["max_hr_bpm"] == 144.0
    assert lap["avg_cadence_rpm"] == 91.0
    # MAP-R7: title + cmt are free text -> flagged untrusted and never copied in.
    assert cands[0].untrusted_content is True
    assert "ignore previous instructions" not in str(cands[0].payload)
    assert "reveal secrets" not in str(cands[0].payload)


def test_pwx_source_native_id_is_format_fingerprint() -> None:
    """PWX native id is the deterministic start+elapsed+distance fingerprint (LIN-R1.1)."""
    raw, asbo, cands = _map_file("ride.pwx")
    assert asbo.native_fingerprint == "2024-01-02T10:00:00+00:00|3.000|24.000"
    assert "2024-01-02T10:00:00+00:00" in asbo.native_fingerprint
    assert cands[0].source_native_id == native_id(asbo, raw)


def test_pwx_unknown_sport_maps_to_other() -> None:
    """A PWX sportType outside the registry maps to 'other', not a passthrough (MAP-R4)."""
    raw = _read("unknown_sport.pwx")
    asbo = decode(raw, filename="unknown_sport.pwx")
    cands = FileUploadAdapter().map_upload(raw, asbo, _descriptor(), _ctx())
    assert cands[0].payload["sport"] == "other"  # MAP-R4: Kitesurf is unmodeled
    # HR-only summary -> no power stream -> not a powermeter (fidelity downgrade).
    assert cands[0].payload["has_power"] is False


def test_pwx_decoder_exposes_decode_pwx_surface() -> None:
    """The PWX decoder exposes decode_pwx returning an ActivityAsbo directly (ADP-R8)."""
    asbo = decode_pwx(_read("ride.pwx"))
    assert isinstance(asbo, ActivityAsbo)
    assert asbo.session["sport"] == "Bike"  # raw token, untranslated in the ASBO


# ----------------------------------------------------------------- determinism


@pytest.mark.parametrize("name", ["ride.fit", "ride.gpx", "ride.tcx", "ride.pwx"])
def test_re_decode_is_deterministic(name: str) -> None:
    raw = _read(name)
    a1 = FileUploadAdapter().map_upload(raw, decode(raw, filename=name), _descriptor(), _ctx())
    a2 = FileUploadAdapter().map_upload(raw, decode(raw, filename=name), _descriptor(), _ctx())
    assert a1[0].content_hash == a2[0].content_hash  # GBO-AC-1
    assert a1[0].source_native_id == a2[0].source_native_id  # FIL-R5
    assert a1[0].payload == a2[0].payload


@pytest.mark.parametrize("name", ["ride.fit", "ride.gpx", "ride.tcx", "ride.pwx"])
def test_all_formats_land_under_the_single_file_import_descriptor(name: str) -> None:
    # CLI-R14 / LIN-R1.1: no per-platform descriptor is inferred from contents.
    _raw, _asbo, cands = _map_file(name)
    assert cands[0].source_descriptor_id == "sd-file-import"


# -------------------------------------------------------- fitdecode fallback


def test_truncated_fit_recovers_via_fitdecode_fallback() -> None:
    raw = _read("ride.fit")
    # Truncate the trailing 2-byte CRC: the strict SDK read fails integrity, the
    # ``fitdecode`` CRC-ignore fallback still recovers the records (CLI-R13).
    truncated = raw[:-2]
    asbo = decode(truncated, filename="ride.fit")
    assert len(asbo.records) >= 1
    cands = FileUploadAdapter().map_upload(truncated, asbo, _descriptor(), _ctx())
    assert cands[0].payload["sport"] in {"cycling", "other"}


# ----------------------------------------------------------------- detection


def test_detect_format_uses_magic_bytes_over_extension() -> None:
    assert detect_format("activity.bin", _read("ride.fit")) == "fit"
    assert detect_format(None, _read("ride.gpx")) == "gpx"
    assert detect_format(None, _read("ride.tcx")) == "tcx"
    assert detect_format(None, _read("ride.pwx")) == "pwx"


def test_unrecognized_bytes_fail_closed() -> None:
    with pytest.raises(FileDecodeError):
        decode(b"this is not an activity file", filename="notes.txt")


def test_map_ignores_non_asbo_input() -> None:
    # The pure map must not crash on a wrong-typed input; it emits nothing.
    assert FileUploadAdapter().map(object(), _descriptor(), _ctx()) == []


def test_pwx_non_finite_numbers_become_typed_gaps_not_nan() -> None:
    """A NaN/Infinity token decodes to a typed gap (None), keeping the payload strict JSON.

    Non-finite floats would make the canonical payload invalid JSON (rejected by Postgres
    JSONB) and non-deterministic (``nan != nan``); they must be dropped, not carried (MAP-R5).
    """
    pwx = (
        b'<?xml version="1.0"?>'
        b'<pwx xmlns="http://www.peaksware.com/PWX/1/0"><workout>'
        b"<sportType>Bike</sportType><time>2024-01-02T10:00:00Z</time>"
        b"<summarydata><duration>3</duration><dist>24</dist>"
        b'<pwr avg="NaN" max="Infinity"/></summarydata>'
        b"<sample><timeoffset>0</timeoffset><pwr>NaN</pwr><hr>140</hr></sample>"
        b"</workout></pwx>"
    )
    asbo = decode(pwx, filename="bad.pwx")
    cands = FileUploadAdapter().map_upload(pwx, asbo, _descriptor(), _ctx())
    assert len(cands) == 1
    payload = cands[0].payload
    assert payload["avg_power_w"] is None
    assert payload["max_power_w"] is None
    # Strict JSON round-trip (json.loads rejects bare NaN/Infinity by default).
    rendered = json.dumps(payload, default=str)
    assert "NaN" not in rendered and "Infinity" not in rendered
    json.loads(rendered)


def test_pwx_summary_only_export_maps_to_summary_only_candidate() -> None:
    """A PWX with a populated <summarydata> but no <sample>/<segment> is kept (SUMMARY_ONLY).

    A summary-only export is a legitimate, common shape — it must map to a typed
    summary-fidelity candidate, not be dropped at decode (ING-R3 partial GBO, DOD-R2).
    """
    _raw, _asbo, cands = _map_file("summary_only.pwx")
    assert len(cands) == 1
    candidate = cands[0]
    assert candidate.trust_tier is Fidelity.SUMMARY_ONLY  # no per-sample stream
    payload = candidate.payload
    assert payload["sport"] == "cycling"
    assert payload["avg_power_w"] == 250.0
    assert payload["total_work_j"] == 900000.0  # 900 kJ -> J (decode-time conversion)
