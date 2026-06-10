"""Decoder error/edge branches for the file-upload formats (CLI-R13, TIER-R5, MAP-R5).

Targets the fail-closed paths of the pure TCX/GPX/PWX/FIT decode layers and the
``file_upload`` dispatch: corrupt/empty documents raise the typed
:class:`FileDecodeError` (never a bare crash); absent optional elements become typed
gaps (``None``, never ``0``); unparseable scalar/instant tokens are dropped rather
than fabricated; gzip-wrapped uploads are unwrapped and a corrupt gzip fails closed.
All inputs are crafted in-test bytes — no fixtures, no network (TST-R1).
"""

from __future__ import annotations

import datetime as _dt
import gzip
from typing import Any, cast

import pytest
from lxml import etree

from wattwise_core.domain.enums import SourceKind
from wattwise_core.ingestion.adapters import _decode_fit as dfit
from wattwise_core.ingestion.adapters import _decode_gpx as dgpx
from wattwise_core.ingestion.adapters import _decode_pwx as dpwx
from wattwise_core.ingestion.adapters import _decode_tcx as dtcx
from wattwise_core.ingestion.adapters._asbo import ActivityAsbo, AsboRecord, FileDecodeError
from wattwise_core.ingestion.adapters._decode_gpx import decode_gpx
from wattwise_core.ingestion.adapters._decode_pwx import decode_pwx
from wattwise_core.ingestion.adapters._decode_tcx import decode_tcx
from wattwise_core.ingestion.adapters.file_upload import (
    FileUploadAdapter,
    decode,
    native_id,
)
from wattwise_core.ingestion.base import FetchContext, FileImportError, SourceDescriptorRef
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.unit

_FETCHED_AT = _dt.datetime(2026, 6, 6, 12, 0, tzinfo=_dt.UTC)


def _descriptor() -> SourceDescriptorRef:
    return SourceDescriptorRef("sd-uuid-1", "file_import", SourceKind.FILE_UPLOAD)


def _ctx() -> FetchContext:
    return FetchContext(ingest_run_id="run-1", fetched_at=_FETCHED_AT, connection_id=None)


# --------------------------------------------------------------------------- TCX


def test_tcx_without_activity_element_fails_closed() -> None:
    """TIER-R5: a TCX with no <Activity> raises the typed FileDecodeError, never a crash."""
    doc = b"<TrainingCenterDatabase><Activities/></TrainingCenterDatabase>"
    with pytest.raises(FileDecodeError, match="no Activity"):
        decode_tcx(doc)


def test_tcx_lap_without_start_time_falls_back_to_record_timestamp() -> None:
    """LIN-R1.1: with no lap StartTime the fingerprint anchors on the first record instant."""
    doc = (
        b"<TrainingCenterDatabase><Activities><Activity Sport='Biking'>"
        b"<Lap><TotalTimeSeconds>60</TotalTimeSeconds><Track><Trackpoint>"
        b"<Time>2026-06-01T10:00:00Z</Time></Trackpoint></Track></Lap>"
        b"</Activity></Activities></TrainingCenterDatabase>"
    )
    asbo = decode_tcx(doc)
    assert asbo.native_fingerprint is not None
    assert asbo.native_fingerprint.startswith("2026-06-01T10:00:00+00:00|")
    # Sport attribute is surfaced; absent Notes leaves no title (MAP-R5: gap, not "").
    assert asbo.session.get("sport") == "Biking"
    assert "title" not in asbo.session


def test_tcx_with_no_timestamps_anywhere_has_no_fingerprint() -> None:
    """LIN-R1.1: a TCX whose laps/records carry no instant yields fingerprint=None."""
    doc = (
        b"<TrainingCenterDatabase><Activities><Activity>"
        b"<Notes>morning ride</Notes>"
        b"<Lap><Track><Trackpoint><Cadence>abc</Cadence>"
        b"<Position><LatitudeDegrees>50.0</LatitudeDegrees></Position>"
        b"<Extensions><TPX><Speed>nan</Speed></TPX></Extensions>"
        b"<!-- a comment child exercises the non-string tag path -->"
        b"</Trackpoint></Track></Lap>"
        b"</Activity></Activities></TrainingCenterDatabase>"
    )
    asbo = decode_tcx(doc)
    assert asbo.native_fingerprint is None
    # Notes become the (untrusted) title; no Sport attribute -> no sport key.
    assert asbo.session.get("title") == "morning ride"
    assert "sport" not in asbo.session
    rec = asbo.records[0]
    # Unparseable / non-finite tokens and a half-present Position are typed gaps.
    assert rec.cadence_rpm is None
    assert rec.speed_mps is None
    assert rec.latlng is None
    assert rec.timestamp is None


def test_tcx_extensions_without_watts_and_naive_or_bad_instants() -> None:
    """MAP-R5: a TPX without Watts is a gap; naive instants are normalized to UTC."""
    doc = (
        b"<TrainingCenterDatabase><Activities><Activity>"
        b"<Lap StartTime='2026-06-01T10:00:00'><TotalTimeSeconds>10</TotalTimeSeconds>"
        b"<Track><Trackpoint><Time>not-a-date</Time>"
        b"<Extensions><TPX><Speed>3.5</Speed></TPX></Extensions>"
        b"</Trackpoint></Track></Lap>"
        b"</Activity></Activities></TrainingCenterDatabase>"
    )
    asbo = decode_tcx(doc)
    rec = asbo.records[0]
    assert rec.timestamp is None  # unparseable Time -> typed gap
    assert rec.power_w is None  # TPX present but no Watts -> gap
    assert rec.speed_mps == 3.5
    lap = asbo.laps[0]
    # Naive StartTime is interpreted as UTC (never dropped, never shifted).
    assert lap.start_time == _dt.datetime(2026, 6, 1, 10, 0, tzinfo=_dt.UTC)


def test_tcx_helper_edges_fail_closed() -> None:
    """MAP-R5: the TCX scalar/instant helpers drop bad tokens rather than fabricate."""
    assert dtcx._as_float("12,5") is None
    assert dtcx._as_float(None) is None
    assert dtcx._as_float("inf") is None
    assert dtcx._as_dt(123) is None
    assert dtcx._findall(None, "Lap") == []
    assert dtcx._local(etree.Comment("x")) == ""


# --------------------------------------------------------------------------- GPX


def test_gpx_without_track_points_fails_closed() -> None:
    """TIER-R5: a GPX with zero track points raises the typed FileDecodeError."""
    doc = (
        b"<?xml version='1.0'?><gpx version='1.1' creator='t' "
        b"xmlns='http://www.topografix.com/GPX/1/1'><trk><trkseg/></trk></gpx>"
    )
    with pytest.raises(FileDecodeError, match="no track points"):
        decode_gpx(doc)


def test_gpx_decoder_internal_typed_error_is_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI-R2: a FileDecodeError raised inside the parse seam surfaces verbatim (no rewrap)."""

    def _boom(_: str) -> Any:
        raise FileDecodeError("inner typed failure")

    monkeypatch.setattr(dgpx.gpxpy, "parse", _boom)
    with pytest.raises(FileDecodeError, match="inner typed failure"):
        decode_gpx(b"<gpx><trk><trkseg><trkpt lat='1' lon='1'/></trkseg></trk></gpx>")


def test_gpx_title_falls_back_to_gpx_level_name_and_no_timestamps_no_fingerprint() -> None:
    """MAP-R7/LIN-R1.1: track without a name uses the gpx-level name; no time -> no print."""
    doc = (
        b"<?xml version='1.0'?><gpx version='1.1' creator='t' "
        b"xmlns='http://www.topografix.com/GPX/1/1'>"
        b"<metadata><name>file level</name></metadata>"
        b"<trk><trkseg><trkpt lat='50.0' lon='8.0'/></trkseg></trk></gpx>"
    )
    asbo = decode_gpx(doc)
    assert asbo.session.get("title") == "file level"
    assert "sport" not in asbo.session  # no track type -> typed gap
    assert asbo.native_fingerprint is None
    assert asbo.records[0].latlng == (50.0, 8.0)


def test_gpx_without_any_name_has_no_title() -> None:
    """MAP-R5: absent track and file names yield no title key (never an empty string)."""
    doc = (
        b"<?xml version='1.0'?><gpx version='1.1' creator='t' "
        b"xmlns='http://www.topografix.com/GPX/1/1'>"
        b"<trk><type>cycling</type><trkseg>"
        b"<trkpt lat='50.0' lon='8.0'><ele>120</ele></trkpt>"
        b"</trkseg></trk></gpx>"
    )
    asbo = decode_gpx(doc)
    assert "title" not in asbo.session
    assert asbo.session.get("sport") == "cycling"


def test_gpx_extension_walker_skips_comments_and_non_iterables() -> None:
    """MAP-R5: a comment node has no usable tag; a non-iterable extension is ignored."""
    out: dict[str, float] = {}
    parent = etree.fromstring("<ext><power>250</power><!-- c --></ext>")
    for child in parent:
        dgpx._walk_extension(child, out)
    assert out == {"power": 250.0}

    class _Opaque:
        tag = "hr"
        text = "150"

    out2: dict[str, float] = {}
    dgpx._walk_extension(_Opaque(), out2)  # not iterable -> no children walked
    assert out2 == {"hr": 150.0}
    assert dgpx._extensions_map(None) == {}


def test_gpx_scalar_and_instant_helpers_fail_closed() -> None:
    """MAP-R5: GPX scalar helpers drop bools/strings/None; naive instants become UTC."""
    assert dgpx._as_float(True) is None
    assert dgpx._as_float("5") is None
    assert dgpx._as_float_text(None) is None
    assert dgpx._latlng(None, 8.0) is None
    assert dgpx._as_dt("2026-06-01") is None  # non-datetime -> gap
    naive = _dt.datetime(2026, 6, 1, 10, 0)  # intentionally naive input
    assert dgpx._as_dt(naive) == naive.replace(tzinfo=_dt.UTC)


# --------------------------------------------------------------------------- PWX


def test_pwx_without_workout_element_fails_closed() -> None:
    """TIER-R5: a <pwx> with no <workout> raises the typed FileDecodeError."""
    with pytest.raises(FileDecodeError, match="no workout"):
        decode_pwx(b"<pwx xmlns='http://www.peaksware.com/PWX/1/0'/>")


def test_pwx_parse_seam_reraises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-R2: a FileDecodeError from the XML seam surfaces verbatim (no double-wrap)."""

    class _Etree:
        @staticmethod
        def XMLParser(**_: object) -> object:  # mirrors the lxml API
            return object()

        @staticmethod
        def fromstring(_: bytes, parser: object) -> Any:
            raise FileDecodeError("inner typed failure")

    monkeypatch.setattr(dpwx, "etree", _Etree)
    with pytest.raises(FileDecodeError, match="inner typed failure"):
        decode_pwx(b"<pwx><workout/></pwx>")


def test_pwx_without_workout_time_has_no_timestamps_and_no_fingerprint() -> None:
    """LIN-R1.1/MAP-R5: no <time> means no sample/lap instants and no fingerprint."""
    doc = (
        b"<pwx xmlns='http://www.peaksware.com/PWX/1/0'><workout>"
        b"<!-- comment exercises the non-string tag path -->"
        b"<sample><timeoffset>5</timeoffset><pwr>200</pwr></sample>"
        b"<segment><summarydata><beginning>0</beginning></summarydata></segment>"
        b"<segment/>"
        b"</workout></pwx>"
    )
    asbo = decode_pwx(doc)
    assert asbo.native_fingerprint is None
    assert asbo.records[0].timestamp is None  # offset present but no start instant
    assert asbo.records[0].power_w == 200.0
    assert asbo.laps[0].start_time is None
    # Second segment has no <summarydata> at all: every lap scalar is a typed gap.
    assert asbo.laps[1].duration_s is None
    assert asbo.laps[1].avg_power_w is None


def test_pwx_helper_edges_fail_closed() -> None:
    """MAP-R5: PWX scalar/instant helpers drop unusable tokens rather than fabricate."""
    assert dpwx._as_float("watts") is None
    assert dpwx._findall(None, "sample") == []
    assert dpwx._as_dt(123) is None
    assert dpwx._as_dt("yesterday") is None
    naive = _dt.datetime(2026, 6, 1, 9, 0)  # intentionally naive input
    assert dpwx._as_dt(naive) == naive.replace(tzinfo=_dt.UTC)
    assert dpwx._as_dt("2026-06-01T09:00:00") == _dt.datetime(2026, 6, 1, 9, 0, tzinfo=_dt.UTC)


# --------------------------------------------------------------------------- FIT


def test_fit_sdk_failure_is_wrapped_typed_then_recovery_also_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI-R13/TIER-R5: an SDK crash becomes FileDecodeError; failed recovery stays typed."""

    class _Decoder:
        def __init__(self, _: object) -> None:
            raise RuntimeError("sdk exploded")

    monkeypatch.setattr(dfit, "Decoder", _Decoder)
    with pytest.raises(FileDecodeError):
        dfit.decode_fit(b"\x0e\x10\x00\x00\x00\x00\x00\x00.FIT\x00\x00garbage")


def test_fit_recovery_path_reraises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-R13: a typed error inside the fitdecode recovery path surfaces verbatim."""

    class _Reader:
        def __init__(self, *_: object, **__: object) -> None:
            raise FileDecodeError("recovery typed failure")

    class _FakeFitdecode:
        FitReader = _Reader
        FitDataMessage = object
        CrcCheck = dfit.fitdecode.CrcCheck

    monkeypatch.setattr(dfit, "fitdecode", _FakeFitdecode)
    with pytest.raises(FileDecodeError, match="recovery typed failure"):
        dfit._decode_with_fitdecode(b"junk")


def test_fit_rr_fingerprint_and_instant_helpers_fail_closed() -> None:
    """MAP-R3/R10: non-list hrv time -> no RR; empty file_id -> no fingerprint."""
    assert dfit._rr_from_hrv("0.8") == []
    assert dfit._rr_from_hrv([None, -0.5, 0.5]) == [500.0]
    assert dfit._fit_fingerprint({}) is None
    naive = _dt.datetime(2026, 6, 1, 8, 0)  # intentionally naive input
    assert dfit._as_dt(naive) == naive.replace(tzinfo=_dt.UTC)


# -------------------------------------------------------------------- file_upload


def test_decode_rejects_non_bytes_input() -> None:
    """TIER-R5: a non-bytes upload fails closed with the typed FileDecodeError."""
    with pytest.raises(FileDecodeError, match="expects bytes"):
        decode(cast(Any, "not-bytes"))


def test_decode_unwraps_gzip_and_strips_gz_suffix_for_detection() -> None:
    """FIL-R1: a ``.gpx.gz`` upload is decompressed and format-detected on the inner name."""
    gpx = (
        b"<?xml version='1.0'?><gpx version='1.1' creator='t' "
        b"xmlns='http://www.topografix.com/GPX/1/1'><trk><trkseg>"
        b"<trkpt lat='50.0' lon='8.0'><time>2026-06-01T10:00:00Z</time></trkpt>"
        b"<trkpt lat='50.001' lon='8.001'><time>2026-06-01T10:00:05Z</time></trkpt>"
        b"</trkseg></trk></gpx>"
    )
    asbo = decode(gzip.compress(gpx), filename="ride.gpx.gz")
    assert len(asbo.records) == 2


def test_decode_corrupt_gzip_fails_closed() -> None:
    """TIER-R5: gzip magic with a corrupt body raises the typed FileDecodeError."""
    with pytest.raises(FileDecodeError, match="decompress"):
        decode(b"\x1f\x8b" + b"\x00" * 32, filename="ride.fit.gz")


def test_native_id_falls_back_to_content_hash_without_fingerprint() -> None:
    """LIN-R1.1: no native fingerprint -> the verbatim-bytes content hash, deterministic."""
    asbo = ActivityAsbo(records=(AsboRecord(timestamp=None),))
    raw = b"some verbatim upload bytes"
    assert native_id(asbo, raw) == content_hash(raw)


def test_map_without_usable_start_time_yields_no_candidate() -> None:
    """ING-R3/MAP-R5: no usable start instant -> no candidate, never a fabricated one."""
    adapter = FileUploadAdapter()
    asbo = ActivityAsbo(records=(AsboRecord(timestamp=None),))
    assert adapter.map(asbo, _descriptor(), _ctx()) == []
    assert adapter.map_upload(b"raw", asbo, _descriptor(), _ctx()) == []


def test_decode_upload_wraps_decoder_failure_in_neutral_file_import_error() -> None:
    """ARCH-R22/FIL-R1: a bad upload surfaces as FileImportError, not a decoder type."""
    adapter = FileUploadAdapter()
    with pytest.raises(FileImportError, match="could not decode"):
        adapter.decode_upload(
            b"\x00\x01\x02 utterly not an activity file",
            filename="mystery.bin",
            source_descriptor=_descriptor(),
            fetch_context=_ctx(),
        )
