"""File-upload source adapter — FIT/GPX/TCX/PWX (CLI-R13/CLI-R14, FIL-R*, LIN-R1.1, MAP-R*).

The single OSS file-upload importer. Every connectionless activity-file upload — a
Garmin ``.fit``, a platform-exported ``.gpx``/``.tcx``/``.pwx``, INCLUDING the compliant
Strava-export path (CLI-R14: Strava is file-upload-only, never a direct API) — lands
under ONE built-in ``source_descriptor`` (``source_key = "file_import"``, LIN-R1.1);
no per-platform descriptor is inferred from file contents, so source-invisibility
(Principle A) holds and a Strava export leaves no Strava-named lineage.

Two strictly separated layers:

* :func:`decode` — impure I/O: verbatim bytes -> a typed :class:`ActivityAsbo` via the
  per-format decoders (FIT: ``garmin-fit-sdk`` + ``fitdecode`` fallback; GPX: ``gpxpy``;
  TCX/PWX: ``lxml``). Malformed input raises a TYPED :class:`FileDecodeError`, never a bare
  crash or a wrong-but-plausible record (TIER-R5 fuzz / CLI-R2).
* :meth:`FileUploadAdapter.map` — **pure and deterministic** (MAP-R1): no clock, no
  randomness, no network. It turns one :class:`ActivityAsbo` into canonical
  :class:`~wattwise_core.domain.candidate.GboCandidate` records carrying ONLY canonical
  field names (MAP-R2), SI units (MAP-R3), canonical sport codes (MAP-R4), real gaps as
  ``None`` never ``0`` (MAP-R5), and free text tagged untrusted (MAP-R7). ``fetched_at``
  comes from :class:`FetchContext`; the map never reads the wall clock. A byte-identical
  re-decode + re-map is deterministic (GBO-AC-1, FIL-R3/FIL-R5).

``source_native_id`` is derived per LIN-R1.1 from the file's OWN immutable identity:
FIT = the ``file_id``-message fingerprint; GPX/TCX/PWX = the per-format start/elapsed/extent
fingerprint; fallback = the ``content_hash`` over the verbatim bytes. The adapter depends
only on its decoders + canonical models + lineage/enums (ADP-R16) and is fully
exercisable offline (ADP-R17, TST-R1).
"""

from __future__ import annotations

import datetime as _dt
import gzip
from typing import Any, ClassVar, Final

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import ActivityFileFormat, AuthArchetype, Fidelity, SourceKind
from wattwise_core.ingestion.adapters import _map_activity as _m
from wattwise_core.ingestion.adapters._asbo import (
    ActivityAsbo,
    AsboLap,
    AsboRecord,
    FileDecodeError,
)
from wattwise_core.ingestion.adapters._decode_fit import decode_fit
from wattwise_core.ingestion.adapters._decode_gpx import decode_gpx
from wattwise_core.ingestion.adapters._decode_pwx import decode_pwx
from wattwise_core.ingestion.adapters._decode_tcx import decode_tcx
from wattwise_core.ingestion.base import (
    FetchContext,
    FileImportError,
    SourceDescriptorRef,
    UploadDecode,
)
from wattwise_core.storage import content_hash

# The single built-in file-upload descriptor slug (LIN-R1.1).
FILE_IMPORT_SOURCE_KEY: Final = "file_import"


def detect_format(filename: str | None, data: bytes) -> str:
    """Detect the file format from name + magic bytes (closed enum: fit/gpx/tcx/pwx).

    Raises :class:`FileDecodeError` for an unrecognized format so an unsupported
    upload fails closed (never a silent best-guess parse, CLI-R2/TIER-R5).
    """
    suffix = "" if filename is None else filename.lower().rsplit(".", 1)[-1]
    head = data[:512].lstrip()
    if suffix == "fit" or _looks_like_fit(data):
        return "fit"
    if suffix == "tcx" or b"TrainingCenterDatabase" in head:
        return "tcx"
    if suffix == "pwx" or b"<pwx" in head:
        return "pwx"
    if suffix == "gpx" or (head.startswith((b"<?xml", b"<gpx")) and b"<gpx" in data[:2048]):
        return "gpx"
    raise FileDecodeError("unrecognized activity-file format (expected FIT/GPX/TCX/PWX)")


def _looks_like_fit(data: bytes) -> bool:
    """FIT header: byte 0 = header size (12 or 14), bytes 8-11 == b".FIT"."""
    return len(data) >= 12 and data[0] in (12, 14) and data[8:12] == b".FIT"


def decode(data: bytes, *, filename: str | None = None) -> ActivityAsbo:
    """Decode verbatim upload bytes into a typed :class:`ActivityAsbo` (impure I/O).

    This is the impure layer kept OUT of the pure :meth:`FileUploadAdapter.map`
    (MAP-R1). Every malformed/corrupt/empty input raises a typed
    :class:`FileDecodeError` (TIER-R5), never a bare crash.
    """
    if not isinstance(data, bytes | bytearray):  # defensive: fuzz feeds odd inputs
        raise FileDecodeError("decode expects bytes")
    blob = bytes(data)
    name = filename
    if blob[:2] == b"\x1f\x8b":  # gzip magic — a compressed export (e.g. `.fit.gz`)
        try:
            blob = gzip.decompress(blob)
        except (OSError, EOFError) as exc:
            raise FileDecodeError("could not decompress the gzipped activity file") from exc
        if name and name.lower().endswith(".gz"):
            name = name[: -len(".gz")]  # detect the inner format on the unwrapped name
    fmt = detect_format(name, blob)
    if fmt == "fit":
        return decode_fit(blob)
    if fmt == "gpx":
        return decode_gpx(blob)
    if fmt == "pwx":
        return decode_pwx(blob)
    return decode_tcx(blob)


def native_id(asbo: ActivityAsbo, raw_bytes: bytes) -> str:
    """Derive the candidate ``source_native_id`` per LIN-R1.1.

    Prefers the file's OWN immutable fingerprint (FIT file_id; GPX/TCX
    start/elapsed/extent); falls back to the ``content_hash`` over the verbatim
    bytes when no stable native identity exists. Deterministic for a byte-identical
    re-upload (FIL-R5, GBO-AC-1).
    """
    if asbo.native_fingerprint:
        return asbo.native_fingerprint
    return content_hash(raw_bytes)


class FileUploadAdapter:
    """The OSS file-upload adapter (ADP-R*; satisfies the ``SourceAdapter`` Protocol).

    Identity metadata is declared as class attributes (ADP-R1). :meth:`map` is pure
    (MAP-R1); :func:`decode` (impure I/O) is invoked by the sync engine OUTSIDE
    ``map``. The ``file_upload`` auth archetype is connectionless (LIN-R1.1).
    """

    source_key: ClassVar[str] = FILE_IMPORT_SOURCE_KEY
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.FILE_UPLOAD
    kind: ClassVar[SourceKind] = SourceKind.FILE_UPLOAD
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"

    def map(
        self,
        asbo: Any,
        source_descriptor: SourceDescriptorRef,
        fetch_context: FetchContext,
    ) -> list[GboCandidate]:
        """Map one decoded :class:`ActivityAsbo` into canonical candidates (MAP-R1).

        Emits exactly ONE ``activity`` candidate per training session. A required
        canonical field absent at the source (no usable ``start_time``) yields no
        candidate rather than a fabricated value (ING-R3/MAP-R5). The map is pure,
        deterministic, and side-effect-free.
        """
        if not isinstance(asbo, ActivityAsbo):
            return []
        start = _m.start_time(asbo)
        if start is None:
            return []  # required canonical field absent -> no fabricated candidate
        return self._build_activity(asbo, start, source_descriptor, fetch_context, None)

    def decode_upload(
        self,
        raw_bytes: bytes,
        *,
        filename: str | None,
        source_descriptor: SourceDescriptorRef,
        fetch_context: FetchContext,
    ) -> UploadDecode:
        """Decode + pure-map one uploaded file (FIL-R1); a bad file → :class:`FileImportError`.

        The file-import seam (:class:`~wattwise_core.ingestion.base.FileImportAdapter`) a
        source-blind consumer drives: it runs the impure :func:`decode` then the pure
        :meth:`map_upload`, wrapping the typed decoder failure in the neutral
        :class:`FileImportError` so the consumer never imports a source-specific error
        (ARCH-R22). Reports the verbatim original's format for tier-1 capture (FIL-R1).
        """
        try:
            asbo = decode(raw_bytes, filename=filename)
            file_format = ActivityFileFormat(detect_format(filename, raw_bytes))
        except FileDecodeError as exc:
            raise FileImportError("could not decode the uploaded activity file") from exc
        candidates = self.map_upload(raw_bytes, asbo, source_descriptor, fetch_context)
        return UploadDecode(candidates=candidates, file_format=file_format)

    def map_upload(
        self,
        raw_bytes: bytes,
        asbo: ActivityAsbo,
        source_descriptor: SourceDescriptorRef,
        fetch_context: FetchContext,
    ) -> list[GboCandidate]:
        """Pure map that also supplies the verbatim bytes for the LIN-R1.1 fallback id.

        Use when the fingerprint may be absent (e.g. a GPX/TCX with no timestamps):
        the ``source_native_id`` falls back to ``content_hash(raw_bytes)``. Still
        pure/deterministic — ``raw_bytes`` is an input, not read from disk here.
        """
        start = _m.start_time(asbo)
        if start is None:
            return []
        native = native_id(asbo, raw_bytes)
        return self._build_activity(asbo, start, source_descriptor, fetch_context, native)

    def _build_activity(
        self,
        asbo: ActivityAsbo,
        start: _dt.datetime,
        descriptor: SourceDescriptorRef,
        ctx: FetchContext,
        source_native_id: str | None,
    ) -> list[GboCandidate]:
        streams = _m.build_streams(asbo)
        laps = _m.build_laps(asbo, start)
        payload = _m.activity_payload(asbo, start, streams, laps)
        canonical_hash = _m.stable_hash(payload)
        # LIN-R1.1: prefer the file's own fingerprint; if absent (and no verbatim
        # bytes were supplied via map_upload) fall back to the payload hash — stable
        # per session, never the empty-bytes constant.
        native = source_native_id or asbo.native_fingerprint or canonical_hash
        has_real_stream = _m.has_per_sample_stream(streams)
        untrusted = _m.has_free_text(asbo)
        return [
            GboCandidate(
                gbo_type="activity",
                source_descriptor_id=descriptor.source_descriptor_id,
                source_native_id=native,
                content_hash=canonical_hash,
                payload=payload,
                observed_at=start,
                fetched_at=ctx.fetched_at,
                confidence=1.0,
                trust_tier=(Fidelity.RAW_STREAM if has_real_stream else Fidelity.SUMMARY_ONLY),
                untrusted_content=untrusted,
                connection_id=ctx.connection_id,
                adapter_version=self.adapter_version,
                mapping_version=self.mapping_version,
            )
        ]


__all__ = [
    "FILE_IMPORT_SOURCE_KEY",
    "ActivityAsbo",
    "AsboLap",
    "AsboRecord",
    "FileDecodeError",
    "FileUploadAdapter",
    "decode",
    "detect_format",
    "native_id",
]
