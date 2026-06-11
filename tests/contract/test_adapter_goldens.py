"""Frozen golden-reference mapping outputs — CI fails on any unreviewed diff (ADP-R14/TST-R2).

For each curated real-shaped ASBO input fixture there is a COMMITTED frozen
expected-GBO output under ``tests/contract/goldens/``. The test serializes the full
mapped candidate list (every payload field, lineage versions, content_hash) and
asserts byte-stable equality with the frozen file — a mapping change that moves ANY
field fails CI until the golden is explicitly regenerated and reviewed in the diff.

Regeneration is deliberately out-of-band: run with ``WATTWISE_UPDATE_GOLDENS=1`` to
rewrite the frozen files, then review the diff. CI never sets that variable, so an
un-reviewed mapping drift can NOT pass (the ADP-R14 fail-on-diff gate).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import pytest

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import SourceKind
from wattwise_core.ingestion.adapters.file_upload import FileUploadAdapter
from wattwise_core.ingestion.adapters.file_upload import decode as decode_file
from wattwise_core.ingestion.adapters.intervals_icu import (
    ActivityWithStreams,
    IntervalsActivityAsbo,
    IntervalsIcuAdapter,
    IntervalsStreamAsbo,
    IntervalsWellnessAsbo,
)
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef

pytestmark = pytest.mark.golden

_FIXTURES = Path(__file__).parent / "fixtures"
_GOLDENS = Path(__file__).parent / "goldens"
_FETCHED_AT = _dt.datetime(2026, 6, 6, 12, 0, tzinfo=_dt.UTC)
_UPDATE = os.environ.get("WATTWISE_UPDATE_GOLDENS") == "1"


def _ctx() -> FetchContext:
    return FetchContext(ingest_run_id="run-1", fetched_at=_FETCHED_AT, connection_id="conn-1")


def _intervals_activity_asbo() -> ActivityWithStreams:
    return ActivityWithStreams(
        activity=IntervalsActivityAsbo.model_validate(
            json.loads((_FIXTURES / "intervals" / "synthetic_cycling_activity.json").read_text())
        ),
        streams=[
            IntervalsStreamAsbo.model_validate(s)
            for s in json.loads(
                (_FIXTURES / "intervals" / "synthetic_cycling_streams.json").read_text()
            )
        ],
    )


def _intervals_wellness_asbo() -> IntervalsWellnessAsbo:
    rows = json.loads((_FIXTURES / "intervals" / "wellness.json").read_text())
    return IntervalsWellnessAsbo.model_validate(rows[0])


_CASES: dict[str, tuple[Any, Any, str]] = {
    # golden name -> (adapter factory, asbo factory, source_key)
    "intervals_activity": (IntervalsIcuAdapter, _intervals_activity_asbo, "intervals_icu"),
    "intervals_wellness": (IntervalsIcuAdapter, _intervals_wellness_asbo, "intervals_icu"),
    "file_ride_fit": (
        FileUploadAdapter,
        lambda: decode_file(
            (_FIXTURES / "file_upload" / "ride.fit").read_bytes(), filename="ride.fit"
        ),
        "file_import",
    ),
    "file_ride_gpx": (
        FileUploadAdapter,
        lambda: decode_file(
            (_FIXTURES / "file_upload" / "ride.gpx").read_bytes(), filename="ride.gpx"
        ),
        "file_import",
    ),
    "file_ride_tcx": (
        FileUploadAdapter,
        lambda: decode_file(
            (_FIXTURES / "file_upload" / "ride.tcx").read_bytes(), filename="ride.tcx"
        ),
        "file_import",
    ),
}


def _serialize(cands: list[GboCandidate]) -> str:
    """A deterministic, reviewable JSON rendering of the full mapped output."""
    rows = [
        {
            "gbo_type": c.gbo_type,
            "source_native_id": c.source_native_id,
            "content_hash": c.content_hash,
            "observed_at": str(c.observed_at),
            "confidence": c.confidence,
            "trust_tier": str(c.trust_tier),
            "untrusted_content": c.untrusted_content,
            "adapter_version": c.adapter_version,
            "mapping_version": c.mapping_version,
            "payload": c.payload,
        }
        for c in cands
    ]
    return json.dumps(rows, sort_keys=True, default=str, indent=1) + "\n"


@pytest.mark.parametrize("name", sorted(_CASES))
def test_mapping_matches_frozen_golden(name: str) -> None:
    """ADP-R14: the full mapped GBO output equals the committed frozen golden, byte-stable."""
    adapter_factory, asbo_factory, source_key = _CASES[name]
    adapter = adapter_factory()
    ref = SourceDescriptorRef("sd-golden", source_key, SourceKind.OAUTH_API)
    rendered = _serialize(adapter.map(asbo_factory(), ref, _ctx()))
    golden = _GOLDENS / f"{name}.golden.json"
    if _UPDATE:
        _GOLDENS.mkdir(exist_ok=True)
        golden.write_text(rendered)
    assert golden.exists(), (
        f"missing frozen golden {golden.name}; generate with WATTWISE_UPDATE_GOLDENS=1 and review"
    )
    assert rendered == golden.read_text(), (
        f"mapping output drifted from frozen golden {golden.name}; if intended, regenerate with "
        "WATTWISE_UPDATE_GOLDENS=1 and review the diff (ADP-R14)"
    )
