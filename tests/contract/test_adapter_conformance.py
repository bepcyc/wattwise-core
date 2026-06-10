"""Generic adapter conformance suite — every registered adapter, purely from fixtures (ONB-R3).

Parametrized over the adapters discovered through the REAL entry-point registry
(``load_registry``), never a hand-picked list: adding a source means adding its
fixture provider below and making THIS suite green for it (ACC-1). Per adapter it
asserts, from recorded fixtures only (TST-R1, no network):

* the capability descriptor is present and valid (ADP-R1/ADP-R2/ONB-R2);
* ``map`` is pure and stable — two maps of the same fixture are byte-identical
  including ``content_hash`` (ADP-R10; the frozen-output gate is
  ``test_adapter_goldens.py``/ADP-R14);
* every mapped candidate's GBO type is within the DECLARED set (ADP-R3);
* no source-shaped key or raw-JSON blob leaks into a canonical payload (ING-R9);
* a garbage ASBO maps to NO candidate — nothing fabricated (ADP-R12);
* import boundaries hold: the adapter module imports no analytics/api/agent layer
  (ADP-R16);
* for the discover-capable adapter: discovery honors the watermark (ADP-R6),
  surfaces a pagination cursor (ADP-R7), and fetch fails CLOSED with a typed
  ``schema_mismatch`` on a malformed payload (CLI-R2).

Partial-discovery/fetch → range-precise gaps (ING-GAP-R5) is engine behavior and is
covered by ``tests/integration/test_sync_discover.py``.
"""

from __future__ import annotations

import ast
import datetime as _dt
import importlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from wattwise_core.config import load_settings
from wattwise_core.domain.enums import GboType, SourceKind
from wattwise_core.ingestion.adapters.file_upload import decode as decode_file
from wattwise_core.ingestion.adapters.intervals_icu import (
    ActivityWithStreams,
    IntervalsActivityAsbo,
    IntervalsIcuAdapter,
    IntervalsStreamAsbo,
)
from wattwise_core.ingestion.base import (
    FetchContext,
    FetchError,
    FetchErrorKind,
    SourceAdapter,
    SourceDescriptorRef,
)
from wattwise_core.ingestion.capability import AuthContext, validate_capability
from wattwise_core.ingestion.registry import load_registry

pytestmark = pytest.mark.contract

_FIXTURES = Path(__file__).parent / "fixtures"
_BASE = "https://intervals.icu"
_FETCHED_AT = _dt.datetime(2026, 6, 6, 12, 0, tzinfo=_dt.UTC)

#: Per-source fixture provider: source_key -> a real-shaped ASBO loaded from the
#: recorded corpus. Onboarding a new source = registering its provider here (ONB-R3);
#: the suite itself never branches on a source name beyond this data table.
_ASBO_PROVIDERS: dict[str, Any] = {
    "intervals_icu": lambda: ActivityWithStreams(
        activity=IntervalsActivityAsbo.model_validate(
            json.loads((_FIXTURES / "intervals" / "synthetic_cycling_activity.json").read_text())
        ),
        streams=[
            IntervalsStreamAsbo.model_validate(s)
            for s in json.loads(
                (_FIXTURES / "intervals" / "synthetic_cycling_streams.json").read_text()
            )
        ],
    ),
    "file_import": lambda: decode_file(
        (_FIXTURES / "file_upload" / "ride.gpx").read_bytes(), filename="ride.gpx"
    ),
}

_REGISTRY = load_registry()
_SOURCE_KEYS = sorted(_REGISTRY.source_keys())


def _adapter(source_key: str) -> SourceAdapter:
    return _REGISTRY.get(source_key)


def _ref(source_key: str) -> SourceDescriptorRef:
    return SourceDescriptorRef("sd-uuid-1", source_key, SourceKind.OAUTH_API)


def _ctx() -> FetchContext:
    return FetchContext(ingest_run_id="run-1", fetched_at=_FETCHED_AT, connection_id="conn-1")


def test_every_registered_adapter_has_a_fixture_provider() -> None:
    """ONB-R3 gate: a registered source without a fixture provider fails the suite."""
    assert set(_SOURCE_KEYS) == set(_ASBO_PROVIDERS), (
        "every registered adapter MUST register a fixture provider in this suite"
    )


@pytest.mark.parametrize("source_key", _SOURCE_KEYS)
def test_capability_descriptor_is_valid(source_key: str) -> None:
    """ADP-R1/R2: the registered adapter exposes a valid machine-readable descriptor."""
    cap = validate_capability(_adapter(source_key))
    assert cap.source_key == source_key
    assert cap.supported_gbo_types


@pytest.mark.parametrize("source_key", _SOURCE_KEYS)
def test_map_is_pure_and_stable(source_key: str) -> None:
    """ADP-R10: mapping the same fixture twice is byte-identical (incl. content_hash)."""
    adapter = _adapter(source_key)
    asbo = _ASBO_PROVIDERS[source_key]()
    first = adapter.map(asbo, _ref(source_key), _ctx())
    second = adapter.map(asbo, _ref(source_key), _ctx())
    assert first and len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.content_hash == b.content_hash
        assert a.payload == b.payload


@pytest.mark.parametrize("source_key", _SOURCE_KEYS)
def test_mapped_types_are_declared(source_key: str) -> None:
    """ADP-R3 declaration honored: every mapped candidate's type is in the declared set."""
    adapter = _adapter(source_key)
    declared = {t.value for t in adapter.capability.supported_gbo_types}
    for cand in adapter.map(_ASBO_PROVIDERS[source_key](), _ref(source_key), _ctx()):
        assert cand.gbo_type in declared


@pytest.mark.parametrize("source_key", _SOURCE_KEYS)
def test_no_source_shaped_or_blob_payload(source_key: str) -> None:
    """ING-R9/MAP-R2: canonical payloads carry named typed fields, never source blobs."""
    adapter = _adapter(source_key)
    for cand in adapter.map(_ASBO_PROVIDERS[source_key](), _ref(source_key), _ctx()):
        for key, value in cand.payload.items():
            assert not key.startswith(("icu_", "source", "garmin", "strava"))
            if isinstance(value, dict):
                # the ONLY structured payload members are typed streams/laps shapes
                assert key in ("streams",), f"raw object passed through under {key!r}"


@pytest.mark.parametrize("source_key", _SOURCE_KEYS)
def test_garbage_asbo_maps_to_nothing(source_key: str) -> None:
    """ADP-R12: an unrecognizable ASBO yields NO candidate — never a fabricated record."""
    assert _adapter(source_key).map(object(), _ref(source_key), _ctx()) == []


@pytest.mark.parametrize("source_key", _SOURCE_KEYS)
def test_import_boundaries_hold(source_key: str) -> None:
    """ADP-R16: the adapter module imports no analytics/api/agent/persistence layer."""
    module = importlib.import_module(type(_adapter(source_key)).__module__)
    tree = ast.parse(Path(module.__file__).read_text())  # type: ignore[arg-type]
    forbidden = (
        "wattwise_core.analytics",
        "wattwise_core.api",
        "wattwise_core.agent",
        "wattwise_core.persistence",
    )
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            assert not name.startswith(forbidden), f"forbidden import {name!r}"


# ------------------------------------------------------------------ discover-capable


def _discover_adapter() -> IntervalsIcuAdapter:
    """The shipped discover-capable adapter with a 1-ref page size (fixture-driven)."""
    return IntervalsIcuAdapter(
        settings=load_settings(
            app__environment="development",
            database_dsn="sqlite+aiosqlite:///:memory:",
            token_signing_key="k" * 32,
            adapters__intervals_icu__discover_page_size=1,
        )
    )


def _mock_listings() -> None:
    respx.get(url__regex=rf"{_BASE}/api/v1/athlete/.*/activities.*").mock(
        return_value=httpx.Response(
            200, json=json.loads((_FIXTURES / "intervals" / "activities_list.json").read_text())
        )
    )
    respx.get(url__regex=rf"{_BASE}/api/v1/athlete/.*/wellness.*").mock(
        return_value=httpx.Response(
            200, json=json.loads((_FIXTURES / "intervals" / "wellness.json").read_text())
        )
    )


@respx.mock
async def test_discover_surfaces_cursor_and_pages_to_completion() -> None:
    """ADP-R5/R7: discovery yields lightweight refs page-by-page with a ``next_cursor``."""
    _mock_listings()
    adapter = _discover_adapter()
    ctx = AuthContext(athlete_native_id="i00000", api_key="test-key")
    window = type("W", (), {"oldest": "2026-05-01", "newest": "2026-06-01"})()
    refs = []
    cursor: str | None = None
    pages = 0
    while True:
        page = await adapter.discover(ctx, window, cursor=cursor, since_watermark=None)
        refs.extend(page.refs)
        pages += 1
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert pages > 1  # the cursor was actually surfaced and followed (ADP-R7)
    assert any(r.gbo_type is GboType.ACTIVITY for r in refs)
    assert any(r.gbo_type is GboType.DAILY_WELLNESS for r in refs)
    for ref in refs:
        assert ref.source_native_id  # lightweight: native id + type (+ hint) only


@respx.mock
async def test_discover_honors_watermark() -> None:
    """ADP-R6: refs already current per the watermark are NOT yielded again."""
    _mock_listings()
    adapter = _discover_adapter()
    ctx = AuthContext(athlete_native_id="i00000", api_key="test-key")
    window = type("W", (), {"oldest": "2026-05-01", "newest": "2026-06-01"})()
    future = _dt.datetime(2027, 1, 1, tzinfo=_dt.UTC)
    page = await adapter.discover(ctx, window, cursor=None, since_watermark=future)
    activity_refs = [r for r in page.refs if r.gbo_type is GboType.ACTIVITY]
    assert activity_refs == []  # every hinted activity ref is current -> skipped


@respx.mock
async def test_fetch_ref_schema_mismatch_fails_closed() -> None:
    """CLI-R2: a malformed source payload raises typed ``schema_mismatch``, never a GBO."""
    respx.get(url__regex=rf"{_BASE}/api/v1/activity/.*/streams.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=rf"{_BASE}/api/v1/activity/.*").mock(
        return_value=httpx.Response(
            200, json=json.loads((_FIXTURES / "intervals" / "malformed_activity.json").read_text())
        )
    )
    adapter = _discover_adapter()
    ctx = AuthContext(athlete_native_id="i00000", api_key="test-key")
    ref = type(
        "R", (), {"source_native_id": "i111", "gbo_type": GboType.ACTIVITY, "last_modified": None}
    )()
    with pytest.raises(FetchError) as err:
        await adapter.fetch_ref(ctx, ref)  # type: ignore[arg-type]
    assert err.value.kind is FetchErrorKind.SCHEMA_MISMATCH
