"""Intervals.icu discover/fetch phase helpers (ADP-R5/R6/R7) — client-driven, stateless.

A focused split of the adapter's five-phase sync side (QUAL-R9): cursor-paginated
discovery over the windowed activity + wellness listings, watermark-honoring ref
filtering, and per-ref fetch. Every function takes the typed client explicitly and
holds no state, so the adapter stays a stateless registry singleton and the phases
are fully exercisable offline against recorded fixtures (ADP-R17/TST-R1).

Cursor format (ADP-R7): ``"act:<offset>"`` pages the activity refs, then
``"well:<offset>"`` pages the wellness refs; ``next_cursor=None`` means discovery is
complete. The cursor is an opaque resume token to the engine — a mid-pagination
failure is reported as a typed gap from exactly the broken cursor (ING-GAP-R5).
"""

from __future__ import annotations

import datetime as _dt

from wattwise_core.domain.enums import GboType
from wattwise_core.ingestion.capability import DiscoveryPage, DiscoveryRef

_ACT = "act"
_WELL = "well"


def parse_cursor(cursor: str | None) -> tuple[str, int]:
    """Decode an opaque discover cursor to ``(stage, offset)`` (``None`` = the start)."""
    if cursor is None:
        return (_ACT, 0)
    stage, _, raw = cursor.partition(":")
    if stage not in (_ACT, _WELL):
        raise ValueError(f"unknown discover cursor stage {stage!r}")
    return (stage, int(raw or 0))


def _parse_hint(value: str | None) -> _dt.datetime | None:
    """Parse a source last-modified hint to a UTC instant (``None`` when unusable)."""
    if not value:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(_dt.UTC)


def _current_per_watermark(
    last_modified: _dt.datetime | None, since_watermark: _dt.datetime | None
) -> bool:
    """True when a ref is already known current per the watermark (ADP-R6 skip).

    A ref is skipped ONLY when its last-modified hint exists and does not indicate a
    change past the watermark; a hint-less ref is always yielded (we cannot prove it
    current — fail-closed toward re-checking, never toward skipping data).
    """
    return (
        since_watermark is not None
        and last_modified is not None
        and last_modified <= since_watermark
    )


def page_of(
    refs: list[DiscoveryRef], offset: int, page_size: int, *, stage: str, last_stage: bool
) -> DiscoveryPage:
    """Slice one cursor page out of a stage's full ref list, surfacing ``next_cursor``."""
    window = refs[offset : offset + page_size]
    end = offset + page_size
    if end < len(refs):
        next_cursor: str | None = f"{stage}:{end}"
    elif last_stage:
        next_cursor = None
    else:
        next_cursor = f"{_WELL}:0"
    return DiscoveryPage(refs=tuple(window), next_cursor=next_cursor)


def activity_refs(
    raw: list[dict[str, object]], since_watermark: _dt.datetime | None
) -> list[DiscoveryRef]:
    """Windowed activity summaries -> lightweight refs, watermark-filtered (ADP-R5/R6).

    The listing's ``start_date`` is the last-modified HINT available on the summary;
    refs already current per the watermark are NOT yielded (ADP-R6). The listing is
    consumed oldest-first as the source returns it (the declared discovery order).
    """
    refs: list[DiscoveryRef] = []
    for row in raw:
        native_id = row.get("id")
        if native_id is None:
            continue
        hint = _parse_hint(str(row.get("start_date")) if row.get("start_date") else None)
        if _current_per_watermark(hint, since_watermark):
            continue
        refs.append(
            DiscoveryRef(
                source_native_id=str(native_id),
                gbo_type=GboType.ACTIVITY,
                last_modified=hint,
            )
        )
    return refs


def wellness_refs(ids: list[str]) -> list[DiscoveryRef]:
    """Windowed wellness day-ids -> lightweight refs (no last-modified hint exists).

    Wellness rows carry no usable change hint, so they are always yielded for the
    window — the watermark cannot prove them current (ADP-R6's "unless changed"
    cannot be evaluated), and the engine's content-hash landing dedups idempotently.
    """
    return [
        DiscoveryRef(source_native_id=day, gbo_type=GboType.DAILY_WELLNESS, last_modified=None)
        for day in ids
    ]


__all__ = [
    "activity_refs",
    "page_of",
    "parse_cursor",
    "wellness_refs",
]
