"""Deterministic fact-sheet rendering for the entailment gate (issue #10 Phase 2).

The decorrelated entailment verifier (proposed GROUND-R11) checks each published sentence
against the canonical facts behind it — and those facts MUST be rendered by CODE, never by
a model, or the second gate would inherit the first gate's failure mode. This module owns
that rendering: the resolved canonical metric snapshots (the same ``(metric, as_of) ->
value`` mapping the numeric verifier read, GROUND-R7 verbatim) and the turn's retrieved
canonical records are serialized into a small, stable, plain-text fact sheet. The verifier
can only VETO against this sheet; nothing here can add a sayable value (fail-closed).

Everything is a pure, synchronous function of its inputs (GRAPH-R4): deterministic
ordering, bounded size, no service or model call.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

#: Conservative size bounds so the verifier input stays small and deterministic. A sheet
#: that would overflow is truncated WITH a marker — an absent fact can only cause a veto
#: (fail-closed), never an unverified pass.
_MAX_SHEET_CHARS = 4000
_MAX_RECORD_CHARS = 240
_TRUNCATION_MARKER = "[fact sheet truncated]"


def render_fact_sheet(
    snapshots: Mapping[tuple[str, str | None], float | None],
    retrieved: Mapping[str, Any] | None = None,
    *,
    request_text: str | None = None,
    max_chars: int = _MAX_SHEET_CHARS,
) -> str:
    """Render the canonical facts a sentence may be checked against (GROUND-R11).

    ``snapshots`` are the pre-resolved canonical values the numeric gate verified against
    (verbatim, GROUND-R7); ``retrieved`` are the turn's gathered canonical records (compact,
    deterministic JSON per record so series/trend statements have backing); ``request_text``
    is the athlete's own request — included so a sentence restating the USER's constraint
    (the request-echo path) is entailed by the sheet instead of vetoed for citing a value
    canonical analytics never computed. Deterministic ordering + bounded size.
    """
    lines: list[str] = []
    for (metric, as_of), value in sorted(
        snapshots.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
    ):
        if value is None:
            continue
        when = f"as of {as_of}" if as_of else "latest value"
        lines.append(f"canonical metric {metric} ({when}): {value:g}")
    for key, record in sorted((retrieved or {}).items()):
        rendered = _render_record(record)
        if rendered:
            lines.append(f"canonical record {key}: {rendered}")
    if request_text and request_text.strip():
        lines.append(f"the athlete's request says: {request_text.strip()}")
    sheet = "\n".join(lines)
    if len(sheet) > max_chars:
        sheet = sheet[: max_chars - len(_TRUNCATION_MARKER) - 1].rstrip()
        sheet = f"{sheet}\n{_TRUNCATION_MARKER}"
    return sheet


def _render_record(record: Any) -> str:
    """One retrieved record as compact, deterministic JSON (bounded; unserializable -> '')."""
    try:
        rendered = json.dumps(record, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return ""
    if len(rendered) > _MAX_RECORD_CHARS:
        rendered = rendered[:_MAX_RECORD_CHARS] + "…"
    return rendered


__all__ = ["render_fact_sheet"]
