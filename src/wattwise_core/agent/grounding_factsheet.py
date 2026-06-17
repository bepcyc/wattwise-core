"""Deterministic fact-sheet rendering: the entailment gate AND the compose context (COMPOSE-R1).

The decorrelated entailment verifier (proposed GROUND-R11) checks each published sentence
against the canonical facts behind it — and those facts MUST be rendered by CODE, never by
a model, or the second gate would inherit the first gate's failure mode. This module owns
that rendering: the resolved canonical metric snapshots (the same ``(metric, as_of) ->
value`` mapping the numeric verifier read, GROUND-R7 verbatim) and the turn's retrieved
canonical records are serialized into a small, stable, plain-text fact sheet. The verifier
can only VETO against this sheet; nothing here can add a sayable value (fail-closed).

It is ALSO the shared seam for the compose-context capability fact sheet (COMPOSE-R1,
:func:`render_capability_factsheet`): the model answers FROM a code-rendered, current-values-
first summary of each gathered capability — never a raw object repr — so the one claim that
can ground leads the salience. The grounder still verifies every claim; the sheet certifies
nothing.

Everything is a pure, synchronous function of its inputs (GRAPH-R4): deterministic
ordering, bounded size, no service or model call.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from wattwise_core.analytics.np_if_tss import LoadMetricsBundle
from wattwise_core.analytics.pmc import PmcDay
from wattwise_core.analytics.result import is_computed

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


# --- compose-context capability fact sheet (COMPOSE-R1) ---
#
# The compose node renders each GATHERED capability as a CODE-rendered, athlete-relevant
# summary whose LEADING lines state the CURRENT canonical values (with a short trend), and
# whose day-by-day series follows as compact "date/position: value" supporting detail —
# NEVER a raw Python repr of the internal result objects (a long warm-up-zero series dumped
# raw makes the stale zeros the dominant signal and steers a reasoning model away from the
# one claim that can ground). This is presentation salience only: the §7 grounder still
# verifies every claim the model makes (the sheet certifies nothing).

#: Bound the per-capability day series so a long history stays compact and deterministic;
#: the LEADING current-value lines are never trimmed (they are the answer-bearing salience).
_MAX_SERIES_DAYS = 14
#: Athlete-facing names for the PMC scalars in the leading lines (no internal codes leak into
#: the prose the model reads — VOICE-R2 is enforced downstream, but the sheet keeps the plain
#: word beside the canonical code the grounder matches on).
_PMC_LEAD = (
    ("fitness", "ctl"),
    ("fatigue", "atl"),
    ("form", "tsb"),
)


def render_capability_factsheet(
    retrieved: Mapping[str, Any], activity_refs: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Render each gathered capability as a current-values-first fact sheet (COMPOSE-R1/-R1a).

    Returns a ``{capability_key: rendered_text}`` mapping the compose context envelopes per
    capability. Each value LEADS with the capability's CURRENT canonical scalar(s) and a short
    trend, then (where a series exists) a compact bounded ``position: value`` tail — never a
    raw object repr. A capability whose record carries no computed value renders an honest
    "no current value available" line (fail-closed salience, never a fabricated number).

    ``activity_refs`` maps a capability key -> the canonical ``activity_id`` the planner
    requested it for (PLAN-R3). When present for an activity-scoped capability, the rendered
    sheet LEADS with ``for activity <id>:`` so the model can author a per-ride claim the §7
    grounder binds to the RIGHT activity (GROUND-R7 per-ride ``activity_tss``, COMPOSE-R1a) —
    rather than a date-keyed snapshot. LIMITATION (v1): one activity per capability key — the
    gather map is capability-keyed, so multiple distinct per-ride requests for the SAME
    capability in one turn collapse to whatever the gather kept (a salience limit only; the
    grounder still resolves each per-ride claim independently from canonical analytics).
    """
    refs = activity_refs or {}
    out: dict[str, str] = {}
    for key, record in retrieved.items():
        out[key] = _render_capability(record, refs.get(key))
    return out


def _render_capability(record: Any, activity_id: str | None = None) -> str:
    """One gathered capability rendered current-values-first (COMPOSE-R1/-R1a).

    When ``activity_id`` is given (an activity-scoped capability, COMPOSE-R1a) the body is
    prefixed with ``for activity <id>:`` so the model can author a per-ride claim keyed to
    that activity; a falsy id renders the body alone (fail-closed, never a guessed id).
    """
    body = _render_capability_body(record)
    return f"for activity {activity_id}: {body}" if activity_id else body


def _render_capability_body(record: Any) -> str:
    """The capability body, dispatched by record shape (COMPOSE-R1)."""
    if _is_pmc_series(record):
        return _render_pmc_series(record)
    bundle = _render_load_bundle(record)
    if bundle is not None:
        return bundle
    rendered = _render_single_metric(record)
    if rendered is not None:
        return rendered
    if isinstance(record, Mapping):
        body = _render_record(record)
        return body or "no current value available"
    # Unknown shape (a typed coverage gap object, a scalar, …): render compactly, never a repr
    # dominated by warm-up internals.
    body = _render_record(record)
    return body or "no current value available"


def _is_pmc_series(record: Any) -> bool:
    """True iff ``record`` is the weekly-load PMC series (a sequence of PmcDay results)."""
    if isinstance(record, str) or not isinstance(record, Sequence):
        return False
    return any(is_computed(item) and isinstance(item.value, PmcDay) for item in record)


def _render_pmc_series(series: Sequence[Any]) -> str:
    """Render the PMC (CTL/ATL/form) series: current values + trend lead, compact tail.

    The LATEST computed day is the athlete's current value (the PMC grid carries forward to
    the reference day, PMC-R6); the trend compares it against the EARLIEST computed day so a
    warm-up-from-zero history reads as "rising from a low base" rather than as a wall of zeros.
    """
    computed = [
        item.value for item in series if is_computed(item) and isinstance(item.value, PmcDay)
    ]
    if not computed:
        return "no current fitness/fatigue/form values available"
    latest, earliest = computed[-1], computed[0]
    lead_parts: list[str] = []
    for plain, code in _PMC_LEAD:
        cur = float(getattr(latest, code))
        prev = float(getattr(earliest, code))
        lead_parts.append(f"current {plain} ({code}) {cur:g} ({_trend_word(prev, cur)})")
    lead = "; ".join(lead_parts)
    tail = _render_pmc_tail(computed)
    return f"{lead}\n{tail}" if tail else lead


def _render_pmc_tail(days: Sequence[PmcDay]) -> str:
    """The compact recent day-by-day tail (positional; bounded), never a repr.

    Labels are POSITIONAL relative to the window's LAST day ("latest", "1 day earlier", …) —
    never "today", because a gathered window need not end on the calendar today and a
    mislabeled date would steer the model toward a falsely-dated claim (GROUND-R7 honesty).
    """
    recent = days[-_MAX_SERIES_DAYS:]
    offset = len(recent) - 1
    lines: list[str] = []
    for i, day in enumerate(recent):
        label = "latest" if i == offset else f"{offset - i} day(s) earlier"
        lines.append(f"  {label}: fitness {day.ctl:g}, fatigue {day.atl:g}, form {day.tsb:g}")
    return "recent days (most recent last):\n" + "\n".join(lines)


def _trend_word(earliest: float, latest: float) -> str:
    """A short plain-language trend from the window's first to last computed value."""
    delta = latest - earliest
    if abs(delta) < 0.05:
        return "steady"
    direction = "rising" if delta > 0 else "falling"
    return f"{direction} from {earliest:g}"


def _render_load_bundle(record: Any) -> str | None:
    """Render a per-activity load-metrics bundle current-values-first (COMPOSE-R1a), or ``None``.

    The per-ride load family (``coggan`` -> ``MetricResult[LoadMetricsBundle]``) carries each
    figure as an INDEPENDENT nested ``MetricResult``; an Unavailable field is simply not stated
    (never a fabricated 0). The per-ride training-stress score is surfaced under its canonical
    grounding code ``activity_tss`` (GROUND-R7, ``MetricName.ACTIVITY_TSS``) — the model states
    it beside the plain words so the per-ride claim keys to the right resolver. Returns a string
    for ANY ``LoadMetricsBundle`` value (so the generic path never repr-dumps the dataclass,
    COMPOSE-R1) and ``None`` for every other record shape.
    """
    if not is_computed(record) or not isinstance(record.value, LoadMetricsBundle):
        return None
    bundle = record.value
    parts: list[str] = []
    for plain, code, field in (
        ("training stress score", "activity_tss", bundle.tss),
        ("intensity factor", "if", bundle.if_),
        ("training stress per hour", "tss_per_hour", bundle.tss_per_hour),
    ):
        if is_computed(field) and _is_real_number(field.value):
            parts.append(f"current {plain} ({code}) {float(field.value):g}")
    np_result = bundle.np
    if is_computed(np_result) and _is_real_number(np_result.value.np_w):
        parts.append(f"normalized power (np_w) {float(np_result.value.np_w):g}")
    return "; ".join(parts) if parts else "no current value available"


def _is_real_number(value: Any) -> bool:
    """True iff ``value`` is a real int/float (never a bool, which is an int subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _render_single_metric(record: Any) -> str | None:
    """Render a single computed metric record current-value-first, or ``None`` if not one.

    Reads the canonical scalar(s) off the Computed value VERBATIM (GROUND-R7); an Unavailable
    or unrecognised value-shape returns ``None`` so the caller falls through to the generic
    compact rendering. A field absent on the value object is simply not stated (never a 0).
    """
    if not is_computed(record):
        if _looks_unavailable(record):
            return "no current value available"
        return None
    value = record.value
    if isinstance(value, PmcDay):  # a single PMC day (defensive; weekly_load is the series)
        return (
            f"current fitness (ctl) {value.ctl:g}; fatigue (atl) {value.atl:g}; "
            f"form (tsb) {value.tsb:g}"
        )
    parts = _scalar_lines(value)
    if parts:
        return "; ".join(parts)
    return None


#: Canonical scalar fields read VERBATIM off a Computed metric value for the leading line.
#: Each tuple is ``(plain_label, canonical_code, attribute_name)``; an attribute the value
#: object does not carry is skipped (never a fabricated 0).
_SCALAR_FIELDS = (
    ("critical power", "critical_power_w", "cp_w"),
    ("anaerobic capacity", "w_prime_j", "w_prime_j"),
    ("HRV", "hrv_rmssd_ms", "rmssd_ms"),
    ("normalized power", "np_w", "np_w"),
)


def _scalar_lines(value: Any) -> list[str]:
    """Current-value lines for any recognised scalar-bearing Computed value (verbatim)."""
    lines: list[str] = []
    for plain, code, attr in _SCALAR_FIELDS:
        raw = getattr(value, attr, None)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            lines.append(f"current {plain} ({code}) {float(raw):g}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lines.append(f"current value {float(value):g}")
    return lines


def _looks_unavailable(record: Any) -> bool:
    """True iff the record is a typed Unavailable / fail-closed result (``available is False``)."""
    return getattr(record, "available", None) is False


__all__ = ["render_capability_factsheet", "render_fact_sheet"]
