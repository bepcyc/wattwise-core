"""Leaf voice/projection layer: the shared coach-voice primitives (VOICE-R7/-R8).

This is the LEAF module of the deliverables family (ARCH-R21 / QUAL-R9): it owns the
voice-contract primitives that BOTH :mod:`wattwise_core.agent.deliverables` (the
free-form answer + weekly digest) and :mod:`wattwise_core.agent.readiness_deliverable`
(the readiness/form deliverable) build on — the grounded-citation shape
(:class:`Citation`), the per-turn observation (:class:`Observation`), the response-length
verbosity knob (:data:`ResponseLength` + :func:`number_cap`), and the DETERMINISTIC
presentation checks/enforcement (leads-with-state, foregrounded-number count, number-cap
demotion, citation projection).

It imports NOTHING from any sibling deliverable / engine / api / persistence module — only
stdlib (and, by contract, ``agent/contracts``/pydantic when needed) — so it sits strictly
BELOW both deliverable modules in the import graph. Hoisting these shared primitives here
(rather than into one deliverable that the other imports back) is what breaks the former
``deliverables`` <-> ``readiness_deliverable`` cycle: both now depend DOWNWARD on this leaf,
and ``deliverables`` re-exports these names so every historical import path stays stable.

The voice contract is a PRESENTATION layer over the graph's fail-closed grounding, never a
relaxation of it (VOICE-R7): this module rewrites no number and certifies no groundedness —
it projects what the graph grounded and runs the deterministic leads-with-state /
number-count checks that gate the two presentation properties (EVAL-R5b.1).

Cited requirements: COACH-R7, COACH-R8, GROUND-R5/-R7, VOICE-R7/-R8/-R9, EVAL-R5b.1.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

# Athlete-facing verbosity (VOICE-R8); the persisted default is ``standard``.
ResponseLength = Literal["short", "standard", "detailed"]

# Number-density CAP per response length (VOICE-R7 defaults; exact ceilings live in
# the loaded persona config, so callers MAY override via ``number_cap``).
_NUMBER_CAP: Mapping[ResponseLength, int] = {"short": 2, "standard": 3, "detailed": 4}

# Matches a foregrounded explicit numeric value in athlete-facing prose for the
# deterministic number-density count (VOICE-R7 / EVAL-R5b.1). Plain integers and
# decimals, optionally signed; standalone, so dates/words are not miscounted.
_NUMBER_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])")

# Tags stripped to read the LEADING athlete-facing sentence out of grounded HTML for
# the deterministic leads-with-state check (the body is sanitized later by the API).
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class Citation:
    """A surviving grounded claim's pointer to its canonical record (GROUND-R5).

    Shape ``{metric, value, as_of}`` referencing a canonical record id (activity /
    analytic-computation / workout / plan), NEVER a source/provider id. ``value`` is
    taken VERBATIM from canonical analytics (GROUND-R7); this layer never recomputes.
    """

    record_id: str
    metric: str | None = None
    value: float | None = None
    as_of: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """One distinct athlete-facing observation carrying a STABLE id (COACH-R8).

    The stable ``observation_id`` is the expand/drill handle a later follow-up turn
    targets without re-stating the original question. ``citations`` are the grounded
    numbers behind the observation, surfaced on demand (VOICE-R9), never as a hero
    metrics dump.
    """

    observation_id: str
    text: str
    citations: tuple[Citation, ...] = ()


# --- deterministic presentation checks (the GATE of EVAL-R5b.1) ---


def first_sentence(html_or_text: str) -> str:
    """Return the leading athlete-facing sentence with markup/whitespace stripped.

    Reads the lead out of the (later-sanitized) grounded body so the leads-with-state
    check (COACH-R7) inspects what the athlete actually sees first.
    """
    plain = _TAG_RE.sub(" ", html_or_text)
    plain = " ".join(plain.split())
    for end in (". ", "! ", "? "):
        idx = plain.find(end)
        if idx != -1:
            return plain[: idx + 1].strip()
    return plain.strip()


def count_foregrounded_numbers(html_or_text: str) -> int:
    """Count explicit foregrounded numeric values in athlete-facing prose (VOICE-R7).

    The deterministic number-density measurement; the caller compares it against the
    per-length cap. Markup is stripped first so attribute digits are not counted.
    """
    plain = _TAG_RE.sub(" ", html_or_text)
    return len(_NUMBER_RE.findall(plain))


def leads_with_state(html_or_text: str) -> bool:
    """True iff the leading sentence reads as a state phrase, not a bare metric token.

    Deterministic gate for COACH-R7 / EVAL-R5b.1: a lead that is ONLY a number or a
    metric/jargon token (no plain-language words) fails. A normal warm sentence — even
    one that mentions a grounded number in passing — passes, because it carries
    sentence words around the value.
    """
    lead = first_sentence(html_or_text)
    if not lead:
        return False
    stripped = _NUMBER_RE.sub(" ", lead)
    words = [w for w in re.findall(r"[^\W\d_]+", stripped, flags=re.UNICODE) if len(w) > 1]
    return len(words) >= 2


# --- citation projection ---


def _to_citation(raw: Mapping[str, Any]) -> Citation:
    """Project one graph citation mapping into the typed :class:`Citation` (GROUND-R5).

    Reads the canonical ``{metric, value, as_of}`` + record-id shape; a citation with
    no resolvable record id is dropped by the caller (no claim without a citation).
    """
    value = raw.get("value")
    return Citation(
        record_id=str(raw.get("record_id", "")),
        metric=_opt_str(raw.get("metric")),
        value=float(value) if isinstance(value, (int, float)) else None,
        as_of=_opt_str(raw.get("as_of")),
    )


def _opt_str(value: Any) -> str | None:
    """Coerce an optional graph field to ``str | None`` without inventing a value."""
    return None if value is None else str(value)


def _project_citations(raw: Sequence[Mapping[str, Any]]) -> tuple[Citation, ...]:
    """Project + filter graph citations: keep only those with a resolvable record id."""
    out = (_to_citation(c) for c in raw)
    return tuple(c for c in out if c.record_id)


# --- number-density cap ---


def number_cap(response_length: ResponseLength) -> int:
    """Return the foregrounded-number ceiling for a response length (VOICE-R7 default)."""
    return _NUMBER_CAP[response_length]


def _enforce_number_cap(html: str, text: str, cap: int) -> tuple[str, str]:
    """Deterministically hold the body to the foregrounded-number cap (VOICE-R7).

    If the projected body foregrounds more explicit numbers than the per-length ceiling,
    the surplus foregrounded numbers (keeping the first ``cap``) are demoted to a plain
    "(value omitted)" token so the cap is ENFORCED on what ships — not merely test-asserted
    (EVAL-R5b.1). The grounded numbers themselves remain available via the citations /
    reveal-numbers follow-up; only the in-prose density is bounded.
    """
    if count_foregrounded_numbers(text) <= cap:
        return html, text
    return _demote_numbers(html, cap), _demote_numbers(text, cap)


def _demote_numbers(body: str, cap: int) -> str:
    """Keep the first ``cap`` foregrounded numbers; replace the rest with a token."""
    seen = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        return match.group(0) if seen <= cap else "(value omitted)"

    return _NUMBER_RE.sub(_sub, body)


__all__ = [
    "Citation",
    "Observation",
    "ResponseLength",
    "count_foregrounded_numbers",
    "first_sentence",
    "leads_with_state",
    "number_cap",
]
