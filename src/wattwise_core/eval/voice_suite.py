"""Voice follow-up liveness eval suite (QA-EVAL-R2.12 / QA-EVAL-R11 / COACH-R8).

A deterministic, network-free grader for the ``voice`` dataset. It certifies the COACH-R8
follow-up contract over the PROJECTION/VOICE surface (the shipped
:mod:`wattwise_core.agent.voice` primitives) — the CODE deciding, never the LLM (EVAL-R5),
exactly as :func:`wattwise_core.eval.suites.grade_readiness` does for readiness. No live
engine is driven (the engine is owned by another slice); the grader asserts the same
follow-up semantics the engine surfaces, against the deterministic voice primitives:

* **EXPAND climbs the ladder (VOICE-R8 / COACH-R8).** An EXPAND follow-up moves one rung up
  the verbosity ladder (``short -> standard -> detailed``, saturating at ``detailed``) so the
  NEXT answer admits MORE foregrounded numbers (the per-length :func:`number_cap` is
  monotone non-decreasing), i.e. is longer — never shorter.
* **DRILL / REVEAL_NUMBERS reveal VERBATIM, same thread, no scope-widen (VOICE-R9 /
  GROUND-R7 / INJECT-R3).** A drill/reveal follow-up targeting a prior observation surfaces
  that observation's ALREADY-grounded canonical ``{metric, value, as_of}`` number VERBATIM
  on the SAME durable ``thread_id``, and the authenticated identity + capability scope are
  UNCHANGED before/after (a follow-up never widens scope). A surfaced number that does not
  match the grounded citation, or a widened scope, is a defect.
* **Length-monotonicity (VOICE-R7).** The foregrounded-number budget is monotone across the
  ladder, so a longer response never admits fewer numbers than a shorter one.

The grade is a single dataclass with a 100% gate and a ``failures`` tuple, mirroring
:class:`wattwise_core.eval.grading.ReadinessGrade`. The dataset's NEGATIVE cases
(``negative_cases``) drive the grader's teeth (each asserted to FAIL its named property in
the suite's own tests), so the gate is provably non-vacuous.

Cited requirements: QA-EVAL-R2.12, QA-EVAL-R11, COACH-R8, VOICE-R7/-R8/-R9, GROUND-R7,
INJECT-R3, EVAL-R5, OUTCOME-R5; EVAL-R1 / TIER-R1 (offline, deterministic, no network).
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from wattwise_core.agent.projection import conversation_id_of, thread_id_for
from wattwise_core.agent.voice import Citation, Observation, ResponseLength, number_cap
from wattwise_core.eval.grading import VoiceGrade

_DATASETS_DIR = Path(__file__).parent / "datasets"
# The verbosity ladder an EXPAND follow-up climbs (COACH-R8 / VOICE-R8), saturating at the
# top rung — the SAME ladder the shipped deliverable uses; replicated here (not imported
# from the agent layer) so the eval slice stays decoupled from the engine modules (ARCH-R21).
_LENGTH_LADDER: tuple[ResponseLength, ...] = ("short", "standard", "detailed")
# A prescribed power/HR figure matches its grounded citation within this tolerance.
_VALUE_TOL = 0.01


def _load(name: str = "voice_liveness") -> dict[str, Any]:
    """Load the versioned checked-in voice dataset (QA-EVAL-R1, no network)."""
    loaded: dict[str, Any] = json.loads(
        (_DATASETS_DIR / f"{name}.json").read_text(encoding="utf-8")
    )
    return loaded


def _expanded_length(current: str) -> ResponseLength:
    """The next length up for an EXPAND follow-up; saturates at ``detailed`` (COACH-R8)."""
    idx = _LENGTH_LADDER.index(current) if current in _LENGTH_LADDER else 1
    return _LENGTH_LADDER[min(idx + 1, len(_LENGTH_LADDER) - 1)]


def _budget_for(length: str) -> int:
    """The foregrounded-number budget for a (possibly buggy/unknown) length string.

    Resolves a length name to the shipped per-length :func:`number_cap`; an unknown length
    (a planted-bug teeth case) maps to ``-1`` so it reads as SHORTER than any real rung,
    keeping the str/Literal boundary explicit without an ``Any`` cast.
    """
    if length not in _LENGTH_LADDER:
        return -1
    return number_cap(_LENGTH_LADDER[_LENGTH_LADDER.index(length)])


def _project_observations(raw: list[dict[str, Any]]) -> tuple[Observation, ...]:
    """Project dataset observations into typed :class:`Observation`s (COACH-R8)."""
    out: list[Observation] = []
    for item in raw:
        cites = tuple(
            Citation(
                record_id=str(c["record_id"]),
                metric=c.get("metric"),
                value=float(c["value"]) if c.get("value") is not None else None,
                as_of=c.get("as_of"),
            )
            for c in item.get("citations", [])
        )
        out.append(
            Observation(
                observation_id=str(item["observation_id"]),
                text=str(item["text"]),
                citations=cites,
            )
        )
    return tuple(out)


def _reveal_target(
    observations: tuple[Observation, ...], target_ref: str | None
) -> tuple[Observation, ...]:
    """The observation(s) a drill/reveal follow-up reveals (matches the shipped semantics).

    With a ``target_ref`` the matching observation is returned; with none, every
    observation carrying grounded numbers is revealed (COACH-R8 / VOICE-R9).
    """
    if target_ref is not None:
        return tuple(o for o in observations if o.observation_id == target_ref)
    return tuple(o for o in observations if o.citations)


def expand_failure(case: dict[str, Any]) -> str | None:
    """Certify an EXPAND follow-up climbs the ladder so the next answer is LONGER.

    The result length MUST be the next rung up (or saturate at ``detailed``) and its
    foregrounded-number budget MUST be >= the starting budget — never shorter (VOICE-R8).
    A positive case derives the result from the shipped ladder (:func:`_expanded_length`);
    a NEGATIVE (teeth) case may plant a buggy ``got_length`` (e.g. a regression that
    shortened the answer) so the deterministic check is proven to FLAG it.
    """
    cid = case["id"]
    start = str(case["start_length"])
    got = str(case["got_length"]) if "got_length" in case else _expanded_length(start)
    expected = str(case.get("expected_length", _expanded_length(start)))
    if "got_length" not in case and got != expected:
        return f"{cid}: EXPAND from {start!r} reached {got!r}, expected {expected!r}"
    if _budget_for(got) < _budget_for(start):
        return f"{cid}: EXPAND made the answer SHORTER ({start!r}->{got!r})"
    if got not in _LENGTH_LADDER or _LENGTH_LADDER.index(got) < _LENGTH_LADDER.index(start):
        return f"{cid}: EXPAND dropped a rung ({start!r}->{got!r}); a follow-up never shortens"
    return None


def reveal_failure(case: dict[str, Any]) -> str | None:
    """Certify a DRILL / REVEAL surfaces the verbatim number, same thread, no scope-widen.

    The targeted observation MUST exist and its grounded citation MUST carry the requested
    metric/value VERBATIM (within tolerance); the authenticated identity + scope MUST be
    UNCHANGED before/after; and the durable thread MUST be the SAME conversation
    (VOICE-R9 / GROUND-R7 / INJECT-R3 / CKPT-R3).
    """
    cid = case["id"]
    observations = _project_observations(case.get("observations", []))
    revealed = _reveal_target(observations, case.get("target_ref"))
    if not revealed:
        return f"{cid}: reveal target {case.get('target_ref')!r} resolved to no observation"
    if (problem := _scope_widen(case)) is not None:
        return problem
    return _verbatim_problem(cid, revealed, case)


def _scope_widen(case: dict[str, Any]) -> str | None:
    """Flag any identity change or capability-scope widening across the follow-up turn."""
    cid = case["id"]
    if case.get("athlete_before") != case.get("athlete_after"):
        return f"{cid}: follow-up changed the authenticated identity (INJECT-R3)"
    before = set(case.get("scope_before", []))
    after = set(case.get("scope_after", []))
    if after - before:
        return f"{cid}: follow-up WIDENED scope by {sorted(after - before)} (COACH-R8)"
    return None


def _verbatim_problem(
    cid: str, revealed: tuple[Observation, ...], case: dict[str, Any]
) -> str | None:
    """Flag a revealed number that is not the grounded citation, or a divergent thread."""
    metric = str(case["requested_metric"])
    value = float(case["requested_value"])
    grounded = [
        c.value
        for obs in revealed
        for c in obs.citations
        if c.metric == metric and c.value is not None
    ]
    if not grounded:
        return f"{cid}: targeted observation has no grounded {metric!r} citation to reveal"
    if all(abs(g - value) > _VALUE_TOL for g in grounded):
        return (
            f"{cid}: revealed {metric}={value} does not match the grounded citation "
            f"{grounded} (a reveal is VERBATIM, never a new number — GROUND-R7)"
        )
    return _thread_problem(cid, case)


def _thread_problem(cid: str, case: dict[str, Any]) -> str | None:
    """Certify the follow-up resumed the SAME durable thread, bound to the SAME athlete.

    A reveal/drill reuses the prior durable ``thread_id`` (CKPT-R3): it must reverse to a
    non-empty conversation id AND reconstruct to itself under the AUTHENTICATED athlete
    (``thread_id_for(athlete_after, conversation_id_of(thread)) == thread``). A thread that
    does not round-trip under the authenticated athlete is a different / cross-identity
    thread — a scope break the follow-up may never take (AGT-SEC-R1 / CKPT-R3).
    """
    thread = case.get("thread_id")
    if thread is None:
        return None
    thread = str(thread)
    convo = conversation_id_of(thread)
    if not convo:
        return f"{cid}: follow-up did not resume a durable thread (CKPT-R3)"
    athlete = str(case.get("athlete_after", ""))
    if athlete and thread_id_for(athlete, convo) != thread:
        return (
            f"{cid}: follow-up thread {thread!r} is not bound to the authenticated athlete "
            f"{athlete!r} — a cross-identity thread (AGT-SEC-R1 / CKPT-R3)"
        )
    return None


def monotone_failure(_case: dict[str, Any]) -> str | None:
    """Certify the foregrounded-number budget is monotone across the ladder (VOICE-R7)."""
    caps = [number_cap(length) for length in _LENGTH_LADDER]
    if any(b < a for a, b in pairwise(caps)):
        return f"{_case['id']}: length-number budget is NOT monotone across {_LENGTH_LADDER}"
    return None


_CHECKS = {
    "expand": expand_failure,
    "reveal_numbers": reveal_failure,
    "drill": reveal_failure,
    "monotone": monotone_failure,
}


def case_failure(case: dict[str, Any]) -> str | None:
    """Dispatch one case to the deterministic check for its follow-up kind (COACH-R8)."""
    check = _CHECKS.get(str(case["kind"]))
    if check is None:
        return f"{case['id']}: unknown voice follow-up kind {case['kind']!r}"
    return check(case)


def grade_voice() -> VoiceGrade:
    """Grade the voice follow-up liveness fixtures deterministically (QA-EVAL-R2.12).

    For each POSITIVE case the grader runs the deterministic check for its follow-up kind
    (EXPAND climbs / DRILL+REVEAL verbatim-same-thread-no-widen / MONOTONE budget) and
    records a failure for each that does not hold. The gate is 100%; the negative-case
    teeth are exercised by the suite's own tests, not here.
    """
    cases = _load()["cases"]
    failures: list[str] = []
    passed = 0
    for case in cases:
        reason = case_failure(case)
        if reason is None:
            passed += 1
        else:
            failures.append(reason)
    return VoiceGrade(len(cases), passed, tuple(failures))


__all__ = [
    "VoiceGrade",
    "case_failure",
    "expand_failure",
    "grade_voice",
    "monotone_failure",
    "reveal_failure",
]
