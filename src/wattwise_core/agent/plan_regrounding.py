"""Re-grounding helpers for a HITL-edited approval-gated plan (doc 50 GROUND-R3 / H3).

The focused sibling of :mod:`wattwise_core.agent.engine` (QUAL-R9 size split) that owns the edit
re-grounding the decision endpoint runs before resuming a paused PLAN: an athlete's ``edit`` is
untrusted prose, so it is re-verified through the SAME fail-closed grounder — built from the LOADED
:class:`CoachBundle` (its metric-equivalence + numeric tolerance + first-party URL allow-list +
dateless-claim lookback, M6) plus the canonical workout-NAME library — and accepted ONLY when it
fully grounds. A partial/abstained/extraction-failed edit is rejected so the run degrades rather
than shipping unverified content (H3).
"""

from __future__ import annotations

from wattwise_core.agent.contracts import ChatModel, GroundDecision, GroundingResult
from wattwise_core.agent.engine_services import (
    CANONICAL_WORKOUT_NAMES,
    ClaimGrounder,
    CoachBundle,
)
from wattwise_core.analytics.service import AnalyticsService


async def reground_plan(
    coach: CoachBundle,
    model: ChatModel,
    svc: AnalyticsService,
    athlete_id: str,
    edited_plan: str,
) -> GroundingResult:
    """Re-ground an EDITED plan body before resume so the edit cannot bypass grounding (GROUND-R3).

    The athlete's edit is untrusted prose: it is run through the SAME fail-closed grounder — built
    from the LOADED :class:`CoachBundle` (metric-equivalence + tolerance + URL allow-list +
    lookback, CFG-R1a / M6) PLUS the canonical workout-NAME library — so any unverified
    number/name/URL is scrubbed. Returns the full :class:`GroundingResult` (NOT just scrubbed text):
    the caller MUST inspect ``decision`` and accept the edit ONLY when it fully grounds (``PROCEED``
    + non-empty), never shipping a partial/abstained/untrusted edit (H3). On extraction failure the
    grounder yields no claims -> nothing is scrubbed and the decision is ``ABSTAIN`` (NOT
    ``PROCEED``), so the untrusted edit is rejected by the caller, never published.
    """
    grounder = ClaimGrounder(
        model,
        svc,
        allow_names=CANONICAL_WORKOUT_NAMES,
        equivalence=coach.equivalence,
        tolerance=coach.tolerance,
        allowed_hosts=coach.allowed_hosts,
        lookback_days=coach.lookback_days,
    )
    return await grounder.ground(athlete_id=athlete_id, draft=edited_plan, retrieved={})


async def accept_edit(
    coach: CoachBundle,
    model: ChatModel,
    svc: AnalyticsService,
    athlete_id: str,
    edited_plan: str,
) -> str | None:
    """Re-ground an edit; return its grounded body ONLY if it fully grounds, else ``None`` (H3).

    The edit is accepted (and published) ONLY when re-grounding decides ``PROCEED`` with non-empty
    grounded text — i.e. EVERY checkable claim grounded and something survives. A ``REGENERATE`` /
    ``REPLAN`` / ``ABSTAIN`` outcome (an unverified or wholly-scrubbed edit, or extraction failure
    that yields no claims and abstains) returns ``None`` so the caller rejects it. The grounder is
    built from the LOADED CoachBundle (M6), so the edit path uses the SAME config-loaded
    equivalence/tolerance/allow-list as every other grounder path.
    """
    result = await reground_plan(coach, model, svc, athlete_id, edited_plan)
    if result.decision is GroundDecision.PROCEED and result.scrubbed_text.strip():
        return result.scrubbed_text
    return None


__all__ = ["accept_edit", "reground_plan"]
