"""The model-driven retrieval planner + its closed plan schema (doc 50, PLAN-R1..R3).

The focused sibling of :mod:`wattwise_core.agent.engine_services` (QUAL-R9 size split) that owns
the CONCRETE production planner seam: the CLOSED :class:`PlanCapability` enum the headline planner
may request (PLAN-R3), the provider-enforced :class:`_PlanSchema` structured plan (PLAN-R2), and
:class:`ModelPlanner` itself (PLAN-R1/R2) — the structured plan IS the selection, fail-closed to a
default capability on a structured-output failure (PLAN-R3 "handled as a re-plan, not a crash").
``engine_services`` imports and re-exports these so every historical
``from wattwise_core.agent.engine_services import ModelPlanner/_PlanSchema/...`` path (and the
``engine`` re-export chain on top of it) stays stable.

Cited requirements: PLAN-R1, PLAN-R2, PLAN-R3, STRUCT-R2, STRUCT-R3, CFG-R3, ARCH-R29, SKILL-R1.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from wattwise_core.agent.capabilities import CAPABILITY_BY_KEY
from wattwise_core.agent.contracts import ChatModel, RetrievalRequest
from wattwise_core.agent.structured import StructuredOutputError, run_structured
from wattwise_core.persistence.types import utcnow


class PlanCapability(StrEnum):
    """The CLOSED set of capabilities the headline planner may request (PLAN-R3 schema enum).

    The date-range capabilities the planner can request without an activity id (the per-activity /
    per-day capabilities need an id the planner does not have at plan time). PLAN-R3: the planner
    MUST be structurally UNABLE to express a capability outside this set — the schema enum (not a
    post-hoc filter) constrains it, so an out-of-registry request is a structured-output VALIDATION
    failure (the model emitting a non-member value never validates), routed as a re-plan, never
    silently dropped. Each member is a key of the single shared capability registry.
    """

    WEEKLY_LOAD = "weekly_load"
    CRITICAL_POWER = "critical_power"
    POWER_CURVE = "power_curve"


_DEFAULT_WINDOW_DAYS = 42


class _PlanSchema(BaseModel):
    """Provider-enforced retrieval plan (PLAN-R2/-R3): which canonical capabilities to gather.

    ``capabilities`` is a list of the CLOSED :class:`PlanCapability` ENUM (PLAN-R3): the model
    cannot emit a capability outside the registry — a non-member value is a structured-output
    validation failure handled as a re-plan (the planner's fail-closed default), NOT a silently
    dropped key. ``extra="forbid"`` rejects any unknown field (STRUCT-R3).
    """

    model_config = {"extra": "forbid"}
    capabilities: list[PlanCapability] = Field(default_factory=list)
    window_days: int = Field(default=_DEFAULT_WINDOW_DAYS, ge=1, le=365)


class ModelPlanner:
    """Model-driven retrieval planner (PLAN-R1/R2): the structured plan IS the selection.

    ``plan_system`` is the loaded planner system prompt (§16 / SKILL-R1): the engine embeds NO
    prompt inline (CFG-R3 / ARCH-R29) — the production wiring injects the verbatim fragment loaded
    from the coach-config bundle, and the empty default (``""``) preserves the prior behaviour for
    any seam that injects no coach-config (the FakeModel suite scripts the plan, so the prompt text
    is immaterial offline).
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        reference_date: _dt.date | None = None,
        plan_system: str = "",
    ) -> None:
        self._model = model
        self._today = reference_date or utcnow().date()
        self._plan_system = plan_system

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        """Emit the next batch of capability requests; fail-closed to a default on error (PLAN-R3).

        The plan's ``capabilities`` are the CLOSED :class:`PlanCapability` enum, so the model cannot
        express an out-of-registry capability: a non-member value is a structured-output validation
        failure (STRUCT-R2) that ``run_structured`` surfaces as :class:`StructuredOutputError`,
        handled here as a RE-PLAN to the default capability (PLAN-R3 "handled as a re-plan, not a
        crash") — never silently dropped.
        """
        try:
            plan = await run_structured(
                self._model,
                system=self._plan_system,
                data=f"question: {request_text}\nopen_gaps: {list(gaps)}\nalready: {list(already)}",
                schema=_PlanSchema,
            )
            keys = [c.value for c in plan.capabilities]
            window = plan.window_days
        except (StructuredOutputError, NotImplementedError):
            keys, window = [PlanCapability.WEEKLY_LOAD.value], _DEFAULT_WINDOW_DAYS
        if not keys:
            keys = [PlanCapability.WEEKLY_LOAD.value]
        frm = self._today - _dt.timedelta(days=window)
        params = {"from_date": frm.isoformat(), "to_date": self._today.isoformat()}
        seen = set(already)
        return [
            RetrievalRequest(capability=k, params=dict(params))
            for k in keys
            if k in CAPABILITY_BY_KEY and k not in seen
        ]


__all__ = ["ModelPlanner", "PlanCapability", "_PlanSchema"]
