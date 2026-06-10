"""Typed degraded-caveat + plan-body wire schemas (API-R11a), shared by the agent slices.

Split out of :mod:`agent_schemas` (QUAL-R9 size ceiling): the ``degraded`` member's
machine-readable coverage caveat and the ``awaiting_approval`` member's structured
``plan`` body, imported by both the core agent schema module and the breadth slices
without an import cycle.

Requirement IDs: API-R11a (status-discriminated union members), API-R6 (additive-safe),
API-R13 (sanitized plan_html), SCHEMA-R9 (source-agnostic coverage classes).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CoverageCaveatOut(BaseModel):
    """The TYPED coverage caveat of a ``degraded`` member (API-R11a / SCHEMA-R9).

    The machine-readable basis a client uses to badge "reduced precision / estimated"
    WITHOUT parsing ``reason_text``: which canonical inputs were ``missing`` /
    ``substituted`` / ``stale`` (source-agnostic class names, never a provider) and the
    resulting ``fidelity``. ``reason``/``inputs_unavailable`` carry the engine's typed
    no-coverage / unconfigured notes. ``extra="allow"`` keeps the caveat additive-safe
    (API-R6) while the named fields stay typed at the boundary.
    """

    model_config = ConfigDict(extra="allow")

    missing: list[str] = Field(default_factory=list)
    substituted: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    fidelity: Literal["full", "partial", "degraded"] | None = None
    reason: str | None = None
    inputs_unavailable: list[str] = Field(default_factory=list)


class PlanBodyOut(BaseModel):
    """The structured ``plan`` object of an ``awaiting_approval`` member (API-R11a).

    The grounded proposed plan body the athlete approves/edits/rejects via the decision
    endpoint. ``plan_html`` is already server-side sanitized (API-R13). The engine's plan
    deliverable grounds workout NAMES into prose (GROUND-R2); a structured
    ``PrescribedWorkout[]`` projection is engine-side work tracked under doc 50 — until it
    lands, this object carries the grounded body so clients branch on a typed ``plan``
    field rather than loose top-level strings.
    """

    plan_html: str
    plan_text: str


__all__ = ["CoverageCaveatOut", "PlanBodyOut"]
