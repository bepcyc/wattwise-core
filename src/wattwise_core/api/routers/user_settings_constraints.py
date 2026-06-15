"""Athlete safety-constraint capture surface — ``/v1/user-settings/constraints`` (MEM-R7, ADR 0008).

The focused sibling of :mod:`wattwise_core.api.routers.user_settings` (QUAL-R9 size split) that owns
the athlete-facing CONSTRAINT capture endpoints (issue #77, proposed MEM-R7 / GROUND-R14): record a
safety constraint (injury / medical advice / hard life limit), list the active set, and lift one.
A constraint is an agent-state memory item (doc 50 MEM-R7), NOT a canonical §3 master-data entity —
so its add/list/lift reach the SAME agent-state store the run path recalls its always-resident
constraint tier from (MEM-R6) and the grounding gate enforces (GROUND-R13/R14), through the shared
engine seam, NOT the canonical store.

The router REUSES the parent module's scope/identity seams (``require_read_scope`` /
``require_write_scope`` / ``current_athlete_id``) so the app factory's single set of overrides gate
both surfaces identically (AUTH-R3/R7/R11): identity is server-derived and every read/write acts
ONLY on that one owner id — no writable caller-identity field exists on a request body (SCHEMA-R4).
The one new seam is :func:`constraint_store` (the agent-state engine), bound by the app factory.

Requirement IDs: API-R32, AUTH-R3, AUTH-R7, AUTH-R11, AUTH-R18, SCHEMA-R4, ERR-R6, MEM-R6, MEM-R7,
GROUND-R14, AGT-SEC-R1.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import Annotated, Literal, Protocol, runtime_checkable

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from wattwise_core.agent.memory import ConstraintSeverity, RecalledItem
from wattwise_core.api.deps import RateLimit
from wattwise_core.api.errors import ProblemError
from wattwise_core.api.routers.user_settings import (
    AthleteId,
    require_read_scope,
    require_write_scope,
)

router = APIRouter(prefix="/v1/user-settings", tags=["user-settings"], dependencies=[RateLimit])

#: The athlete-facing constraint severity tokens (GROUND-R14); mirrors the agent ConstraintSeverity.
ConstraintSeverityLiteral = Literal["hard", "soft"]

_Read = Depends(require_read_scope)
_Write = Depends(require_write_scope)


@runtime_checkable
class ConstraintStore(Protocol):
    """The AGENT-STATE-backed athlete-constraint capture seam (MEM-R7 / GROUND-R14, ADR 0008).

    A CONSTRAINT is an agent-state memory item, NOT canonical master-data — so its add/list/lift
    reach the SAME store the run path recalls its constraint tier from (MEM-R6), through the shared
    engine. Identity is server-derived (AGT-SEC-R1); the seam never widens scope from a client arg.
    """

    async def add_constraint(
        self,
        *,
        athlete_id: str,
        content: str,
        severity: ConstraintSeverity,
        effective_until: _dt.datetime | None,
    ) -> RecalledItem: ...

    async def list_active_constraints(self, *, athlete_id: str) -> Sequence[RecalledItem]: ...

    async def lift_constraint(self, *, athlete_id: str, memory_item_id: str) -> bool: ...


def constraint_store() -> ConstraintStore:
    """The agent-state constraint store seam; the app factory overrides it (fail-closed)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


ConstraintStoreDep = Annotated[ConstraintStore, Depends(constraint_store)]


class ConstraintCreate(BaseModel):
    """A new athlete-stated CONSTRAINT to record (MEM-R7 / GROUND-R14, ADR 0008 §5).

    ``content`` is the athlete's own stated limit ("doctor said no running for 6 weeks"), preserved
    verbatim (MEM-R2). ``severity`` selects whether a contradicting prescription is VETOED (``hard``
    — an absolute contraindication, never published) or surfaced as a CAUTION (``soft`` — a relative
    one, the shared-decision default). ``effective_until`` is an optional self-expiry instant.
    ``additionalProperties:false`` rejects a forged ``athlete_id`` (SCHEMA-R4): identity is
    server-derived, never client-supplied.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=2048)
    severity: ConstraintSeverityLiteral = "soft"
    effective_until: _dt.datetime | None = None


class ConstraintOut(BaseModel):
    """One active constraint as the capture surface returns it (MEM-R7, personalization only).

    Carries the stable ``memory_item_id`` (the lift handle), the verbatim ``content``, the
    ``severity``, ``inferred`` (engine-derived vs athlete-stated), and the optional
    ``effective_until``. Carries NO canonical analytic number (MEM-R1).
    """

    model_config = ConfigDict(extra="forbid")

    memory_item_id: str
    content: str
    severity: ConstraintSeverityLiteral
    inferred: bool
    effective_until: _dt.datetime | None = None


class ConstraintList(BaseModel):
    """The owner's ACTIVE constraint set (HARD-first, then most-recent; MEM-R6/-R7)."""

    model_config = ConfigDict(extra="forbid")

    constraints: list[ConstraintOut] = Field(default_factory=list)


def _constraint_out(item: RecalledItem) -> ConstraintOut:
    """Project a recalled constraint row onto the wire shape (severity defaults closed to soft)."""
    severity: ConstraintSeverityLiteral = (
        "hard" if item.severity is ConstraintSeverity.HARD else "soft"
    )
    return ConstraintOut(
        memory_item_id=item.memory_item_id,
        content=item.content,
        severity=severity,
        inferred=item.inferred,
        effective_until=item.effective_until,
    )


@router.get(
    "/constraints",
    response_model=ConstraintList,
    operation_id="listUserConstraints",
    dependencies=[_Read],
)
async def list_constraints(store: ConstraintStoreDep, athlete_id: AthleteId) -> ConstraintList:
    """List the owner's ACTIVE safety constraints (MEM-R6/-R7, HARD-first then most-recent).

    Reads the always-resident constraint tier (MEM-R6, ADR 0008 §3) — the SAME set the run path
    recalls and the grounding gate enforces — scoped to the server-derived owner (AGT-SEC-R1).
    Personalization context only, never a canonical number (MEM-R1).
    """
    items = await store.list_active_constraints(athlete_id=athlete_id)
    return ConstraintList(constraints=[_constraint_out(item) for item in items])


@router.post(
    "/constraints",
    response_model=ConstraintOut,
    status_code=201,
    operation_id="createUserConstraint",
    dependencies=[_Write],
)
async def create_constraint(
    body: ConstraintCreate, store: ConstraintStoreDep, athlete_id: AthleteId
) -> ConstraintOut:
    """Record a new athlete-stated safety constraint (MEM-R7 / GROUND-R14, ADR 0008 §5).

    Persists the constraint in the athlete's own words into the agent-state store, scoped to the
    server-derived owner (AGT-SEC-R1) — so the run path recalls it (MEM-R6) and a contradicting
    prescription is vetoed (``hard``) or cautioned (``soft``) at the grounding gate. Identity is
    never taken from the client (SCHEMA-R4). Returns the persisted constraint.
    """
    severity = ConstraintSeverity.HARD if body.severity == "hard" else ConstraintSeverity.SOFT
    item = await store.add_constraint(
        athlete_id=athlete_id,
        content=body.content,
        severity=severity,
        effective_until=body.effective_until,
    )
    return _constraint_out(item)


@router.delete(
    "/constraints/{memory_item_id}",
    status_code=204,
    operation_id="liftUserConstraint",
    dependencies=[_Write],
)
async def lift_constraint(
    memory_item_id: str, store: ConstraintStoreDep, athlete_id: AthleteId
) -> None:
    """Lift (clear) the owner's constraint by id (MEM-R7, the shared-decision StARRT stance).

    A LIFTED constraint stops gating. The lift is scoped to BOTH the id AND the server-derived
    owner (AGT-SEC-R1): a cross-athlete / unknown / non-UUID id lifts nothing and reads as a 404 —
    a foreign row is never confirmed to exist (fail-closed). Returns ``204`` on a successful lift.
    """
    lifted = await store.lift_constraint(athlete_id=athlete_id, memory_item_id=memory_item_id)
    if not lifted:
        raise ProblemError("not-found")


__all__ = [
    "ConstraintCreate",
    "ConstraintList",
    "ConstraintOut",
    "ConstraintStore",
    "constraint_store",
    "router",
]
