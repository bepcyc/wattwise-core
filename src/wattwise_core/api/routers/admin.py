"""Admin/operator router — plan + model-policy config and system diagnosis (§8.16).

The operator console of API-R10: ``/v1/admin/*`` and ``GET /v1/system/diagnose`` all
require the ``admin`` scope (AUTH-R12) and are NOT athlete-facing, so precise technical
language is allowed (API-R21 binds athlete-visible copy only).

- ``GET /v1/admin/plans`` / ``PUT /v1/admin/plans/{plan_id}`` — the entitlement plan
  definitions. OSS ships exactly ONE default plan; a PUT validates the bounds through
  the same fail-closed ``validate_plan`` gate the boot uses (ENT-R6) and swaps the
  live resolver + resolved plan on app state, so the change is enforced at the real
  gates immediately. (The shape persists for the process; durable multi-plan storage
  is the commercial layer's concern, API-R36.)
- ``GET``/``PUT /v1/admin/model-policy`` — the dynamic model-selection policy
  (``allowed_tiers``/``default_tier``/``reasoning_ceiling``; the tier/reasoning enums
  match doc 50 exactly, SCHEMA-R3). The configured tier is an escalation CEILING —
  no caller can force a specific model per request (API-R38); OSS runs its one
  configured model (MODEL-R4), so the policy surface reports/bounds that seam.
- ``GET /v1/admin/model-catalog`` — the read-only operator catalog (never surfaced to
  athletes, API-R38).
- ``GET /v1/system/diagnose`` — admin-scoped operational self-diagnosis (AUTH-R12):
  deterministic checks over both stores + configuration, never per-athlete data.

Requirement IDs: API-R10 (§8.16/§8.9), AUTH-R7, AUTH-R11, AUTH-R12, API-R36, API-R38,
SCHEMA-R3 (model_tier/reasoning), ENT-R6 (validated plan swap), LIMIT-R1.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.deps import AppSettings, DbSession, RateLimit
from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.api.problems import not_found
from wattwise_core.entitlement import Entitlements, OssEntitlementResolver, validate_plan

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[RateLimit])
system_router = APIRouter(prefix="/v1/system", tags=["system"], dependencies=[RateLimit])

_Admin = Depends(require_scopes(Scope.ADMIN))

#: The single OSS plan id (one default all-permissive plan; more plans are commercial).
DEFAULT_PLAN_ID = "default"

ModelTier = Literal["flash", "pro", "frontier"]
Reasoning = Literal["low", "medium", "high"]


class PlanOut(BaseModel):
    """One entitlement plan definition on the wire (§8.16)."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    can_use_agent: bool
    can_ingest: bool
    can_export: bool
    node_visit_ceiling: int
    max_output_tokens: int
    wall_clock_seconds: float
    max_tool_iterations: int
    request_rate_per_minute: int


class PlanUpdate(BaseModel):
    """``PUT /v1/admin/plans/{plan_id}`` body: the editable plan bounds (§8.16)."""

    model_config = ConfigDict(extra="forbid")

    can_use_agent: bool = True
    can_ingest: bool = True
    can_export: bool = True
    node_visit_ceiling: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    wall_clock_seconds: float = Field(gt=0)
    max_tool_iterations: int = Field(ge=1)
    request_rate_per_minute: int = Field(ge=1)


class PlanList(BaseModel):
    """``GET /v1/admin/plans``: the (single, in OSS) plan set."""

    data: list[PlanOut]


class ModelPolicy(BaseModel):
    """The dynamic model-selection policy (§8.16; tier/reasoning enums per doc 50)."""

    model_config = ConfigDict(extra="forbid")

    allowed_tiers: list[ModelTier]
    default_tier: ModelTier
    reasoning_ceiling: Reasoning


class ModelCatalogEntry(BaseModel):
    """One catalog row for the operator's policy configuration (read-only)."""

    tier: ModelTier
    model: str
    input_cost_per_million_usd: float
    output_cost_per_million_usd: float


class ModelCatalog(BaseModel):
    """``GET /v1/admin/model-catalog``: available tiers + costs (operator-only)."""

    data: list[ModelCatalogEntry]


class DiagnosisCheck(BaseModel):
    """One deterministic operational check in the system diagnosis."""

    code: str
    ok: bool
    detail: str


class SystemDiagnosis(BaseModel):
    """``GET /v1/system/diagnose`` (admin, AUTH-R12): typed operational checks."""

    model_config = ConfigDict(extra="forbid")

    checks: list[DiagnosisCheck]
    overall_ok: bool


def _plan_out(plan: Entitlements) -> PlanOut:
    """Project the resolved entitlement plan onto the wire shape."""
    return PlanOut(
        plan_id=DEFAULT_PLAN_ID,
        can_use_agent=plan.can_use_agent,
        can_ingest=plan.can_ingest,
        can_export=plan.can_export,
        node_visit_ceiling=plan.node_visit_ceiling,
        max_output_tokens=plan.max_output_tokens,
        wall_clock_seconds=plan.wall_clock_seconds,
        max_tool_iterations=plan.max_tool_iterations,
        request_rate_per_minute=plan.request_rate_per_minute,
    )


def _live_plan(request: Request) -> Entitlements:
    """The app's resolved, validated entitlement plan (fail-closed when unwired)."""
    plan = getattr(request.app.state, "entitlement_plan", None)
    if not isinstance(plan, Entitlements):
        raise ProblemError("internal-error")
    return plan


@router.get("/plans", response_model=PlanList, operation_id="listAdminPlans", dependencies=[_Admin])
async def list_plans(request: Request) -> PlanList:
    """The entitlement plan definitions (§8.16). OSS ships exactly one default plan."""
    return PlanList(data=[_plan_out(_live_plan(request))])


@router.put(
    "/plans/{plan_id}",
    response_model=PlanOut,
    operation_id="updateAdminPlan",
    dependencies=[_Admin],
)
async def update_plan(plan_id: str, body: PlanUpdate, request: Request) -> PlanOut:
    """Edit the resolved plan shape (§8.16) through the SAME fail-closed validation.

    An unknown ``plan_id`` → ``404`` (OSS has exactly the one default plan). The update
    runs ``validate_plan`` (ENT-R6 — an invalid bound is rejected, never applied) and
    then swaps BOTH the resolver and the resolved plan on app state, so every gate that
    reads them enforces the new shape immediately.
    """
    if plan_id != DEFAULT_PLAN_ID:
        raise not_found()
    plan = validate_plan(Entitlements(**body.model_dump()))
    request.app.state.entitlement_resolver = OssEntitlementResolver(plan)
    request.app.state.entitlement_plan = plan
    return _plan_out(plan)


@router.get(
    "/model-policy",
    response_model=ModelPolicy,
    operation_id="getModelPolicy",
    dependencies=[_Admin],
)
async def get_model_policy(request: Request, settings: AppSettings) -> ModelPolicy:
    """The model-selection policy in force (§8.16).

    OSS runs ONE configured model (MODEL-R4): the policy defaults to that model's
    configured tier/reasoning seam and reflects any operator PUT for this process.
    """
    stored = getattr(request.app.state, "model_policy", None)
    if isinstance(stored, ModelPolicy):
        return stored
    return ModelPolicy(
        allowed_tiers=[settings.agent__tier],
        default_tier=settings.agent__tier,
        reasoning_ceiling=settings.agent__reasoning_effort,
    )


@router.put(
    "/model-policy",
    response_model=ModelPolicy,
    operation_id="updateModelPolicy",
    dependencies=[_Admin],
)
async def put_model_policy(body: ModelPolicy, request: Request) -> ModelPolicy:
    """Set the model-selection policy ceiling (§8.16, API-R38).

    The enums are closed (SCHEMA-R3: ``flash|pro|frontier`` / ``low|medium|high``) —
    an unknown member is ``422``. The ``default_tier`` must be allowed; the policy is
    a CEILING (per-node choices stay engine-owned; no caller can force a model).
    """
    if body.default_tier not in body.allowed_tiers:
        raise ProblemError(
            "validation-error",
            errors=[
                FieldError(
                    code="default_tier_not_allowed",
                    message="default_tier must be one of allowed_tiers",
                    pointer="/default_tier",
                )
            ],
        )
    request.app.state.model_policy = body
    return body


@router.get(
    "/model-catalog",
    response_model=ModelCatalog,
    operation_id="getModelCatalog",
    dependencies=[_Admin],
)
async def model_catalog(settings: AppSettings) -> ModelCatalog:
    """The read-only model catalog for policy configuration (§8.16, operator-only).

    OSS runs one configured BYO-key model (MODEL-R4), so the catalog reports that one
    seam: its configured tier, model id, and per-million costs (CFG-R1a-loaded; the
    operator sets real rates). Never surfaced to athletes (API-R38).
    """
    return ModelCatalog(
        data=[
            ModelCatalogEntry(
                tier=settings.agent__tier,
                model=settings.agent__model,
                input_cost_per_million_usd=settings.agent__cost__input_per_million_usd,
                output_cost_per_million_usd=settings.agent__cost__output_per_million_usd,
            )
        ]
    )


@system_router.get(
    "/diagnose",
    response_model=SystemDiagnosis,
    operation_id="systemDiagnose",
    dependencies=[_Admin],
)
async def system_diagnose(
    request: Request, session: DbSession, settings: AppSettings
) -> SystemDiagnosis:
    """Admin-scoped operational self-diagnosis (AUTH-R12, §8.9).

    Deterministic checks: the canonical database answers a real ``SELECT 1``, the
    default entitlement plan is loaded + validated, the LLM seam has a key, and the
    signing key is configured. No per-athlete data appears; failures carry technical
    detail (this is the operator console, not an athlete surface).
    """
    checks: list[DiagnosisCheck] = []
    try:
        await session.execute(text("SELECT 1"))
        checks.append(DiagnosisCheck(code="database", ok=True, detail="reachable"))
    except Exception as exc:  # a diagnosis reports, never raises
        checks.append(DiagnosisCheck(code="database", ok=False, detail=type(exc).__name__))
    plan = getattr(request.app.state, "entitlement_plan", None)
    plan_ok = isinstance(plan, Entitlements)
    checks.append(
        DiagnosisCheck(
            code="default_plan_loaded",
            ok=plan_ok,
            detail="loaded" if plan_ok else "missing",
        )
    )
    checks.append(
        DiagnosisCheck(
            code="llm_configured",
            ok=settings.llm_api_key is not None,
            detail="present" if settings.llm_api_key is not None else "unconfigured",
        )
    )
    checks.append(
        DiagnosisCheck(
            code="signing_key",
            ok=settings.token_signing_key is not None,
            detail="present" if settings.token_signing_key is not None else "unconfigured",
        )
    )
    return SystemDiagnosis(checks=checks, overall_ok=all(c.ok for c in checks))


__all__ = [
    "DEFAULT_PLAN_ID",
    "ModelCatalog",
    "ModelPolicy",
    "PlanOut",
    "SystemDiagnosis",
    "router",
    "system_router",
]
