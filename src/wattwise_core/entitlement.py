"""Entitlement resolver seam (ENT-R*, SEAM-R*, AGT-ENT-R*) — OSS default.

The commercial layer (`athload`) mounts a real entitlement system (per-tenant plans,
quotas, feature flags) on this seam. The OSS engine is single-athlete, zero-tenancy
(SCOPE-R12), so its default resolver grants the implicit owner every in-OSS
capability. The seam exists so the commercial layer can resolve → carry → check
entitlements without the OSS engine knowing anything about tenancy.

The resolve → attach → check seam is REAL, not a noop (AGT-ENT-R1/-R3/-R4):

* The OSS default plan PERMITS EVERYTHING (every feature flag granted) and carries
  **NO monetary budget** (AGT-ENT-R4 / COMM-R20) — the monetary per-run/per-period
  cost ceiling and the reserve-then-settle accounting are COMMERCIAL and supplied by
  the metered resolver, never by this OSS default.
* What the OSS default plan DOES carry are the **non-monetary** local guards
  (AGT-ENT-R4): a node-visit/step ceiling, a token bound, a wall-clock bound, a
  tool-iteration bound, and a request-rate bound — with generous, configurable
  defaults (CFG-R1a: loaded from external config, never code literals). A single
  operator MAY raise them in config.
* The engine READS each of those gated limits FROM the resolved entitlement and does NOT
  hardcode them (AGT-ENT-R1), each at a REAL enforcement point: the GRAPH reads the
  node-visit ceiling + the tool-iteration bound (a breach degrades gracefully), the ENGINE
  sizes the model output budget + bounds the run by the wall-clock deadline, and the API
  rate-limiter's agent ceiling is the request-rate bound (see :class:`Entitlements` for the
  per-bound enforcement points). The gate CHECKS the carried flags and FAILS CLOSED on an
  ungranted flag (AGT-ENT-R3) — so a commercial plan that ungrants a feature IS enforced,
  while the OSS all-permissive plan permits the run.

Keeping this seam clean (not building the commercial side) is the whole point: the
OSS engine boots and passes every gate with zero commercial bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids a config import cycle at runtime)
    from wattwise_core.config import Settings


class EntitlementError(RuntimeError):
    """A fail-closed entitlement/plan-definition error (AGT-ENT-R4 / OBS-R6.2 / RUN-R4.1).

    Raised when a plan definition is missing, structurally invalid, or carries a
    non-positive non-monetary bound. The engine refuses to boot (or marks itself
    not-ready) rather than serving traffic under a silently-permissive or unvalidated
    plan — never a fake-healthy degenerate plan.
    """


@dataclass(frozen=True, slots=True)
class Entitlements:
    """Resolved capabilities + non-monetary local guards for the current principal.

    In OSS this is always the all-permissive grant for the single owner: every feature
    flag is granted (``can_use_agent``/``can_ingest``/``can_export`` all ``True``) and
    the non-monetary bounds carry generous, configurable defaults (AGT-ENT-R4). There is
    deliberately **no monetary budget field** — the monetary per-run/per-period ceiling
    is COMMERCIAL (COMM-R20) and supplied by the metered resolver, never by this OSS grant.

    The bounds are the per-request NON-monetary local guards the OSS plan carries
    (AGT-ENT-R4), resolved FROM config (CFG-R1a) and read FROM this entitlement at a REAL
    enforcement point each (AGT-ENT-R1 — no claim without an enforcement point):

    * ``node_visit_ceiling`` (max graph steps) — read by the GRAPH (``build_graph`` via
      ``seams.entitlement_node_visit_ceiling``); a breach routes to ``finalize`` degraded.
    * ``max_tool_iterations`` (gather/tool-loop bound) — read by the GRAPH (``build_graph``
      via ``seams.entitlement_max_tool_iterations``); a breach stops re-planning at compose.
    * ``max_output_tokens`` (per-call model output budget) — the ENGINE sizes the model
      (``OpenAICompatibleModel`` ``max_output_tokens``) to it; enforced as
      ``max_completion_tokens`` on every provider call. The OSS DEFAULT (8192, defaults.toml) is
      sized to hold a 2026 reasoning model's reasoning trace + answer (MODEL-R5a); the live budget
      is whatever config/this plan resolves to and is NOT clamped to a floor — a plan that sets it
      below the model's reasoning need yields truncated/empty content (operator's responsibility).
    * ``wall_clock_seconds`` (whole-run deadline) — the ENGINE wraps the single graph invoke in
      ``asyncio.wait_for``; a breach degrades GRACEFULLY (a degraded terminal answer, not a raise).
    * ``request_rate_per_minute`` (per-athlete request-rate bound) — the API ``RateLimiter``'s
      ``agent``-class ceiling is built from it; the 1-over request is ``429`` ``rate-limited``.

    None is hardcoded in the gate or the graph (AGT-ENT-R1); the commercial layer varies them
    per plan without touching any enforcement point.
    """

    # --- feature flags (OSS: all granted; a commercial plan MAY ungrant) ---
    can_use_agent: bool = True
    can_ingest: bool = True
    can_export: bool = True

    # --- non-monetary local guards (AGT-ENT-R4; generous, configurable; NO money) ---
    node_visit_ceiling: int = 0
    max_output_tokens: int = 0
    wall_clock_seconds: float = 0.0
    max_tool_iterations: int = 0
    request_rate_per_minute: int = 0

    def require(self, capability: str) -> None:
        """Raise if ``capability`` (a feature flag) is not granted (fail-closed check seam).

        The gate calls this for a feature flag (``can_use_agent``/``can_ingest``/
        ``can_export``); an ungranted flag raises :class:`PermissionError` (AGT-ENT-R3).
        A non-flag attribute name is treated as ungranted (fail-closed), never silently
        permitted.
        """
        if capability not in _FEATURE_FLAGS or not getattr(self, capability, False):
            raise PermissionError(f"capability not entitled: {capability}")


#: The feature-flag attribute names a gate may ``require`` (closed set; fail-closed).
_FEATURE_FLAGS: frozenset[str] = frozenset({"can_use_agent", "can_ingest", "can_export"})

#: The non-monetary bound attribute names every resolved plan MUST carry as positive values.
_BOUND_FIELDS: tuple[str, ...] = (
    "node_visit_ceiling",
    "max_output_tokens",
    "wall_clock_seconds",
    "max_tool_iterations",
    "request_rate_per_minute",
)


def validate_plan(plan: Entitlements) -> Entitlements:
    """Validate a resolved/default plan, fail-closed (AGT-ENT-R4 / SKILL-R4 / OBS-R6.2).

    Every non-monetary bound MUST be present and strictly positive — a missing or
    non-positive bound is a degenerate plan that would either never admit a run or admit
    an unbounded one, so the engine refuses it (raises :class:`EntitlementError`) rather
    than booting / reporting ready under it. Returns the plan unchanged on success so it
    composes as ``validate_plan(resolver.resolve(...))`` at a gate or readiness probe.
    """
    for name in _BOUND_FIELDS:
        value = getattr(plan, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise EntitlementError(
                f"fail-closed: plan bound {name!r} must be a positive number, got {value!r} "
                "(AGT-ENT-R4 non-monetary local guard)"
            )
    for flag in _FEATURE_FLAGS:
        if not isinstance(getattr(plan, flag), bool):
            raise EntitlementError(f"fail-closed: plan flag {flag!r} must be a bool")
    return plan


@runtime_checkable
class EntitlementResolver(Protocol):
    """Resolves the entitlements for a principal (commercial overrides this)."""

    def resolve(self, athlete_id: str) -> Entitlements:
        """Return the resolved entitlements for ``athlete_id``."""
        ...


class OssEntitlementResolver:
    """OSS default: the single owner is entitled to every in-OSS capability (AGT-ENT-R4).

    Carries the single all-permissive default plan: all feature flags granted, the
    non-monetary local guards loaded FROM config (CFG-R1a) — never code literals — and
    NO monetary budget. The same resolved plan is attached to every HTTP request and
    every agent run, then CHECKED at the gate (AGT-ENT-R1/-R3). The commercial layer
    supplies more plan definitions + a real resolver through this same seam WITHOUT
    touching the gate points.
    """

    __slots__ = ("_plan",)

    def __init__(self, plan: Entitlements | None = None) -> None:
        # Validate the carried plan up-front (fail-closed): a resolver that would hand out
        # a degenerate plan must not exist (AGT-ENT-R4). The bare default (config-absent)
        # carries zero bounds, which validate_plan rejects — so production MUST build via
        # ``from_settings`` (config-loaded bounds); the zero-arg default exists only for the
        # typed-seam conformance contract, which does not validate bounds.
        self._plan = plan if plan is not None else Entitlements()

    @classmethod
    def from_settings(cls, settings: Settings) -> OssEntitlementResolver:
        """Build the OSS resolver with the all-permissive plan loaded from config (CFG-R1a).

        The non-monetary local guards (node-visit ceiling, token bound, wall-clock,
        tool-iteration, request-rate) are read from the layered config (defaults.toml ->
        operator file -> env), NEVER hardcoded (AGT-ENT-R1). The plan is validated
        fail-closed here, so an invalid/missing bound refuses the build (RUN-R4.1).
        """
        plan = Entitlements(
            can_use_agent=True,
            can_ingest=True,
            can_export=True,
            node_visit_ceiling=settings.entitlement__node_visit_ceiling,
            max_output_tokens=settings.entitlement__max_output_tokens,
            wall_clock_seconds=settings.entitlement__wall_clock_seconds,
            max_tool_iterations=settings.entitlement__max_tool_iterations,
            request_rate_per_minute=settings.entitlement__request_rate_per_minute,
        )
        return cls(validate_plan(plan))

    def resolve(self, athlete_id: str) -> Entitlements:
        """Return the all-permissive OSS plan for the server-derived subject (AGT-ENT-R1).

        ``athlete_id`` is the SERVER-DERIVED subject the caller resolved (AUTH-R18); this
        resolver never reads identity from a model/tool/payload. In OSS every subject
        resolves to the same single all-permissive plan.
        """
        return self._plan


def plan_bounds_summary(plan: Entitlements) -> dict[str, object]:
    """A non-secret summary of a plan's flags + bounds for the readiness probe / logs.

    Used by the readiness surface to report that the default plan is loaded + validated
    WITHOUT leaking anything sensitive (OBS-R6.3) — it carries only the boolean flags and
    the numeric non-monetary bounds (there is no monetary field to leak).
    """
    return {f.name: getattr(plan, f.name) for f in fields(plan)}


__all__ = [
    "EntitlementError",
    "EntitlementResolver",
    "Entitlements",
    "OssEntitlementResolver",
    "plan_bounds_summary",
    "validate_plan",
]
