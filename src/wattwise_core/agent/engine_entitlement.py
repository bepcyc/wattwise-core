"""Per-run entitlement resolution + model sizing for the engine (MED-2 / AGT-ENT-R1).

The QUAL-R9 size split of :mod:`wattwise_core.agent.engine`: the pure helpers that pick the
entitlement governing one run and re-size the model's output budget from it. Behavior is
identical to the former engine methods; the engine delegates here.
"""

from __future__ import annotations

from typing import cast

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.entitlement import Entitlements


def effective_entitlement(default: Entitlements, per_request: Entitlements | None) -> Entitlements:
    """The entitlement that governs THIS run: the per-request one if supplied, else the default.

    MED-2 resolve -> attach -> check: the API attaches the per-request resolved entitlement and
    threads it into the deliverable methods; when present it is the authority for this run. When
    ``None`` (a direct OSS/test caller) the config-resolved default governs — identical in OSS,
    but the seam is REAL end to end so the commercial layer can vary the plan per request.
    """
    return per_request if per_request is not None else default


def sized_model(model: ChatModel, entitlement: Entitlements) -> ChatModel:
    """The model whose per-call output budget is the entitlement's token bound (AGT-ENT-R1).

    When the model supports re-sizing (``with_output_budget``) and the carried entitlement names
    a positive ``max_output_tokens``, return a view sized to that bound; otherwise the model is
    used as-is (a FakeModel / a bare grant keeps its construction-time budget).
    """
    budget = entitlement.max_output_tokens
    resize = getattr(model, "with_output_budget", None)
    if resize is not None and isinstance(budget, int) and budget > 0:
        return cast(ChatModel, resize(budget))
    return model
