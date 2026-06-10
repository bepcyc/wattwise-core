"""Construction helpers for the deployable agent engine (QUAL-R9 size split of :mod:`engine`).

The focused sibling that owns the two factory functions that BUILD a
:class:`~wattwise_core.agent.engine.GraphAgentEngine`: ``build_agent_engine`` (from settings + the
database, loading the §16 coach-config + the config-resolved OSS entitlement) and
``build_agent_engine_with_model`` (the injected-model / FakeModel seam the tests use). They live
here so the main ``engine`` module stays under the QUAL-R9 module ceiling; ``engine`` re-exports
both so every historical ``from wattwise_core.agent.engine import build_agent_engine`` path stays
stable.

Cited: RUN-R4.1, ARCH-R13, DEPLOY-R4, AGT-ENT-R1, AGT-ENT-R4, MODEL-R5a, MED-2, QUAL-R9.
"""

from __future__ import annotations

from typing import Any

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.agent.engine import GraphAgentEngine
from wattwise_core.agent.engine_services import CoachBundle
from wattwise_core.agent.model import OpenAICompatibleModel
from wattwise_core.agent.state_db import (
    AgentStateDatabase,
    build_agent_state_database,
)
from wattwise_core.entitlement import OssEntitlementResolver
from wattwise_core.identity import OWNER_SUBJECT
from wattwise_core.persistence import Database


def build_agent_engine(database: Database, settings: Any) -> GraphAgentEngine | None:
    """Build the production engine from settings, or ``None`` when no model is configured.

    The OSS engine boots without an LLM key (RUN-R4.1 does not require one); when the key is
    absent this returns ``None`` and the API leaves the agent endpoints surfacing a typed,
    jargon-free unavailable rather than failing the whole boot. When a model IS configured the
    engine is wired with a DEDICATED agent-state database (its own engine/pool, ARCH-R13/DEPLOY-R4)
    so the durable checkpointer never contends with the canonical pool (SPIKE-3 deadlock-freedom).

    The config-resolved OSS entitlement (the all-permissive plan with the config-loaded
    non-monetary bounds, AGT-ENT-R4) is resolved here once and becomes the engine's DEFAULT
    authority: the model's per-call output budget is sized to its token bound (AGT-ENT-R1,
    MODEL-R5a) and the engine reads its ceiling / tool-iteration / wall-clock guards from it. The
    API may still thread a per-REQUEST entitlement into a deliverable call to override it (MED-2).
    The config-loaded CKPT-R4 dedup window (CFG-R1a, never baked) wires the idempotent run path.
    """
    if settings.llm_api_key is None:
        return None
    state_db = build_agent_state_database(settings)
    entitlement = OssEntitlementResolver.from_settings(settings).resolve(OWNER_SUBJECT)
    return GraphAgentEngine(
        database,
        OpenAICompatibleModel(settings=settings, max_output_tokens=entitlement.max_output_tokens),
        state_db=state_db,
        coach=CoachBundle.from_settings(settings),
        entitlement=entitlement,
        dedup_window_seconds=settings.agent__idempotency_dedup_window_seconds,
    )


def build_agent_engine_with_model(
    database: Database,
    model: ChatModel,
    *,
    state_db: AgentStateDatabase | None = None,
    coach: CoachBundle | None = None,
) -> GraphAgentEngine:
    """Build the engine with an injected model + optional ``state_db``/``coach`` (FakeModel seam).

    The durable tests pass a REAL pooled ``state_db`` (file-sqlite/PG/MariaDB); when omitted the
    engine lazily builds the per-process file-sqlite fallback. ``coach`` injects a §16 coach-config
    (the live test passes the loaded bundle; deterministic FakeModel tests pass the empty default,
    since FakeModel scripts exact canonical claims needing no prompt steering or equivalence).
    """
    return GraphAgentEngine(database, model, state_db=state_db, coach=coach)


__all__ = ["build_agent_engine", "build_agent_engine_with_model"]
