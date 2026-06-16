"""The diagnosis + memory READ/ERASE engine methods, factored off the engine (QUAL-R9 size split).

The focused sibling of :mod:`wattwise_core.agent.engine` that owns the DETERMINISTIC, non-graph
engine surfaces the deployable :class:`~wattwise_core.agent.engine.GraphAgentEngine` exposes
alongside the graph-driven deliverables: the data-quality / coverage DIAGNOSIS (API-R15) and the
athlete-scoped memory list / get / delete / erase seam (MEM-R3/-R4). They are split out as a mixin
so the main engine module stays under the size ceiling while these cohesive, model-free surfaces
live in one place (mirroring :mod:`engine_readiness`).

Neither surface routes through the durable checkpointer or the LLM: diagnosis projects the
canonical analytics envelope deterministically (fail-closed, GROUND-R7), and the memory seam is a
scoped relational read/delete over the dedicated agent-state store. Both keep identity
SERVER-DERIVED (AGT-SEC-R1) — the ``athlete_id`` is the authenticated owner and is never widened
from a client argument.

Cited requirements: API-R15, MEM-R1, MEM-R3, MEM-R4, GROUND-R7, OUTCOME-R3/-R4/-R5, AGT-SEC-R1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.agent.deliverables import Digest, Readiness, readiness_assessment
from wattwise_core.agent.diagnose_deliverable import AgentDiagnosis, diagnose_coverage
from wattwise_core.agent.digest_history import digest_history as read_digest_history
from wattwise_core.agent.engine_constraints import ConstraintCaptureMixin
from wattwise_core.agent.engine_memory import (
    delete_memory,
    erase_memory,
    get_memory,
    list_memory,
)
from wattwise_core.agent.engine_readiness import (
    connection_sync_suspect,
    gather_readiness_inputs,
    localized_readiness_narrator,
)
from wattwise_core.agent.engine_services import CoachBundle
from wattwise_core.agent.memory import (
    COACH_NUMERIC_DETAIL_LEVEL_PREF_PREFIX,
    RESPONSE_LENGTH_PREF_PREFIX,
    RESPONSE_LENGTHS,
    MemoryItemKind,
    OssMemoryStore,
    RecalledItem,
    UntrustedMemoryWriteError,
    response_length_from_items,
)
from wattwise_core.agent.state_db import AgentStateDatabase
from wattwise_core.analytics.constants import READINESS_SYNC_STALE_AFTER_DAYS
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.persistence.types import utcnow
from wattwise_core.seams import SessionProvider

# The persisted-default verbosity preference (MEM-R1, VOICE-R8 §382) lives in the AGENT-STATE store
# as a single ``preference``-kind memory item with the stable marker prefix owned by ``memory`` (the
# store), NOT in the canonical master-data store: it is an agent-interaction preference (§382),
# recalled on the run path as the DEFAULT response length when a request carries none. The prefix +
# closed verbosity set + the scan helper are SINGLE-SOURCED in ``memory`` so the run-path default
# and the ``GET /v1/user-settings/response-length`` endpoint resolve verbosity identically.


async def _read_stored_response_length(state_db: AgentStateDatabase, *, athlete_id: str) -> str:
    """The persisted agent-state verbosity default, else ``standard`` (the ONE store read).

    The single agent-state read shared by the run-path default
    (:meth:`DeliverableEngineMixin.resolve_default_response_length`) and the
    ``GET /v1/user-settings/response-length`` surface
    (:meth:`DeliverableEngineMixin.get_response_length_preference`) — so write and read are ONE
    source (the VOICE-R8 §382 store-split). Scans the owner's memory for the single
    ``response_length=`` ``PREFERENCE`` item, falling back closed to ``standard`` when unset.
    """
    async with state_db.session() as session:
        store = OssMemoryStore(session)
        items = await store.fetch_relevant(
            athlete_id=athlete_id, query=RESPONSE_LENGTH_PREF_PREFIX, limit=50
        )
    return response_length_from_items(items)


def _compose_recalled(
    constraints: Sequence[RecalledItem], items: Sequence[RecalledItem]
) -> list[dict[str, object]]:
    """Compose the recalled context: the constraint core tier first, then the pool (MEM-R6).

    The non-evictable active-constraint set is PREPENDED ahead of the keyword/recency pool and
    de-duplicated by id, so a constraint that ALSO surfaced in the pool is not rendered twice and is
    never dropped by ``limit`` (the salience-tier fix for #77 Hole 1). The persisted
    verbosity/numeric-detail markers are internal run-default knobs, not personalization prose, so
    they are kept OUT of the recalled context (VOICE-R2). A CONSTRAINT item additionally carries its
    ``severity`` so the downstream grounding gate (GROUND-R13/R14) can select veto vs caution.
    """
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for item in (*constraints, *items):
        if item.memory_item_id in seen:
            continue
        if item.content.startswith(RESPONSE_LENGTH_PREF_PREFIX) or item.content.startswith(
            COACH_NUMERIC_DETAIL_LEVEL_PREF_PREFIX
        ):
            continue
        seen.add(item.memory_item_id)
        projection: dict[str, object] = {
            "kind": item.kind.value,
            "content": item.content,
            "inferred": item.inferred,
        }
        if item.severity is not None:
            projection["severity"] = item.severity.value
        out.append(projection)
    return out


class _EngineSeams(Protocol):
    """The engine seams the diagnosis/readiness/memory methods read (supplied by the engine).

    The mixin is structural: it depends on the engine-owned ``SessionProvider`` seam (the ONE
    canonical-store choke point of SEAM-R11 / ARCH-R31 — the canonical reads here flow through it,
    never around it), the injected ``ChatModel`` + loaded ``CoachBundle`` the readiness narration
    uses, and the lazily-built dedicated agent-state database the host engine already owns (a
    SEPARATE store, ARCH-R13 — not the canonical store) — no graph / checkpoint coupling.
    """

    _sessions: SessionProvider
    _model: ChatModel
    _coach: CoachBundle

    async def _agent_state_db(self) -> AgentStateDatabase: ...


class DeliverableEngineMixin(ConstraintCaptureMixin):
    """Diagnosis (API-R15) + athlete-scoped memory seam (MEM-R3/-R4) for the engine.

    Mixed into :class:`~wattwise_core.agent.engine.GraphAgentEngine`; every method is
    DETERMINISTIC (no LLM, no graph) and keeps ``athlete_id`` server-derived (AGT-SEC-R1). The
    diagnosis is read-only over the canonical store; the memory methods read/erase the dedicated
    agent-state store, scoped strictly to the owner. The athlete safety-constraint capture surface
    (MEM-R7 / GROUND-R14) is inherited from :class:`ConstraintCaptureMixin` (QUAL-R9 size split).
    """

    async def diagnose(
        self: _EngineSeams, *, athlete_id: str, locale: str = "en"
    ) -> AgentDiagnosis:
        """Narrate canonical data-quality / coverage for the athlete (API-R15, fail-closed).

        DETERMINISTIC: probes each canonical analytic input through the analytics service and
        projects the typed ``Computed``/``Unavailable`` envelope into per-input coverage lines
        (present/missing/stale) with NO model call and NO retrieval planner, so there is nothing to
        fabricate (GROUND-R7 / OUTCOME-R5). Degrades visibly when the athlete has no usable
        canonical coverage at all (OUTCOME-R3). Read-only; no agent-state pool opened. ``locale`` is
        accepted for the API copy boundary; the deliverable carries no athlete-facing numbers
        (VOICE-R7).
        """
        async with self._sessions.session(subject=athlete_id) as session:
            return await diagnose_coverage(AnalyticsService(session), athlete_id)

    async def readiness(
        self: _EngineSeams,
        *,
        athlete_id: str,
        locale: str = "en",
        response_length: str = "standard",
    ) -> Readiness:
        """Build the readiness/form deliverable from canonical inputs (QA-EVAL-R2.4).

        Gathers the readiness inputs DETERMINISTICALLY (the fixed readiness JTBD does NOT route
        through the retrieval planner) then drives :func:`readiness_assessment` with the same
        model-backed narrator + canonical grounder the answers use; the delivered verdict is always
        the deterministic oracle's (canonical wins), numbers surface only as grounded citations.
        Readiness does NOT route through the durable checkpointer (a single deterministic
        assessment, not a resumable conversation), so no agent-state pool is opened here.

        The narrator speaks the requested language via the any-language DIRECTIVE (issue #17,
        LANG-R1/-R3): see :func:`localized_readiness_narrator`.
        """
        async with self._sessions.session(subject=athlete_id) as session:
            svc = AnalyticsService(session)
            # The MNAR disambiguator (issue #12): read whether a connector that should be
            # delivering is broken/stalled, so the freshness gate fires on missing data, not rest.
            sync_suspect = await connection_sync_suspect(
                session,
                athlete_id,
                reference_date=utcnow().date(),
                sync_stale_after_days=READINESS_SYNC_STALE_AFTER_DAYS,
            )
            inputs = await gather_readiness_inputs(svc, athlete_id, sync_suspect=sync_suspect)
            return await readiness_assessment(
                athlete_id,
                form=inputs.form,
                as_of=inputs.as_of,
                hrv_rmssd=inputs.hrv_rmssd,
                hrv_baseline=inputs.hrv_baseline,
                sufficiency=inputs.sufficiency,
                # LANG-R1/-R3 (issue #17): the narrator's system prompt carries the run locale's
                # any-language DIRECTIVE (compose_system) — NOT an enumerated language pack.
                narrate=localized_readiness_narrator(self._model, self._coach, locale),
                grounder=self._coach.grounder(self._model, svc),
                response_length=response_length,  # type: ignore[arg-type]
            )

    async def list_memory(
        self: _EngineSeams, *, athlete_id: str, limit: int = 50, offset: int = 0
    ) -> Sequence[RecalledItem]:
        """List the athlete's durable memory rows, newest first, paginated (MEM-R3/-R4).

        The read seam over the dedicated agent-state memory table, scoped STRICTLY to the
        server-derived owner ``athlete_id`` (MEM-R3 / AGT-SEC-R1) — another athlete's rows are never
        listed. Returns personalization context only, never a canonical number (MEM-R1).
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await list_memory(session, athlete_id=athlete_id, limit=limit, offset=offset)

    async def get_memory(
        self: _EngineSeams, *, athlete_id: str, memory_item_id: str
    ) -> RecalledItem | None:
        """Fetch ONE memory row by id, scoped to the owner, else ``None`` (MEM-R3, fail-closed).

        Looks up by BOTH the id AND the server-derived ``athlete_id`` (AGT-SEC-R1): a foreign /
        unknown / non-UUID id returns ``None`` and is never disclosed (the router maps that to a
        404, indistinguishable from truly absent).
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await get_memory(session, athlete_id=athlete_id, memory_item_id=memory_item_id)

    async def delete_memory(self: _EngineSeams, *, athlete_id: str, memory_item_id: str) -> bool:
        """Delete ONE memory row by id, scoped to the owner; True iff erased (MEM-R3 erasure).

        Privacy MUST (PRIV-R8 / CKPT-R8): the guarded delete matches BOTH the id AND the
        server-derived ``athlete_id``, so a cross-athlete / unknown id erases nothing and returns
        ``False`` (router -> 404). The session commits the delete (or rolls back on error).
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await delete_memory(
                session, athlete_id=athlete_id, memory_item_id=memory_item_id
            )

    async def erase_memory(self: _EngineSeams, *, athlete_id: str) -> int:
        """Erase ALL of the athlete's memory rows; returns the count (MEM-R3 erasure / PRIV-R8).

        The whole-athlete erasure scoped to the server-derived owner only, never widening to another
        identity. Returns how many rows were removed so the endpoint reports it.
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await erase_memory(session, athlete_id=athlete_id)

    # --- MEM-R4 run-path recall + episode write (the ONE MemoryStore seam) ---

    async def recall_memory_for_run(
        self: _EngineSeams, *, athlete_id: str, query: str
    ) -> list[dict[str, object]]:
        """Recall durable personalization memory for a coaching run (MEM-R4/MEM-R6).

        The run-path RECALL half of the ONE athlete-scoped MemoryStore seam (MEM-R4): it queries the
        OSS relational store (recency/keyword) scoped STRICTLY to the server-derived owner (MEM-R3),
        returning a plain serializable projection (``kind``/``content``/``inferred``) the engine
        injects so the agent personalizes its answer (MEM-R1/-R2) — never a canonical number.
        The FULL active-constraint core tier is PREPENDED and de-duplicated (MEM-R6 salience tier,
        ADR 0008 §3) by :func:`_compose_recalled`: a standing constraint is never evicted by usage.
        """
        now = utcnow()
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            store = OssMemoryStore(session)
            constraints = await store.fetch_active_constraints(athlete_id=athlete_id, now=now)
            items = await store.fetch_relevant(athlete_id=athlete_id, query=query, limit=8)
        return _compose_recalled(constraints, items)

    async def record_run_episode(self: _EngineSeams, *, athlete_id: str, content: str) -> None:
        """Record a completed coaching turn as a durable episode (MEM-R4 write-episode).

        The run-path WRITE half of the SAME seam: after a COMPLETED run the engine preserves the
        athlete's own request as a raw episode (MEM-R2) through ``write_episode``, scoped to the
        server-derived owner (MEM-R3). The write is ``trusted`` because the content is the
        authenticated athlete's OWN words, never source-synced/scraped text (MEM-R3/INJECT-R3). A
        blank request records nothing. The write is best-effort: a refusal/error never fails the
        already-delivered answer (the episode is personalization, not the answer itself).
        """
        text = content.strip()
        if not text:
            return
        state_db = await self._agent_state_db()
        try:
            async with state_db.session() as session:
                store = OssMemoryStore(session)
                await store.write_episode(
                    athlete_id=athlete_id,
                    kind=MemoryItemKind.PLAN_HISTORY,
                    content=text,
                    trusted=True,
                )
        except UntrustedMemoryWriteError:
            # Owner-originated content is trusted, so this should not arise; if it ever does the
            # write is refused fail-closed and the delivered answer is unaffected (MEM-R3).
            return

    async def resolve_default_response_length(
        self: _EngineSeams, *, athlete_id: str, requested: str | None
    ) -> str:
        """Resolve the run's response length, applying the PERSISTED default (MEM-R1 / VOICE-R8).

        A per-request value WINS for this single call WITHOUT mutating any stored default (R8).
        When NONE is given the engine reads the athlete's PERSISTED verbosity preference — held in
        the AGENT-STATE store as a ``preference``-kind memory item (MEM-R1, §382: an agent-
        interaction preference, NOT a canonical master-data entity) — and applies it as the default,
        mirroring how language resolves (LANG-R2). With no persisted preference it falls back to
        ``standard`` (VOICE-R8 default); an unrecognized persisted value also falls back closed.
        """
        if requested in RESPONSE_LENGTHS:
            return requested
        state_db = await self._agent_state_db()
        return await _read_stored_response_length(state_db, athlete_id=athlete_id)

    async def get_response_length_preference(self: _EngineSeams, *, athlete_id: str) -> str:
        """Read the athlete's PERSISTED verbosity default, else ``standard`` (VOICE-R8 / §8.10).

        The READ half of the ``GET /v1/user-settings/response-length`` contract (doc 60 §8.10):
        returns the stored agent-state preference (MEM-R1, §382 — NOT canonical master-data) or the
        spec default ``standard`` when unset. Reads EXACTLY the same agent-state preference the run
        path applies as its default (via :func:`_read_stored_response_length`) — a single source of
        truth (the VOICE-R8 store-split fix).
        """
        state_db = await self._agent_state_db()
        return await _read_stored_response_length(state_db, athlete_id=athlete_id)

    async def set_response_length_preference(
        self: _EngineSeams, *, athlete_id: str, value: str
    ) -> None:
        """Persist the athlete's verbosity default into the AGENT-STATE store (VOICE-R8 / §8.10).

        The WRITE half of the ``PUT /v1/user-settings/response-length`` contract (doc 60 §8.10):
        UPSERTS the single ``preference``-kind memory item the run-path read scans (MEM-R1, §382 —
        the agent-state store, NOT the canonical §3 master-data the dropped ``Athlete`` column held)
        so the persisted preference actually reaches the run as its default. One preference row,
        never duplicated; ``value`` is the caller-validated closed token (short/standard/detailed).
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            store = OssMemoryStore(session)
            await store.upsert_preference(
                athlete_id=athlete_id,
                marker=RESPONSE_LENGTH_PREF_PREFIX,
                content=f"{RESPONSE_LENGTH_PREF_PREFIX}{value}",
            )

    async def digest_history(
        self: _EngineSeams, *, athlete_id: str, limit: int = 50, before_week_end: str | None = None
    ) -> list[Digest]:
        """The stored weekly-review history, newest first, keyset-paged (API-R14).

        Reads the agent-state store's recorded grounded reviews VERBATIM (GROUND-R7 — no
        recomputation); ``before_week_end`` is the exclusive keyset bound the router's
        signed cursor carries (PAGE-R5). Owner-scoped (AGT-SEC-R1).
        """
        state_db = await self._agent_state_db()
        return await read_digest_history(
            state_db, athlete_id=athlete_id, limit=limit, before_week_end=before_week_end
        )


__all__ = ["DeliverableEngineMixin"]
