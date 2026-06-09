"""The graceful no-LLM agent engine (RUN-R4.1).

Factored out of :mod:`wattwise_core.agent.engine` (QUAL-R9 size split) and re-exported from
it so ``from wattwise_core.agent.engine import UnconfiguredAgentEngine`` stays stable. When the
OSS deployment has no LLM key configured the API binds THIS engine instead of the live
:class:`~wattwise_core.agent.engine.GraphAgentEngine`, so the coaching surface returns a typed,
jargon-free ``degraded`` answer rather than failing the boot or erroring the endpoint.

The LLM-shaped surfaces (``answer`` / ``readiness`` / ``digest``) degrade VISIBLY — a typed,
localized "not switched on" body, never a guessed number (GROUND-R7). The DETERMINISTIC,
non-LLM surfaces MUST still work without a model (RUN-R4.1): the data-quality ``diagnose``
projects the canonical analytics coverage envelope with no model call (API-R15), and the
per-item memory list / get / delete / erase seam (MEM-R3, a privacy MUST that can NEVER depend
on an LLM, PRIV-R8) reads/erases the dedicated agent-state store. Both keep identity
SERVER-DERIVED (AGT-SEC-R1). The canonical ``Database`` (read-only for diagnose) and the
dedicated agent-state store are injected so these surfaces bind in the real ``create_app()``
no-LLM path exactly as the live engine does; the agent-state store lazily falls back to a
per-process FILE-sqlite store (a REAL pool, never ``:memory:``) when none is injected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import AgentAnswer, Digest, Readiness
from wattwise_core.agent.diagnose_deliverable import AgentDiagnosis, diagnose_coverage
from wattwise_core.agent.engine_memory import (
    delete_memory,
    erase_memory,
    get_memory,
    list_memory,
)
from wattwise_core.agent.memory import RecalledItem
from wattwise_core.agent.state_db import (
    AgentStateDatabase,
    build_agent_state_database,
    fallback_state_dsn,
)
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.entitlement import Entitlements
from wattwise_core.persistence import Database


class UnconfiguredAgentEngine:
    """Graceful no-model engine when the OSS deployment has no LLM configured (RUN-R4.1).

    The engine boots without a model; the LLM-shaped coaching surfaces (``answer`` /
    ``readiness`` / ``digest``) return a typed, jargon-free ``degraded`` body (no internals
    leaked, VOICE-R2/-R3) rather than the boot failing or the endpoint erroring. The
    DETERMINISTIC surfaces still work with no model: ``diagnose`` narrates canonical coverage
    (API-R15) and the per-item memory read/erase seam (MEM-R3 / PRIV-R8) operates over the
    dedicated agent-state store — a privacy MUST that can never require an LLM. Configuring a
    model upgrades the deployment to the live :class:`~wattwise_core.agent.engine.GraphAgentEngine`
    in place.
    """

    _MESSAGE: ClassVar[dict[str, str]] = {
        "en": "Coaching isn't switched on for this account yet.",
        "de": "Coaching ist fuer dieses Konto noch nicht aktiviert.",
        "ru": "Trener poka ne podklyuchyon dlya etoy uchyotnoy zapisi.",
    }

    def __init__(
        self,
        database: Database | None = None,
        *,
        state_db: AgentStateDatabase | None = None,
    ) -> None:
        """Bind the canonical DB (read-only, for ``diagnose``) + the agent-state store seams.

        ``database`` powers the DETERMINISTIC diagnosis (read-only over canonical analytics);
        ``state_db`` is the dedicated agent-state store the non-LLM memory seam reads/erases.
        Both default to ``None`` so historical no-arg construction (the LLM-shaped fallback) keeps
        working: the memory/diagnose surfaces lazily build a per-process file-sqlite agent-state
        store (a REAL pool, not ``:memory:``) and only require ``database`` when ``diagnose`` is
        actually called (else they fail closed rather than fabricate).
        """
        self._db = database
        self._state_db = state_db

    def _message(self, locale: str) -> str:
        """The localized "not switched on" copy for the requested locale, else English."""
        return self._MESSAGE.get((locale or "en").split("-", 1)[0].lower(), self._MESSAGE["en"])

    async def _agent_state_db(self) -> AgentStateDatabase:
        """The dedicated agent-state store, lazily built on a REAL file-sqlite pool (ARCH-R13).

        An injected ``state_db`` (production / the real ``create_app`` no-LLM path) is used as-is.
        The lazy fallback builds a per-process FILE-sqlite store on its own real pool (NEVER
        ``:memory:``, which a single connection can't model) so the non-LLM memory seam works even
        when no dedicated store is wired.
        """
        if self._state_db is None:
            self._state_db = build_agent_state_database(dsn=fallback_state_dsn())
            await self._state_db.create_all()
        return self._state_db

    async def answer(
        self,
        *,
        athlete_id: str,
        question: str | None,
        thread_id: str | None,
        response_length: str,
        follow_up: dict[str, Any] | None,
        locale: str,
        entitlement: Entitlements | None = None,
    ) -> AgentAnswer:
        # ``entitlement`` (MED-2) is accepted for seam-parity with the live engine but unused here:
        # the no-LLM fallback runs no graph, so there are no bounds to read — it always degrades.
        text = self._message(locale)
        return AgentAnswer(
            status=RunStatus.DEGRADED,
            thread_id=thread_id or "unconfigured",
            answer_html=f"<p>{text}</p>",
            answer_text=text,
            coverage_caveat={"reason": "agent_unconfigured"},
        )

    async def readiness(
        self, *, athlete_id: str, locale: str = "en", response_length: str = "standard"
    ) -> Readiness:
        """Typed graceful readiness when no LLM is configured (RUN-R4.1, mirrors :meth:`answer`).

        No model and no canonical read: returns an abstaining :class:`Readiness` with no
        verdict and a jargon-free "not switched on" state sentence (no internals leaked,
        VOICE-R2/-R3), so the readiness endpoint degrades gracefully rather than erroring.
        """
        text = self._message(locale)
        return Readiness(
            verdict=None,
            status=RunStatus.DEGRADED,
            as_of=None,
            summary_html=f"<p>{text}</p>",
            summary_text=text,
            coverage={"reason": "agent_unconfigured"},
        )

    async def diagnose(self, *, athlete_id: str, locale: str = "en") -> AgentDiagnosis:
        """DETERMINISTIC data-quality / coverage narration — works with NO LLM (API-R15).

        The diagnosis projects the canonical analytics ``Computed``/``Unavailable`` envelope
        deterministically (no model call, nothing to fabricate, GROUND-R7), so it MUST work on a
        no-LLM deployment exactly as the live engine's does. It reads the SAME canonical store
        through :func:`diagnose_coverage`. ``locale`` is accepted for the API copy boundary; the
        deliverable carries no athlete-facing numbers (VOICE-R7). With no canonical ``database``
        wired this fails closed (an unwired engine never invents coverage).
        """
        if self._db is None:  # pragma: no cover - production always wires the canonical DB
            raise RuntimeError("diagnose requires the canonical database (RUN-R4.1)")
        async with self._db.session() as session:
            return await diagnose_coverage(AnalyticsService(session), athlete_id)

    async def digest(
        self, *, athlete_id: str, week_end: str, entitlement: Entitlements | None = None
    ) -> Digest:
        """A DEGRADED weekly digest when no LLM is configured (RUN-R4.1, abstains visibly).

        No model means no grounded weekly review can be composed, so the digest abstains VISIBLY
        (``degraded`` + the typed ``agent_unconfigured`` caveat) rather than guessing a number
        (OUTCOME-R3/-R4, GROUND-R7). It carries the localized "not switched on" copy and no
        observations/citations — never a fabricated weekly summary.
        """
        text = self._message("en")
        return Digest(
            status=RunStatus.DEGRADED,
            thread_id=f"{athlete_id}:digest:{week_end}",
            week_end=week_end,
            digest_html=f"<p>{text}</p>",
            digest_text=text,
            coverage_caveat={"reason": "agent_unconfigured"},
        )

    async def list_memory(
        self, *, athlete_id: str, limit: int = 50, offset: int = 0
    ) -> Sequence[RecalledItem]:
        """List the athlete's durable memory rows — NON-LLM (MEM-R3/-R4, works with no model).

        The memory surface is outside the agent cost gate and never requires a model; it reads the
        dedicated agent-state store scoped STRICTLY to the server-derived owner (MEM-R3 /
        AGT-SEC-R1) so it functions identically whether or not an LLM is configured.
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await list_memory(session, athlete_id=athlete_id, limit=limit, offset=offset)

    async def get_memory(
        self, *, athlete_id: str, memory_item_id: str
    ) -> RecalledItem | None:
        """Fetch ONE memory row by id, owner-scoped, else ``None`` — NON-LLM (MEM-R3)."""
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await get_memory(session, athlete_id=athlete_id, memory_item_id=memory_item_id)

    async def delete_memory(self, *, athlete_id: str, memory_item_id: str) -> bool:
        """Erase ONE memory row by id, owner-scoped — NON-LLM, a privacy MUST (MEM-R3 / PRIV-R8).

        Per-item erasure can NEVER depend on an LLM: the guarded delete matches BOTH the id AND the
        server-derived ``athlete_id`` over the dedicated agent-state store, so it works on a no-LLM
        deployment and a cross-athlete / unknown id erases nothing (router -> 404). The session
        commits the delete.
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await delete_memory(
                session, athlete_id=athlete_id, memory_item_id=memory_item_id
            )

    async def erase_memory(self, *, athlete_id: str) -> int:
        """Erase ALL the athlete's memory rows; returns the count — NON-LLM (MEM-R3 / PRIV-R8)."""
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await erase_memory(session, athlete_id=athlete_id)


__all__ = ["UnconfiguredAgentEngine"]
