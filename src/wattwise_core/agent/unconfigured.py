"""The graceful no-LLM agent engine (RUN-R4.1).

Factored out of :mod:`wattwise_core.agent.engine` (QUAL-R9 size split) and re-exported from
it so ``from wattwise_core.agent.engine import UnconfiguredAgentEngine`` stays stable. When the
OSS deployment has no LLM key configured the API binds THIS engine instead of the live
:class:`~wattwise_core.agent.engine.GraphAgentEngine`, so the coaching surface returns a typed,
jargon-free ``degraded`` answer rather than failing the boot or erroring the endpoint.
"""

from __future__ import annotations

from typing import Any, ClassVar

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import AgentAnswer, Readiness


class UnconfiguredAgentEngine:
    """Graceful no-op engine when the OSS deployment has no LLM configured (RUN-R4.1).

    The engine boots without a model; the coaching surface then returns a typed,
    jargon-free ``degraded`` answer (no internals leaked, VOICE-R2/-R3) rather than the
    boot failing or the endpoint erroring. Configuring a model upgrades it in place.
    """

    _MESSAGE: ClassVar[dict[str, str]] = {
        "en": "Coaching isn't switched on for this account yet.",
        "de": "Coaching ist fuer dieses Konto noch nicht aktiviert.",
        "ru": "Trener poka ne podklyuchyon dlya etoy uchyotnoy zapisi.",
    }

    async def answer(
        self,
        *,
        athlete_id: str,
        question: str | None,
        thread_id: str | None,
        response_length: str,
        follow_up: dict[str, Any] | None,
        locale: str,
    ) -> AgentAnswer:
        text = self._MESSAGE.get((locale or "en").split("-", 1)[0].lower(), self._MESSAGE["en"])
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
        text = self._MESSAGE.get((locale or "en").split("-", 1)[0].lower(), self._MESSAGE["en"])
        return Readiness(
            verdict=None,
            status=RunStatus.DEGRADED,
            as_of=None,
            summary_html=f"<p>{text}</p>",
            summary_text=text,
            coverage={"reason": "agent_unconfigured"},
        )


__all__ = ["UnconfiguredAgentEngine"]
