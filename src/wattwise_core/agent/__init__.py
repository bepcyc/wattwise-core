"""Trustworthy LangGraph coaching agent (doc 50, Principle C).

Typed graph state + durable checkpointing, provider-enforced structured outputs,
deterministic fail-closed grounding, and an offline eval suite. Phase-1 deliverables:
grounded Q&A + the weekly digest (weekly load review). Shared seams live in
:mod:`wattwise_core.agent.contracts`.
"""

from __future__ import annotations

from wattwise_core.agent.contracts import AgentState, RunStatus

__all__ = ["AgentState", "RunStatus"]
