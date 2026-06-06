"""Bounded-retry structured-output helper (STRUCT-R1, STRUCT-R2).

Every verdict the agent emits (intent, retrieval plan, coverage judgement, reflection
decision, grounding-span extraction, plan structure, readiness call) is a
provider-enforced structured output: native JSON-schema-constrained decoding against a
closed typed schema (STRUCT-R1). This module owns the ONE call seam every verdict node
uses to obtain such an output reliably.

:func:`run_structured` wraps a :class:`~wattwise_core.agent.contracts.ChatModel`'s
``structured`` call with a small bounded retry (STRUCT-R2): a transient validation /
provider error is retried up to ``max_attempts`` times, after which a typed
:class:`StructuredOutputError` is raised for the calling node to route per its failure
policy. There is NO free-text-then-parse fallback — regex / brace-matching /
``json.loads`` over a chat string is forbidden for any verdict (STRUCT-R1); when the
provider cannot yield a schema-valid object we fail closed with a typed control event,
never a best-effort guess.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.observability.logging import get_logger

_DEFAULT_MAX_ATTEMPTS = 3

_logger = get_logger(__name__)


class StructuredOutputError(RuntimeError):
    """A verdict's structured output could not be produced within the retry bound.

    Raised by :func:`run_structured` after ``max_attempts`` provider attempts fail to
    yield a schema-valid object (STRUCT-R2). It is a TYPED control event: the calling
    node catches it and routes per its own failure policy (e.g. re-plan, abstain). It is
    NOT a free-text fallback (STRUCT-R1) and MUST NOT be swallowed into a fabricated
    verdict.
    """

    def __init__(self, schema_name: str, attempts: int, cause: Exception | None) -> None:
        self.schema_name = schema_name
        self.attempts = attempts
        super().__init__(
            f"structured output for {schema_name!r} failed after {attempts} attempt(s)"
        )
        if cause is not None:
            self.__cause__ = cause


async def run_structured[M: BaseModel](
    model: ChatModel,
    *,
    system: str,
    data: str,
    schema: type[M],
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> M:
    """Obtain one schema-valid verdict from ``model``, with bounded retry (STRUCT-R2).

    ``schema`` is a closed typed model (STRUCT-R3); ``model.structured`` performs
    provider-enforced JSON-schema-constrained decoding (STRUCT-R1). On a validation or
    provider error the call is retried up to ``max_attempts`` times; if every attempt
    fails the function raises :class:`StructuredOutputError` (a typed control event),
    NEVER a free-text-parsed object.

    Args:
        model: the configured chat model seam (one OpenAI-compatible model, MODEL-R4).
        system: the instruction-region system prompt (trusted).
        data: the delimited untrusted-data region to analyse (INJECT-R1).
        schema: the closed verdict schema to constrain + validate against.
        max_attempts: small bounded retry count (>= 1).

    Returns:
        A validated instance of ``schema``.

    Raises:
        StructuredOutputError: when no schema-valid output is produced in the bound.
        ValueError: when ``max_attempts`` is not positive (programming error).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    schema_name = schema.__name__
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await model.structured(system=system, data=data, schema=schema)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            _logger.warning(
                "structured_output_retry",
                schema=schema_name,
                attempt=attempt,
                max_attempts=max_attempts,
                error_type=type(exc).__name__,
            )
    raise StructuredOutputError(schema_name, max_attempts, last_error)


__all__ = ["StructuredOutputError", "run_structured"]
