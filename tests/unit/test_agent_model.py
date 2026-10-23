"""Unit tests for the model-routing seam + structured-output helper (MODEL-R*, STRUCT-R*).

Offline only: the live :class:`OpenAICompatibleModel` is exercised through an injected
in-memory stub client (no network); every other test uses :class:`FakeModel`. Together
they assert: provider-enforced structured decoding at the verdict temperature (MODEL-R5), the
refusal/unparsed fail-closed paths (STRUCT-R1), the bounded retry that raises a typed
:class:`StructuredOutputError` rather than free-text-parsing (STRUCT-R2), and that the
deterministic fake is reproducible for the eval suite.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.agent.model import (
    FakeModel,
    ModelResponseError,
    OpenAICompatibleModel,
)
from wattwise_core.agent.structured import StructuredOutputError, run_structured
from wattwise_core.config import load_settings
from wattwise_core.observability import runtrace

pytestmark = pytest.mark.unit


class _Verdict(BaseModel):
    decision: str
    score: int


class _OtherVerdict(BaseModel):
    label: str


# --------------------------------------------------------------------------- #
# FakeModel — deterministic, network-free                                      #
# --------------------------------------------------------------------------- #


async def test_fake_model_returns_scripted_instance() -> None:
    want = _Verdict(decision="proceed", score=3)
    model = FakeModel(scripted={"_Verdict": want})
    got = await model.structured(system="s", data="d", schema=_Verdict)
    assert got is want


async def test_fake_model_set_response_registers_by_schema_name() -> None:
    model = FakeModel()
    model.set_response(_Verdict(decision="abstain", score=0))
    got = await model.structured(system="s", data="d", schema=_Verdict)
    assert got.decision == "abstain"


async def test_fake_model_unscripted_schema_raises() -> None:
    model = FakeModel(scripted={"_Verdict": _Verdict(decision="x", score=1)})
    with pytest.raises(KeyError):
        await model.structured(system="s", data="d", schema=_OtherVerdict)


async def test_fake_model_compose_returns_canned_prose() -> None:
    model = FakeModel(prose="You're recovered and sharp today.")
    out = await model.compose(system="voice", context="ctx")
    assert out == "You're recovered and sharp today."


def test_fake_model_satisfies_chatmodel_protocol() -> None:
    assert isinstance(FakeModel(), ChatModel)


# --------------------------------------------------------------------------- #
# run_structured — bounded retry, typed failure, no free-text fallback         #
# --------------------------------------------------------------------------- #


async def test_run_structured_happy_path() -> None:
    model = FakeModel(scripted={"_Verdict": _Verdict(decision="proceed", score=2)})
    got = await run_structured(model, system="s", data="d", schema=_Verdict)
    assert got.score == 2


class _FlakyModel:
    """Fails ``fail_times`` then yields a valid instance (drives the retry path)."""

    def __init__(self, *, fail_times: int, instance: BaseModel) -> None:
        self._remaining = fail_times
        self._instance = instance
        self.calls = 0

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise ValueError("transient provider/validation error")
        assert isinstance(self._instance, schema)
        return self._instance

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        return ""


async def test_run_structured_retries_then_succeeds() -> None:
    model = _FlakyModel(fail_times=2, instance=_Verdict(decision="proceed", score=1))
    got = await run_structured(model, system="s", data="d", schema=_Verdict, max_attempts=3)
    assert got.decision == "proceed"
    assert model.calls == 3


async def test_run_structured_exhausts_bound_and_raises_typed_error() -> None:
    model = _FlakyModel(fail_times=99, instance=_Verdict(decision="x", score=0))
    with pytest.raises(StructuredOutputError) as excinfo:
        await run_structured(model, system="s", data="d", schema=_Verdict, max_attempts=2)
    err = excinfo.value
    assert err.schema_name == "_Verdict"
    assert err.attempts == 2
    assert model.calls == 2
    # The typed control event preserves the underlying cause; it is NEVER a parsed guess.
    assert isinstance(err.__cause__, ValueError)


async def test_run_structured_rejects_non_positive_attempts() -> None:
    model = FakeModel(scripted={"_Verdict": _Verdict(decision="x", score=0)})
    with pytest.raises(ValueError, match="max_attempts"):
        await run_structured(model, system="s", data="d", schema=_Verdict, max_attempts=0)


async def test_run_structured_no_free_text_fallback_on_failure() -> None:
    """A perpetual failure yields the typed error, never a fabricated verdict."""
    model = _FlakyModel(fail_times=99, instance=_Verdict(decision="x", score=0))
    with pytest.raises(StructuredOutputError):
        await run_structured(model, system="s", data="d", schema=_Verdict, max_attempts=3)


# --------------------------------------------------------------------------- #
# OpenAICompatibleModel — offline via an injected stub client                  #
# --------------------------------------------------------------------------- #


class _StubMessage:
    def __init__(
        self,
        *,
        parsed: BaseModel | None = None,
        refusal: str | None = None,
        content: str | None = None,
    ) -> None:
        self.parsed = parsed
        self.refusal = refusal
        self.content = content


class _StubCompletion:
    def __init__(self, message: _StubMessage) -> None:
        self.choices = [type("Choice", (), {"message": message})()]


class _RecordingCompletions:
    """Records the kwargs of parse/create and returns canned completions (no network)."""

    def __init__(self, *, parse_message: _StubMessage, create_content: str) -> None:
        self._parse_message = parse_message
        self._create_content = create_content
        self.parse_kwargs: dict[str, Any] | None = None
        self.create_kwargs: dict[str, Any] | None = None

    async def parse(self, **kwargs: Any) -> _StubCompletion:
        self.parse_kwargs = kwargs
        return _StubCompletion(self._parse_message)

    async def create(self, **kwargs: Any) -> _StubCompletion:
        self.create_kwargs = kwargs
        return _StubCompletion(_StubMessage(content=self._create_content))


class _StubClient:
    def __init__(self, completions: _RecordingCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def _dev_settings() -> Any:
    # Development env so load_settings does not require production secrets (fail-closed).
    return load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        api__json_body_max_bytes=262144,
        agent__model="test-model",
        agent__temperature=0.7,
        agent__verdict_temperature=0.2,
        agent__max_output_tokens=256,
    )


def _make_model(completions: _RecordingCompletions) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        settings=_dev_settings(),
        client=_StubClient(completions),  # type: ignore[arg-type]
    )


async def test_openai_model_structured_uses_verdict_temperature_and_schema() -> None:
    """Verdicts decode at the configured verdict temperature with the schema as response_format."""
    want = _Verdict(decision="proceed", score=4)
    completions = _RecordingCompletions(
        parse_message=_StubMessage(parsed=want), create_content="prose"
    )
    model = _make_model(completions)
    got = await model.structured(system="sys", data="untrusted", schema=_Verdict)
    assert got is want
    kwargs = completions.parse_kwargs
    assert kwargs is not None
    # MODEL-R5: verdicts decode at the configured verdict temperature, never the prose one.
    assert kwargs["temperature"] == 0.2
    assert kwargs["response_format"] is _Verdict
    assert kwargs["model"] == "test-model"
    # INJECT-R1: untrusted data is a separate user message, never folded into system.
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["system", "user"]
    assert kwargs["messages"][1]["content"] == "untrusted"


async def test_openai_model_structured_refusal_raises_value_error() -> None:
    completions = _RecordingCompletions(
        parse_message=_StubMessage(refusal="I cannot"), create_content="prose"
    )
    model = _make_model(completions)
    with pytest.raises(ModelResponseError):
        await model.structured(system="s", data="d", schema=_Verdict)
    # The fail-closed path is a ValueError, so run_structured retries it (STRUCT-R2).
    assert issubclass(ModelResponseError, ValueError)


async def test_openai_model_structured_unparsed_raises() -> None:
    completions = _RecordingCompletions(
        parse_message=_StubMessage(parsed=None), create_content="prose"
    )
    model = _make_model(completions)
    with pytest.raises(ModelResponseError):
        await model.structured(system="s", data="d", schema=_Verdict)


async def test_openai_model_refusal_drives_run_structured_to_typed_error() -> None:
    completions = _RecordingCompletions(
        parse_message=_StubMessage(refusal="no"), create_content="prose"
    )
    model = _make_model(completions)
    with pytest.raises(StructuredOutputError):
        await run_structured(model, system="s", data="d", schema=_Verdict, max_attempts=2)


async def test_openai_model_sends_provider_preferences_on_every_call() -> None:
    """Both call paths carry the configured OpenRouter provider preferences.

    The shipped defaults skip Relace and every fp4-class endpoint, whose fp4 serving ignored
    ``reasoning_effort`` and produced 32k-token runaway thinking traces.
    """
    completions = _RecordingCompletions(
        parse_message=_StubMessage(parsed=_Verdict(decision="x", score=1)), create_content="prose"
    )
    model = OpenAICompatibleModel(
        settings=load_settings(
            app__environment="development", database_dsn="sqlite+aiosqlite:///:memory:"
        ),
        client=_StubClient(completions),  # type: ignore[arg-type]
    )
    await model.structured(system="s", data="d", schema=_Verdict)
    await model.compose(system="v", context="c")
    for kwargs in (completions.parse_kwargs, completions.create_kwargs):
        assert kwargs is not None
        provider = kwargs["extra_body"]["provider"]
        assert "relace" in provider["ignore"]
        assert "fp4" not in provider["quantizations"]
        assert "nvfp4" not in provider["quantizations"]


async def test_openai_model_sends_no_provider_preferences_when_unset() -> None:
    """Empty preferences send no ``provider`` field, so non-OpenRouter endpoints are unaffected."""
    completions = _RecordingCompletions(
        parse_message=_StubMessage(parsed=None), create_content="ok"
    )
    model = OpenAICompatibleModel(
        settings=load_settings(
            app__environment="development",
            database_dsn="sqlite+aiosqlite:///:memory:",
            agent__provider__ignore=[],
            agent__provider__quantizations=[],
        ),
        client=_StubClient(completions),  # type: ignore[arg-type]
    )
    await model.compose(system="v", context="c")
    assert completions.create_kwargs is not None
    assert completions.create_kwargs["extra_body"] is None


def test_default_temperatures_are_non_zero_for_reasoning_models() -> None:
    """The shipped defaults never decode greedily: a reasoning model loops at temperature 0.

    At 0 the nightly live leg saw the model spend its whole 32k output budget on a repeating
    thinking trace with no answer, on both compose and structured calls.
    """
    settings = load_settings(
        app__environment="development", database_dsn="sqlite+aiosqlite:///:memory:"
    )
    assert settings.agent__temperature > 0.0
    assert settings.agent__verdict_temperature > 0.0


async def test_openai_model_compose_uses_bounded_temperature() -> None:
    completions = _RecordingCompletions(
        parse_message=_StubMessage(parsed=None), create_content="warm prose"
    )
    model = _make_model(completions)
    out = await model.compose(system="voice", context="ctx", max_tokens=4096)
    assert out == "warm prose"
    kwargs = completions.create_kwargs
    assert kwargs is not None
    # MODEL-R5: prose uses the configured bounded temperature, not 0.
    assert kwargs["temperature"] == 0.7
    # The per-call request is clamped by the configured output ceiling.
    assert kwargs["max_completion_tokens"] == 256


async def test_openai_model_sends_configured_reasoning_effort() -> None:
    """MODEL-R5a: the configured effort reaches the provider on BOTH call paths.

    Tagging it on the span alone left a reasoning model at its provider default effort, which
    exhausted the output budget on the weekly digest and returned empty content.
    """
    completions = _RecordingCompletions(
        parse_message=_StubMessage(parsed=_Verdict(decision="x", score=1)), create_content="prose"
    )
    model = _make_model(completions)
    await model.structured(system="s", data="d", schema=_Verdict)
    await model.compose(system="v", context="c")
    assert completions.parse_kwargs is not None
    assert completions.create_kwargs is not None
    effort = _dev_settings().agent__reasoning_effort
    assert completions.parse_kwargs["reasoning_effort"] == effort
    assert completions.create_kwargs["reasoning_effort"] == effort


async def test_openai_model_compose_raises_on_empty_content() -> None:
    """MODEL-SEAM-1: empty compose content is a failure mode, not a valid empty draft."""
    completions = _RecordingCompletions(parse_message=_StubMessage(parsed=None), create_content="")
    model = _make_model(completions)
    with pytest.raises(ModelResponseError, match="empty compose content"):
        await model.compose(system="v", context="c")


class _SequencedCreate:
    """Returns one scripted ``(content, finish_reason)`` completion per ``create`` call."""

    def __init__(self, script: list[tuple[str, str]]) -> None:
        self._script = list(script)
        self.calls = 0

    async def create(self, **_: Any) -> Any:
        content, finish = self._script[self.calls]
        self.calls += 1
        choice = type("Choice", (), {"message": _StubMessage(content=content)})()
        choice.finish_reason = finish
        return type("Completion", (), {"choices": [choice]})()


def _sequenced_model(script: list[tuple[str, str]]) -> tuple[OpenAICompatibleModel, Any]:
    completions = _SequencedCreate(script)
    model = OpenAICompatibleModel(
        settings=_dev_settings(),
        client=_StubClient(completions),  # type: ignore[arg-type]
    )
    return model, completions


async def test_openai_model_compose_retries_transient_empty_content() -> None:
    """MODEL-R5a: an empty answer NOT cut off at the cap is a provider glitch, retried once."""
    model, completions = _sequenced_model([("", "stop"), ("warm prose", "stop")])
    assert await model.compose(system="v", context="c") == "warm prose"
    assert completions.calls == 2


async def test_openai_model_compose_fails_closed_when_output_cap_reached() -> None:
    """MODEL-R5a: empty content at the output cap is budget exhaustion — no retry."""
    model, completions = _sequenced_model([("", "length"), ("unused", "stop")])
    with pytest.raises(ModelResponseError, match="budget exhausted"):
        await model.compose(system="v", context="c")
    assert completions.calls == 1


async def test_openai_model_compose_fails_after_bounded_empty_retries() -> None:
    model, completions = _sequenced_model([("", "stop"), ("", "stop")])
    with pytest.raises(ModelResponseError, match="finish_reason='stop'"):
        await model.compose(system="v", context="c")
    assert completions.calls == 2


def test_openai_model_satisfies_chatmodel_protocol() -> None:
    completions = _RecordingCompletions(parse_message=_StubMessage(parsed=None), create_content="")
    assert isinstance(_make_model(completions), ChatModel)


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _UsageCompletions:
    """Returns a structured completion carrying a real provider ``usage`` object (no network)."""

    def __init__(self, parsed: BaseModel, usage: _Usage) -> None:
        self._parsed = parsed
        self._usage = usage

    async def parse(self, **kwargs: Any) -> Any:
        """Return a parsed completion whose ``usage`` reports provider token counts."""
        completion = _StubCompletion(_StubMessage(parsed=self._parsed))
        completion.usage = self._usage  # type: ignore[attr-defined]
        return completion


async def test_structured_records_provider_usage_and_computed_cost_on_span() -> None:
    """A model span records model/tier/effort + provider token usage + computed cost (AGT-OBS-R2).

    The cost is COMPUTED from the config-loaded per-million rates (flash default: $0.15 in /
    $0.60 out per M). Reading provider usage instead of discarding it is what this asserts —
    reverting to the prior code (no usage read) leaves the span fields all ``None``.
    """
    settings = load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        agent__model="minimax/minimax-m3.1-flash",
    )
    completions = _UsageCompletions(_Verdict(decision="proceed", score=4), _Usage(1000, 500))
    model = OpenAICompatibleModel(
        settings=settings,
        client=_StubClient(completions),  # type: ignore[arg-type]
    )
    with runtrace.run_trace("athlete:conv"):
        await model.structured(system="s", data="d", schema=_Verdict)
        trace = runtrace.active_trace()
        assert trace is not None
        span = trace.spans[-1]
    assert span.model == "minimax/minimax-m3.1-flash"
    assert span.tier == "flash"
    assert span.reasoning_effort == "low"
    assert span.schema_id == "_Verdict"
    assert span.prompt_tokens == 1000
    assert span.completion_tokens == 500
    # cost = 1000/1e6 * 0.15 + 500/1e6 * 0.60 = 0.00015 + 0.0003 = 0.00045
    assert span.cost_usd == pytest.approx(0.00045)
