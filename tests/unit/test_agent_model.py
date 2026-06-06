"""Unit tests for the model-routing seam + structured-output helper (MODEL-R*, STRUCT-R*).

Offline only: the live :class:`OpenAICompatibleModel` is exercised through an injected
in-memory stub client (no network); every other test uses :class:`FakeModel`. Together
they assert: provider-enforced structured decoding at temperature 0 (MODEL-R5), the
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
        agent__model="test-model",
        agent__temperature=0.7,
        agent__max_output_tokens=256,
    )


def _make_model(completions: _RecordingCompletions) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        settings=_dev_settings(),
        client=_StubClient(completions),  # type: ignore[arg-type]
    )


async def test_openai_model_structured_uses_zero_temperature_and_schema() -> None:
    want = _Verdict(decision="proceed", score=4)
    completions = _RecordingCompletions(
        parse_message=_StubMessage(parsed=want), create_content="prose"
    )
    model = _make_model(completions)
    got = await model.structured(system="sys", data="untrusted", schema=_Verdict)
    assert got is want
    kwargs = completions.parse_kwargs
    assert kwargs is not None
    # MODEL-R5: verdicts decode at temperature 0; the schema is the response_format.
    assert kwargs["temperature"] == 0.0
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


async def test_openai_model_compose_handles_empty_content() -> None:
    completions = _RecordingCompletions(parse_message=_StubMessage(parsed=None), create_content="")
    model = _make_model(completions)
    out = await model.compose(system="v", context="c")
    assert out == ""


def test_openai_model_satisfies_chatmodel_protocol() -> None:
    completions = _RecordingCompletions(parse_message=_StubMessage(parsed=None), create_content="")
    assert isinstance(_make_model(completions), ChatModel)
