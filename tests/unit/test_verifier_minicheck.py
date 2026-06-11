"""Unit tests for the MiniCheck adapter's fail-closed paths (issue #10, GROUND-R11).

The ML stack is an OPERATOR opt-in (never a base dependency), so the offline suite pins
the adapter's behaviour WITHOUT it: a missing backend raises the typed
:class:`VerifierUnavailableError` (which the gate maps to a recorded degradation), and
the label-resolution helper identifies the supported-label probability by NAME only —
refusing a checkpoint whose labels it cannot identify rather than guessing positionally.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from wattwise_core.agent.verifier_minicheck import (
    MiniCheckVerifier,
    VerifierUnavailableError,
    _supported_probability,
)

pytestmark = pytest.mark.unit


async def test_missing_ml_stack_raises_the_typed_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the optional 'transformers' package the adapter fails closed, typed.

    The entailment gate catches this and degrades the run to the deterministic layers
    with a recorded counter — the adapter's job is to raise the TYPED error, never to
    crash with a bare ImportError or to fake a score.
    """
    real_import = builtins.__import__

    def _no_transformers(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_transformers)
    verifier = MiniCheckVerifier("lytang/MiniCheck-RoBERTa-Large")
    with pytest.raises(VerifierUnavailableError):
        await verifier.support(sentence="Your CTL is 84.", facts="ctl: 84")


def test_supported_probability_resolves_by_label_name() -> None:
    """The supported-label score is found by NAME (LABEL_1/supported/entailment), never index."""
    outputs = [[{"label": "LABEL_0", "score": 0.2}, {"label": "LABEL_1", "score": 0.8}]]
    assert _supported_probability(outputs, "m") == pytest.approx(0.8)
    named = [{"label": "supported", "score": 0.7}, {"label": "unsupported", "score": 0.3}]
    assert _supported_probability(named, "m") == pytest.approx(0.7)


def test_unrecognizable_labels_fail_closed() -> None:
    """A checkpoint with unidentifiable labels is refused (no positional guessing)."""
    with pytest.raises(VerifierUnavailableError):
        _supported_probability([{"label": "mystery", "score": 0.9}], "m")
    with pytest.raises(VerifierUnavailableError):
        _supported_probability({"label": "supported", "score": 0.9}, "m")
