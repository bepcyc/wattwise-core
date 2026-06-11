"""MiniCheck-class local entailment verifier (issue #10 Phase 2, proposed GROUND-R11).

The production adapter behind the
:class:`~wattwise_core.agent.grounding_entailment.EntailmentVerifier` seam: a small,
self-hosted grounded fact-checking classifier (MiniCheck — Tang, Laban & Durrett, EMNLP
2024, arXiv:2404.10774) scoring ``P(sentence is supported by facts)``. The default model
id is loaded config (``[agent.entailment]``, CFG-R1a); the MIT-licensed
``lytang/MiniCheck-RoBERTa-Large`` checkpoint is the suggested default — a standard
binary sequence-pair classifier (document, claim), so the adapter works with any
MiniCheck-style checkpoint exposing a supported/unsupported label pair.

Decorrelation is the point (issue #10): this verifier shares no weights with the drafting
model, so it does not inherit the generator/extractor failure correlation the binding bug
exploits. It runs fully offline — no athlete text leaves the process.

The heavy ML stack (``transformers``; installed by the OPERATOR, never a base dependency)
is imported lazily on first use. Any import/load/score failure raises
:class:`VerifierUnavailableError`; the entailment gate maps that to its fail-closed
"degrade to the deterministic layers + record it" path — a missing verifier can never
fail open, and it never crashes an athlete's turn.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
from typing import Any

#: Label names (casefolded) a MiniCheck-style checkpoint uses for "supported"; the
#: adapter resolves the score by LABEL NAME, never by positional guess, and refuses a
#: checkpoint whose labels it cannot identify (fail-closed).
_SUPPORTED_LABELS = frozenset({"1", "label_1", "supported", "entailment", "yes"})


class VerifierUnavailableError(RuntimeError):
    """The entailment verifier cannot run (missing dependency / unloadable checkpoint).

    The gate treats this as a recorded degradation to the deterministic grounding layers
    (issue #10: never silently open) — the caller logs and counts it.
    """


class MiniCheckVerifier:
    """Local MiniCheck-style sequence-pair classifier behind the verifier seam."""

    def __init__(self, model_id: str, *, device: str = "cpu") -> None:
        self._model_id = model_id
        self._device = device
        self._pipeline: Any = None
        self._lock = threading.Lock()

    async def support(self, *, sentence: str, facts: str) -> float:
        """The probability that ``facts`` support ``sentence`` (GROUND-R11 seam).

        Inference runs on a worker thread so the agent's event loop never blocks on a
        local model forward pass.
        """
        return await asyncio.to_thread(self._score, sentence, facts)

    def _score(self, sentence: str, facts: str) -> float:
        pipe = self._ensure_pipeline()
        try:
            outputs = pipe({"text": facts, "text_pair": sentence}, top_k=None, truncation=True)
        except Exception as exc:
            raise VerifierUnavailableError(
                f"entailment verifier {self._model_id} failed to score"
            ) from exc
        return _supported_probability(outputs, self._model_id)

    def _ensure_pipeline(self) -> Any:
        """Build the classification pipeline once (lazy heavy import, fail-closed)."""
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._build_pipeline()
            return self._pipeline

    def _build_pipeline(self) -> Any:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise VerifierUnavailableError(
                "the entailment verifier needs the optional 'transformers' package; "
                "install it in the deployment or disable [agent.entailment]"
            ) from exc
        try:
            return transformers.pipeline(
                "text-classification", model=self._model_id, device=self._device
            )
        except Exception as exc:
            raise VerifierUnavailableError(
                f"entailment verifier checkpoint {self._model_id} failed to load"
            ) from exc


def _supported_probability(outputs: Any, model_id: str) -> float:
    """Resolve the supported-label probability from pipeline output, by NAME only."""
    candidates: Any = outputs
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], list):
        candidates = candidates[0]
    if not isinstance(candidates, list):
        raise VerifierUnavailableError(f"verifier {model_id} returned an unexpected shape")
    for item in candidates:
        label = str(item.get("label", "")).casefold()
        if label in _SUPPORTED_LABELS:
            return float(item.get("score", 0.0))
    raise VerifierUnavailableError(
        f"verifier {model_id} exposes no recognizable supported/unsupported labels"
    )


__all__ = ["MiniCheckVerifier", "VerifierUnavailableError"]
