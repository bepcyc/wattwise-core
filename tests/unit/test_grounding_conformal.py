"""Unit tests for split-conformal threshold calibration (issue #10, proposed GROUND-R12).

The math under the guarantee: with ``tau`` the ``ceil((n+1)(1-alpha))``-th smallest
per-example exceedance, publishing sentences scoring above ``tau`` bounds the chance a
new deliverable ships any unsupported sentence by ``alpha`` (split conformal, finite
sample). Tests pin the quantile arithmetic, the fail-closed small-``n`` floor, the
group-conditional split, and the fail-closed artifact loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wattwise_core.agent.grounding_conformal import (
    CalibrationError,
    CalibrationProvenance,
    CalibrationRecord,
    conformal_threshold,
    conformal_thresholds,
    load_calibration,
    prompt_sha256,
)
from wattwise_core.agent.grounding_entailment import gate_from_settings

pytestmark = pytest.mark.unit


def _record(
    example: str, score: float, *, supported: bool, group: str = "number"
) -> CalibrationRecord:
    return CalibrationRecord(example_id=example, group=group, score=score, supported=supported)


_CLAIM_PROMPT = "Extract every factual numeric claim."
_PROVENANCE = CalibrationProvenance(
    model_id="lytang/MiniCheck-RoBERTa-Large",
    claim_prompt_sha256=prompt_sha256(_CLAIM_PROMPT),
    dataset_version="grounding-corpus-v1",
)


def _write_artifact(
    path: Path,
    records: list[dict[str, Any]],
    *,
    provenance: dict[str, Any] | None | str = "match",
) -> Path:
    """Write a calibration artifact; ``provenance='match'`` stamps the expected pin."""
    payload: dict[str, Any] = {"records": records}
    if provenance == "match":
        payload["provenance"] = {
            "model_id": _PROVENANCE.model_id,
            "claim_prompt_sha256": _PROVENANCE.claim_prompt_sha256,
            "dataset_version": _PROVENANCE.dataset_version,
        }
    elif provenance is not None:
        payload["provenance"] = provenance
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_threshold_is_the_conformal_quantile_of_example_exceedances() -> None:
    """The threshold is the ceil((n+1)(1-alpha))-th smallest per-example exceedance.

    Nine examples whose worst-unsupported scores are 0.1..0.9 with alpha=0.2: the rank is
    ceil(10 * 0.8) = 8, so tau is the 8th smallest (0.8) — sentences must score ABOVE it
    to publish.
    """
    records = [_record(f"e{i}", i / 10, supported=False) for i in range(1, 10)]
    assert conformal_threshold(records, alpha=0.2) == pytest.approx(0.8)


def test_supported_only_examples_have_zero_exceedance() -> None:
    """An example with NO unsupported sentence contributes exceedance 0 (clean evidence).

    With every calibration example clean, the quantile sits at 0 — every positive score
    publishes, exactly the behaviour a perfect verifier history justifies.
    """
    records = [
        _record("e1", 0.99, supported=True),
        _record("e2", 0.95, supported=True),
        _record("e3", 0.97, supported=True),
        _record("e4", 0.96, supported=True),
    ]
    assert conformal_threshold(records, alpha=0.5) == 0.0


def test_too_few_examples_for_alpha_fails_closed_to_one() -> None:
    """Insufficient calibration data certifies nothing (tau = 1.0, fail-closed).

    With n=2 and alpha=0.05 the conformal rank exceeds n — no finite-sample bound exists,
    so the threshold pins to 1.0 and no sentence can certify against the guarantee. More
    calibration data, not a looser bound, is the way out.
    """
    records = [_record("e1", 0.2, supported=False), _record("e2", 0.3, supported=False)]
    assert conformal_threshold(records, alpha=0.05) == 1.0


def test_empty_calibration_group_fails_closed_to_one() -> None:
    """A group with NO calibration records yields the fail-closed 1.0 threshold."""
    assert conformal_threshold([], alpha=0.5) == 1.0


def test_worst_unsupported_score_per_example_drives_the_bound() -> None:
    """Within one example only the WORST unsupported score matters (max-exceedance).

    A deliverable publishes a bad sentence iff its highest-scoring unsupported sentence
    clears the threshold, so the per-example statistic is the max.
    """
    records = [
        _record("e1", 0.2, supported=False),
        _record("e1", 0.7, supported=False),
        _record("e1", 0.99, supported=True),
        _record("e2", 0.4, supported=False),
    ]
    # Exceedances: e1 -> 0.7, e2 -> 0.4; alpha=0.5 -> rank ceil(3*0.5)=2 -> 0.7.
    assert conformal_threshold(records, alpha=0.5) == pytest.approx(0.7)


def test_calibrated_threshold_bounds_the_empirical_violation_rate() -> None:
    """On an exchangeable holdout the violation rate respects the alpha bound (coverage).

    A deterministic population of exceedances split into calibration/holdout halves: the
    threshold from the calibration half must keep the holdout's exceedance rate within
    alpha plus the finite-sample slack (1/(n+1)).
    """
    population = [(i % 100) / 100 for i in range(200)]
    calibration = [
        _record(f"c{i}", score, supported=False) for i, score in enumerate(population[::2])
    ]
    holdout = population[1::2]
    alpha = 0.2
    tau = conformal_threshold(calibration, alpha=alpha)
    violations = sum(1 for score in holdout if score > tau) / len(holdout)
    assert violations <= alpha + 1 / (len(calibration) + 1)


def test_group_conditional_thresholds_are_independent() -> None:
    """Each claim class calibrates on ITS OWN records (group-conditional validity)."""
    records = [
        *[_record(f"n{i}", 0.2, supported=False, group="number") for i in range(9)],
        *[_record(f"s{i}", 0.9, supported=False, group="statement") for i in range(9)],
    ]
    thresholds = conformal_thresholds(records, alpha=0.2, groups=("number", "statement"))
    assert thresholds["number"] == pytest.approx(0.2)
    assert thresholds["statement"] == pytest.approx(0.9)


def test_invalid_alpha_and_scores_are_rejected() -> None:
    """A degenerate alpha or an out-of-range score is a calibration error (fail-closed)."""
    with pytest.raises(CalibrationError):
        conformal_threshold([], alpha=0.0)
    with pytest.raises(CalibrationError):
        conformal_threshold([_record("e1", 1.5, supported=False)], alpha=0.5)


def test_artifact_loader_roundtrips_and_fails_closed(tmp_path: Path) -> None:
    """The JSON artifact loads typed records; malformed artifacts raise (CFG-R1a)."""
    artifact = _write_artifact(
        tmp_path / "calibration.json",
        [{"example_id": "e1", "group": "number", "score": 0.4, "supported": False}],
    )
    (record,) = load_calibration(artifact, expected=_PROVENANCE)
    assert record == _record("e1", 0.4, supported=False)
    broken = tmp_path / "broken.json"
    broken.write_text('{"provenance": {}, "records": "not a list"}', encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(broken, expected=_PROVENANCE)
    with pytest.raises(CalibrationError):
        load_calibration(tmp_path / "missing.json", expected=_PROVENANCE)


# --- provenance pinning (the QA-EVAL-R12 cassette-pin rule applied to calibration) ----------


def test_unstamped_artifact_fails_the_boot_closed(tmp_path: Path) -> None:
    """An artifact with no provenance stamp (incl. the legacy bare array) is refused.

    A calibration with unknown provenance cannot certify anything — the conformal
    guarantee is conditional on the scores coming from THIS verifier/prompt — so a
    missing stamp fails exactly like a malformed artifact (fail-closed at load).
    """
    records = [{"example_id": "e1", "group": "number", "score": 0.4, "supported": False}]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(legacy, expected=_PROVENANCE)
    unstamped = _write_artifact(tmp_path / "unstamped.json", records, provenance=None)
    with pytest.raises(CalibrationError):
        load_calibration(unstamped, expected=_PROVENANCE)


def test_blank_or_missing_provenance_field_fails_closed(tmp_path: Path) -> None:
    """Every provenance field must be a non-blank string (a blank stamp pins nothing)."""
    records = [{"example_id": "e1", "group": "number", "score": 0.4, "supported": False}]
    for stamp in (
        {"model_id": _PROVENANCE.model_id, "claim_prompt_sha256": _PROVENANCE.claim_prompt_sha256},
        {
            "model_id": _PROVENANCE.model_id,
            "claim_prompt_sha256": _PROVENANCE.claim_prompt_sha256,
            "dataset_version": "  ",
        },
    ):
        artifact = _write_artifact(tmp_path / "partial.json", records, provenance=stamp)
        with pytest.raises(CalibrationError):
            load_calibration(artifact, expected=_PROVENANCE)


def test_stale_model_or_prompt_stamp_fails_the_boot_closed(tmp_path: Path) -> None:
    """A stamp recorded under another verifier model or claim prompt is STALE (refused).

    The mis-calibration this closes: swap the verifier checkpoint or reword the
    claim-extraction prompt and the old thresholds silently stop meaning P<=alpha.
    """
    records = [{"example_id": "e1", "group": "number", "score": 0.4, "supported": False}]
    stale_model = _write_artifact(
        tmp_path / "stale-model.json",
        records,
        provenance={
            "model_id": "someone/other-checkpoint",
            "claim_prompt_sha256": _PROVENANCE.claim_prompt_sha256,
            "dataset_version": _PROVENANCE.dataset_version,
        },
    )
    with pytest.raises(CalibrationError, match="model"):
        load_calibration(stale_model, expected=_PROVENANCE)
    stale_prompt = _write_artifact(
        tmp_path / "stale-prompt.json",
        records,
        provenance={
            "model_id": _PROVENANCE.model_id,
            "claim_prompt_sha256": prompt_sha256("a different extraction prompt"),
            "dataset_version": _PROVENANCE.dataset_version,
        },
    )
    with pytest.raises(CalibrationError, match="prompt"):
        load_calibration(stale_prompt, expected=_PROVENANCE)


def test_dataset_version_pin_is_exact_when_configured_and_lax_when_empty(
    tmp_path: Path,
) -> None:
    """A configured dataset pin must match exactly; an empty pin accepts any stamped label."""
    records = [{"example_id": "e1", "group": "number", "score": 0.4, "supported": False}]
    artifact = _write_artifact(tmp_path / "calibration.json", records)
    with pytest.raises(CalibrationError, match="dataset_version"):
        load_calibration(
            artifact,
            expected=CalibrationProvenance(
                model_id=_PROVENANCE.model_id,
                claim_prompt_sha256=_PROVENANCE.claim_prompt_sha256,
                dataset_version="grounding-corpus-v2",
            ),
        )
    unpinned = CalibrationProvenance(
        model_id=_PROVENANCE.model_id,
        claim_prompt_sha256=_PROVENANCE.claim_prompt_sha256,
        dataset_version="",
    )
    assert load_calibration(artifact, expected=unpinned) == (_record("e1", 0.4, supported=False),)


def test_gate_seam_pins_the_artifact_to_the_configured_model_and_prompt(
    tmp_path: Path,
) -> None:
    """``gate_from_settings`` derives the expected pin FROM the loaded config (CFG-R1a).

    Boot with a matching artifact builds the calibrated gate; boot with an artifact
    recorded under another claim prompt fails closed — the seam, not just the loader,
    enforces the pin.
    """
    records = [
        {"example_id": f"e{i}", "group": g, "score": 0.1, "supported": False}
        for i in range(30)
        for g in ("number", "statement")
    ]
    artifact = _write_artifact(tmp_path / "calibration.json", records)
    settings = SimpleNamespace(
        agent__entailment__enabled=True,
        agent__entailment__model_id=_PROVENANCE.model_id,
        agent__entailment__device="cpu",
        agent__entailment__threshold_number=0.5,
        agent__entailment__threshold_statement=0.5,
        agent__entailment__alpha=0.05,
        agent__entailment__calibration_path=str(artifact),
        agent__entailment__calibration_dataset_version="grounding-corpus-v1",
        agent__entailment__max_checks=16,
        agent__coach__prompts={"claim_system": _CLAIM_PROMPT},
    )
    assert gate_from_settings(settings) is not None
    settings.agent__coach__prompts = {"claim_system": "a reworded extraction prompt"}
    with pytest.raises(CalibrationError, match="prompt"):
        gate_from_settings(settings)
