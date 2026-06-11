"""Split-conformal calibration of the entailment thresholds (issue #10 Phase 3).

The proposed GROUND-R12 layer: the entailment gate's publication threshold should carry a
GUARANTEE, not a vibe. Given a labelled calibration set of verifier scores (one record per
checked sentence, labelled supported/unsupported by the offline GROUND-R8 corpus), this
module computes per-group split-conformal thresholds such that, under exchangeability,
the probability that a NEW deliverable publishes any unsupported sentence is at most
``alpha`` (finite-sample, distribution-free). The construction follows conformal language
modelling / conformal factuality (Quach et al., arXiv:2306.10193; Cherian, Gibbs & Candès,
arXiv:2406.09714) with GROUP-CONDITIONAL calibration per claim class (number vs statement
sentences), per the multi-verifier conformal-factuality literature (arXiv:2602.01285):

* per calibration EXAMPLE ``i`` (one deliverable), the exceedance score is
  ``E_i = max(score of its UNSUPPORTED sentences)`` (``0`` when none) — a new example
  publishes a bad sentence iff ``E > tau``;
* ``tau`` is the ``ceil((n + 1) * (1 - alpha))``-th smallest of the ``E_i`` — the standard
  split-conformal upper quantile, so ``P(E_new > tau) <= alpha``;
* too FEW calibration examples for the requested ``alpha`` yields ``tau = 1.0`` — nothing
  certifies, publish nothing on this guarantee (fail-closed, never an extrapolated bound).

Pure, deterministic functions over their inputs; the calibration artifact is loaded
content (CFG-R1a), produced offline and shipped/mounted like any other config. Honest
caveat, by design: the guarantee assumes the calibration and deployment distributions are
exchangeable — recalibration belongs in the release checklist (issue #10 Phase 4).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """What a calibration artifact was RECORDED with (the QA-EVAL-R12 pin, applied here).

    The conformal guarantee is exchangeability-conditional: scores collected under a
    DIFFERENT verifier checkpoint or a different claim-extraction prompt are a different
    distribution, so an artifact silently carried across a model/prompt change would
    mis-calibrate the gate while looking perfectly healthy. Mirroring the recorded-cassette
    rule (QA-EVAL-R12 (a): cassette metadata out of sync with the prompt/model version
    pinned in config fails the build), every artifact STAMPS the verifier ``model_id``,
    the SHA-256 of the claim-extraction system prompt, and a ``dataset_version`` label —
    and :func:`load_calibration` REFUSES a stale or unstamped artifact at boot
    (fail-closed, exactly like a malformed one).
    """

    model_id: str
    claim_prompt_sha256: str
    dataset_version: str


def prompt_sha256(prompt: str) -> str:
    """The canonical SHA-256 hex digest a provenance stamp pins a prompt by."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """One labelled verifier outcome from the offline calibration corpus (GROUND-R8).

    ``example_id`` groups records into deliverables (the conformal unit); ``group`` is the
    claim class (``"number"`` / ``"statement"``); ``score`` is the verifier's support
    probability for the sentence; ``supported`` is the oracle label.
    """

    example_id: str
    group: str
    score: float
    supported: bool


class CalibrationError(ValueError):
    """A calibration artifact is malformed or insufficient (fail-closed at load)."""


def conformal_threshold(records: Sequence[CalibrationRecord], alpha: float) -> float:
    """The split-conformal publication threshold for ONE group of records.

    Publishing sentences with ``score > tau`` then bounds the chance a new deliverable
    ships any unsupported sentence of this group by ``alpha``. With too few calibration
    examples for ``alpha`` the threshold is ``1.0`` — no sentence can certify against it
    (fail-closed): more calibration data, not a looser bound, is the way out.
    """
    if not 0.0 < alpha < 1.0:
        raise CalibrationError(f"alpha must be in (0, 1), got {alpha}")
    exceedances: dict[str, float] = {}
    for record in records:
        if not 0.0 <= record.score <= 1.0:
            raise CalibrationError(f"score must be in [0, 1], got {record.score}")
        worst = exceedances.setdefault(record.example_id, 0.0)
        if not record.supported and record.score > worst:
            exceedances[record.example_id] = record.score
    scores = sorted(exceedances.values())
    n = len(scores)
    if n == 0:
        return 1.0
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return 1.0
    return scores[rank - 1]


def conformal_thresholds(
    records: Iterable[CalibrationRecord], alpha: float, *, groups: Sequence[str]
) -> dict[str, float]:
    """Group-conditional thresholds (one split-conformal bound PER claim class)."""
    pool = list(records)
    return {
        group: conformal_threshold([r for r in pool if r.group == group], alpha) for group in groups
    }


def load_calibration(
    path: Path, *, expected: CalibrationProvenance
) -> tuple[CalibrationRecord, ...]:
    """Load the calibration artifact, provenance-pinned and fail-closed (CFG-R1a).

    Artifact schema: ``{"provenance": {"model_id": str, "claim_prompt_sha256": str,
    "dataset_version": str}, "records": [{"example_id": str, "group": str,
    "score": float, "supported": bool}, ...]}``. The provenance stamp is CHECKED against
    ``expected`` (the configured verifier checkpoint and the SHA-256 of the configured
    claim-extraction prompt — the QA-EVAL-R12 cassette-pin rule applied to calibration):
    a missing, blank, or mismatched field raises :class:`CalibrationError`, so a STALE
    artifact (recorded under another model/prompt/dataset) stops the boot exactly like a
    malformed one — never a silently mis-calibrated guarantee. An empty
    ``expected.dataset_version`` pins only that a version IS stamped (any non-blank
    label); a non-empty pin must match exactly. Any structural problem in the records
    likewise raises so a misconfigured artifact stops the boot.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"calibration artifact unreadable: {path}") from exc
    if not isinstance(raw, dict):
        raise CalibrationError(
            "calibration artifact must be a JSON object with 'provenance' and 'records'"
        )
    _check_provenance(raw.get("provenance"), expected)
    items = raw.get("records")
    if not isinstance(items, list):
        raise CalibrationError("calibration 'records' must be a JSON array of records")
    records: list[CalibrationRecord] = []
    for item in items:
        if not isinstance(item, dict):
            raise CalibrationError("each calibration record must be a JSON object")
        try:
            records.append(
                CalibrationRecord(
                    example_id=str(item["example_id"]),
                    group=str(item["group"]),
                    score=float(item["score"]),
                    supported=bool(item["supported"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"malformed calibration record: {item!r}") from exc
    return tuple(records)


def _check_provenance(stamped: object, expected: CalibrationProvenance) -> None:
    """Refuse a missing/blank/stale provenance stamp (fail-closed, QA-EVAL-R12 pattern)."""
    if not isinstance(stamped, dict):
        raise CalibrationError(
            "calibration artifact carries no 'provenance' stamp "
            "(model_id, claim_prompt_sha256, dataset_version are required)"
        )
    fields: dict[str, str] = {}
    for name in ("model_id", "claim_prompt_sha256", "dataset_version"):
        value = stamped.get(name)
        if not isinstance(value, str) or not value.strip():
            raise CalibrationError(f"calibration provenance field missing or blank: {name!r}")
        fields[name] = value
    if fields["model_id"] != expected.model_id:
        raise CalibrationError(
            "stale calibration artifact: recorded with verifier model "
            f"{fields['model_id']!r}, configured model is {expected.model_id!r} — "
            "recalibrate (the conformal guarantee does not transfer across checkpoints)"
        )
    if fields["claim_prompt_sha256"] != expected.claim_prompt_sha256:
        raise CalibrationError(
            "stale calibration artifact: recorded under a different claim-extraction "
            "prompt (sha256 mismatch) — recalibrate against the configured prompt"
        )
    if expected.dataset_version and fields["dataset_version"] != expected.dataset_version:
        raise CalibrationError(
            "stale calibration artifact: dataset_version "
            f"{fields['dataset_version']!r} does not match the configured pin "
            f"{expected.dataset_version!r}"
        )


__all__ = [
    "CalibrationError",
    "CalibrationProvenance",
    "CalibrationRecord",
    "conformal_threshold",
    "conformal_thresholds",
    "load_calibration",
    "prompt_sha256",
]
