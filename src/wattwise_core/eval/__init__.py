"""Offline agent evaluation harness (doc 50 EVAL-R*, INJECT-R*; doc 80 QA-EVAL-R*).

This package is the CI-gated, network-free evaluation suite for the trustworthy
coaching agent. It exists so groundedness, abstention, schema-conformance, and
prompt-injection isolation are proven by DETERMINISTIC graders rather than by any
model self-assertion (OUTCOME-R5, EVAL-R1).

Three pieces:

* :mod:`wattwise_core.eval.runner` — loads versioned checked-in datasets
  (QA-EVAL-R1), runs a minimal reference coaching pipeline over the stable
  :mod:`wattwise_core.agent.contracts` seam with a deterministic offline model
  (``FakeModel`` / recorded-response mode, QA-EVAL-R9, TIER-R1: no network), and
  produces a machine-readable scorecard (EVAL-R9).
* :mod:`wattwise_core.eval.grading` — the deterministic graders that bind the hard
  QA-EVAL-R6 thresholds: grounding faithfulness >= 99% with zero fabricated numbers,
  abstention 100%, structured-output conformance 100% (OUTCOME-R5).
* ``datasets/`` — versioned checked-in cases: grounding/faithfulness, abstention /
  fail-closed (QA-EVAL-R2.2), and a prompt-injection corpus (INJ-R2 / INJECT-R4).

The harness deliberately depends ONLY on ``agent.contracts`` (the published seam) and
on the canonical analytics result envelope, so it can be authored independently of the
in-flight graph/grounding sibling modules (doc 10 layer rule). The reference pipeline
here re-expresses the contract semantics (claim extraction via the model, then a
deterministic match/scrub against canonical evidence, GROUND-R3 "when in doubt,
scrub") purely to exercise the graders; it is the graders, not the model, that gate.
"""

from __future__ import annotations

from wattwise_core.eval.grading import (
    AbstentionGrade,
    GroundingGrade,
    SchemaGrade,
    grade_abstention,
    grade_grounding,
    grade_injection,
    grade_schema,
)
from wattwise_core.eval.runner import (
    EvalMode,
    EvalRunner,
    RunnerOutcome,
    Scorecard,
    load_dataset,
)

__all__ = [
    "AbstentionGrade",
    "EvalMode",
    "EvalRunner",
    "GroundingGrade",
    "RunnerOutcome",
    "SchemaGrade",
    "Scorecard",
    "grade_abstention",
    "grade_grounding",
    "grade_injection",
    "grade_schema",
    "load_dataset",
]
