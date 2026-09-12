"""QA-EVAL-R12: real CLI baseline publication and subsequent evaluation roundtrip."""

from pathlib import Path

import pytest

from wattwise_core.eval.__main__ import main as cli_main

pytestmark = pytest.mark.integration


def test_update_baseline_cli_rewrites_and_run_stays_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # QA-EVAL-R12(c): `update-baseline` actually rewrites the artifact (no longer a no-op),
    # and a subsequent `run` (with its non-regression gate active) stays green against it.
    target = tmp_path / "baseline-scorecard.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("wattwise_core.eval.baseline.BASELINE_PATH", target)
    assert cli_main(["update-baseline"]) == 0
    assert target.exists(), "update-baseline MUST write the baseline artifact (not a no-op)"
    out = tmp_path / "scorecard.json"
    assert cli_main(["run", "--mode=recorded", f"--scorecard={out}"]) == 0
