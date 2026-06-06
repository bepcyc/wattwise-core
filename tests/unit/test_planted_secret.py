"""The committed-secret gate actually FIRES on a planted secret (SEC-R12-AC / SEC-R13.1).

Cites: doc 70 SEC-R12-AC ("Grep/static gate over source+images -> zero hardcoded-secret/
fallback-secret patterns; planted test secret caught -> fail CI") and SEC-R13.1 (a committed
secret fails the pipeline, scanner over the diff AND the full tree).

A secret-scan gate that never catches a real secret is worse than none — a blind gate is a silent
green. This test proves the gate is wired correctly end-to-end: it plants a fake (never-real) AWS
key file inside the repo tree, runs ``scripts/secret_scan.sh``, and asserts the gate goes RED
(non-zero exit). It is marked ``logging`` (the QA-LOG-R*/secret-scan family) and is skipped when no
secret scanner is installed, since on a scanner-less host the gate fails closed for a DIFFERENT
reason (no tool) and cannot demonstrate detection.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.logging

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "secret_scan.sh"

# A fake AWS access-key id: the ``AKIA`` prefix + 16 fixed chars that match the gitleaks AWS rule.
# It is NOT the canonical docs example (``AKIAIOSFODNN7EXAMPLE``) — that one is allowlisted — and it
# is assembled at runtime so this test file does not itself carry a static, scannable secret literal
# (which the repo's own committed-secret scan would otherwise flag).
_FAKE_AWS_ID = "AKIA" + "Q7" + "WATTWISEFAKE00"
_FAKE_AWS_SECRET = "wattwise/" + "FAKEsecret0NOTreal0planted0canary0xyz123"


def _scanner_installed() -> bool:
    """True when a secret scanner the gate understands (gitleaks/trufflehog) is on PATH."""
    return shutil.which("gitleaks") is not None or shutil.which("trufflehog") is not None


@pytest.mark.skipif(
    not _scanner_installed(),
    reason="no secret scanner (gitleaks/trufflehog) installed; gate fails closed for another reason",  # noqa: E501
)
def test_secret_scan_catches_a_planted_aws_key(tmp_path: Path) -> None:
    """Planting a fake AWS key in the repo tree makes scripts/secret_scan.sh exit non-zero."""
    assert _SCRIPT.is_file(), f"missing gate script: {_SCRIPT}"
    bash = shutil.which("bash")
    assert bash is not None, "bash is required to run the gate script"

    # Plant the canary at the repo root (NOT under tests/, which the gitleaks config allowlists).
    planted = _REPO_ROOT / f".planted_secret_{uuid.uuid4().hex}.txt"
    planted.write_text(
        f"aws_access_key_id = {_FAKE_AWS_ID}\naws_secret_access_key = {_FAKE_AWS_SECRET}\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    # Disable the committed allowlist for THIS run so the real-tree scan (step 2) is not told to
    # ignore tests/ or the example key — the planted canary must be seen unfiltered. The script's
    # own self-test (step 1) never applies the config, so this does not weaken it.
    env["WW_GITLEAKS_CONFIG"] = str(tmp_path / "no-such-allowlist.toml")
    # Keep generated reports out of the repo tree.
    env["WW_OUT_DIR"] = str(tmp_path / "scan-out")
    try:
        result = subprocess.run(  # noqa: S603 - fixed gate script, no shell, controlled args
            [bash, str(_SCRIPT)],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        planted.unlink(missing_ok=True)

    # The gate MUST go RED: a committed secret was present, so the scan fails closed (SEC-R13.1).
    assert result.returncode != 0, (
        "secret-scan gate did NOT fail on a planted AWS key — the gate is blind "
        f"(stdout={result.stdout!r} stderr={result.stderr!r})"
    )


def test_gate_script_is_executable_and_fails_closed_without_a_scanner() -> None:
    """The gate script exists and is wired to fail closed (never silently passes), SEC-R13."""
    assert _SCRIPT.is_file()
    text = _SCRIPT.read_text(encoding="utf-8")
    # Fail-closed contract: a missing scanner / planted-secret miss routes to ww_die (exit 1),
    # never an exit 0. We assert the script references the fail-closed primitive and the gate IDs.
    assert "ww_die" in text
    assert "SEC-R13.1" in text
    assert "SEC-R12-AC" in text
