"""Guard the T-MUT ratchet baseline (TIER-R6): floors are well-formed and only ratchet up.

``mutation-floors.toml`` is a COMMITTED ratchet baseline — each per-package mutation-score
floor records the package's current measured score and may only ever be RAISED, never
lowered. These tests enforce that contract cheaply:

* every floor is a real number in ``[0, 1]`` and the file parses;
* the gate's declared floored packages match the file (no silent drift);
* no committed floor decreased vs the file's previous git revision (the only edits are
  upward ratchets). The git-history leg is skipped when git/HEAD is unavailable (e.g. a
  shallow CI checkout or the file's introducing commit), so it never fails spuriously.
"""

from __future__ import annotations

import subprocess
import tomllib

import pytest
from tools.mutation_gate import _FLOORS_PATH, load_floors

_REPO_ROOT = _FLOORS_PATH.parent


@pytest.mark.unit
def test_floors_file_parses_and_values_are_scores_in_unit_interval() -> None:
    floors = load_floors()
    assert floors, "mutation-floors.toml declares no [floors]"
    for package, floor in floors.items():
        assert isinstance(floor, float), f"{package} floor must be a number"
        assert 0.0 <= floor <= 1.0, f"{package} floor {floor} outside [0, 1]"


@pytest.mark.unit
def test_floored_packages_are_the_mutated_packages() -> None:
    # The ratchet must cover exactly the analytics + adapter packages the [tool.mutmut]
    # campaign mutates — a floor file that drifts from the mutated set is a silent hole.
    floors = set(load_floors())
    assert floors == {
        "wattwise_core.analytics",
        "wattwise_core.ingestion.adapters",
    }


def _previous_committed_floors() -> dict[str, float] | None:
    """Floors from ``git show HEAD:mutation-floors.toml``; None if unavailable."""
    proc = subprocess.run(
        ["git", "show", "HEAD:mutation-floors.toml"],  # noqa: S607 - PATH-resolved git, dev/CI tooling
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None  # no HEAD revision of the file yet (introducing commit / shallow checkout)
    data = tomllib.loads(proc.stdout)
    return {pkg: float(score) for pkg, score in data.get("floors", {}).items()}


@pytest.mark.unit
def test_committed_floors_never_decrease_vs_previous_revision() -> None:
    previous = _previous_committed_floors()
    if previous is None:
        pytest.skip("no previous committed mutation-floors.toml revision to compare")
    current = load_floors()
    for package, prior in previous.items():
        if package not in current:
            continue  # package removed from the campaign is handled elsewhere
        assert current[package] >= prior, (
            f"{package} floor lowered {prior:.3f} -> {current[package]:.3f}; "
            "mutation-floors.toml may only RATCHET UP (TIER-R6)"
        )
