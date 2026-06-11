"""T-MUT mutation-score gate (TIER-R6 / CI-R1 item 17): per-package ratchet floors.

``python -m tools.mutation_gate run`` drives ``mutmut run`` over the correctness core
(the analytics package and the ASBO->GBO adapter layer, ``[tool.mutmut]`` in
``pyproject.toml``), then parses ``mutmut results --all true`` and compares each
package's measured mutation score against its committed floor in ``mutation-floors.toml``.

The floors are a **committed ratchet baseline** (TIER-R6): each floor is the package's
current measured score and is only ever RAISED, never lowered. The gate has two modes:

* ``--advisory`` (the PR leg): MEASURE + REPORT the score and its delta vs the committed
  floor, but ALWAYS exit 0 — a PR is never blocked on mutation score.
* ``--enforce`` (the nightly leg): FAIL (exit 1) if any package's measured score drops
  BELOW its committed floor.

``score`` re-checks an existing results listing without re-running the (expensive)
mutation campaign; it takes the same ``--advisory``/``--enforce`` flag. The per-mutant
listing + the aggregate scores are written to ``reports/mutation-report.txt`` /
``reports/mutation-scores.json`` as retained CI artifacts (CI-R6). The floors live in
data (``mutation-floors.toml``), not hardcoded here (CFG-R1a style) so raising a floor is
a one-line committed change. The Justfile/tools layer owns all gate logic (CI-R0).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

#: Committed ratchet baseline (TIER-R6). Floors are DATA, not hardcoded (CFG-R1a):
#: this is the repo-root file each measured score is gated against; it only ratchets up.
_FLOORS_PATH = Path(__file__).resolve().parent.parent / "mutation-floors.toml"

#: Mutant statuses that count as NOT killed. ``killed`` and ``timeout`` (the mutant
#: broke the suite's time budget — detected) count as caught; a survivor or a
#: suspicious/skipped mutant does not.
_CAUGHT = frozenset({"killed", "timeout"})
# "no tests" counts as a GENERATED-but-uncaught mutant (no covering test selected is a
# survivor for floor purposes); "not checked" (an interrupted run) is excluded so a
# partial campaign never reads as a fake pass/fail tally.
_COUNTED = frozenset({"killed", "timeout", "survived", "suspicious", "no_tests"})

_RESULT_LINE = re.compile(r"^(?P<name>[\w.]+__mutmut_\d+):\s*(?P<status>[a-z_ ]+)$")


def load_floors(path: Path = _FLOORS_PATH) -> dict[str, float]:
    """Read the committed per-package ratchet floors from ``mutation-floors.toml``."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    floors = data.get("floors", {})
    return {pkg: float(score) for pkg, score in floors.items()}


@dataclass
class PackageScore:
    """Killed/generated tally for one floored package."""

    package: str
    floor: float
    killed: int = 0
    total: int = 0

    @property
    def score(self) -> float:
        return 1.0 if self.total == 0 else self.killed / self.total

    @property
    def passed(self) -> bool:
        return self.score >= self.floor

    @property
    def delta(self) -> float:
        return self.score - self.floor


def parse_results(listing: str, floors: dict[str, float]) -> dict[str, PackageScore]:
    """Tally per-package mutation scores from a ``mutmut results --all true`` listing."""
    scores = {pkg: PackageScore(pkg, floor) for pkg, floor in floors.items()}
    for raw in listing.splitlines():
        match = _RESULT_LINE.match(raw.strip())
        if match is None:
            continue
        name = match.group("name")
        status = match.group("status").strip().replace(" ", "_")
        if status not in _COUNTED:
            continue
        for pkg, tally in scores.items():
            if name.startswith(pkg + "."):
                tally.total += 1
                if status in _CAUGHT:
                    tally.killed += 1
                break
    return scores


def _write_artifacts(listing: str, scores: dict[str, PackageScore]) -> None:
    """Retain the per-mutant report + aggregate scores as CI artifacts (CI-R6)."""
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "mutation-report.txt").write_text(listing, encoding="utf-8")
    payload = {
        pkg: {
            "floor": tally.floor,
            "killed": tally.killed,
            "total": tally.total,
            "score": tally.score,
            "delta": tally.delta,
            "passed": tally.passed,
        }
        for pkg, tally in scores.items()
    }
    (reports / "mutation-scores.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _gate(listing: str, *, enforce: bool) -> int:
    """Report per-package scores vs the committed ratchet floor.

    ``enforce=False`` (PR/advisory): always returns 0. ``enforce=True`` (nightly):
    returns 1 if any package dropped below its committed floor (TIER-R6).
    """
    scores = parse_results(listing, load_floors())
    _write_artifacts(listing, scores)
    mode = "ENFORCE" if enforce else "ADVISORY"
    below_floor = False
    for tally in scores.values():
        ok = tally.passed
        status = "PASS" if ok else ("FAIL" if enforce else "BELOW-FLOOR")
        print(
            f"[{status}] {tally.package}: mutation score {tally.score:.3f} "
            f"(floor {tally.floor:.2f}, delta {tally.delta:+.3f}; "
            f"{tally.killed}/{tally.total} caught)"
        )
        if not ok:
            below_floor = True
    if enforce and below_floor:
        print(
            "T-MUT gate FAILED: a package dropped below its committed ratchet floor "
            "(mutation-floors.toml)",
            file=sys.stderr,
        )
        return 1
    if below_floor:
        print(f"T-MUT {mode}: a package is below floor (reported, not blocking)")
    else:
        print(f"T-MUT {mode}: all packages at or above their committed floor")
    return 0


def _results_listing() -> str:
    proc = subprocess.run(
        ["uv", "run", "mutmut", "results", "--all", "true"],  # noqa: S607 - PATH-resolved uv, dev/CI tooling
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 and not proc.stdout:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(1)
    return proc.stdout


def _add_mode_flag(parser: argparse.ArgumentParser) -> None:
    """Add the mutually-exclusive ``--enforce``/``--advisory`` mode (default advisory)."""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--enforce",
        dest="enforce",
        action="store_true",
        help="FAIL if any package is below its committed floor (nightly leg)",
    )
    mode.add_argument(
        "--advisory",
        dest="enforce",
        action="store_false",
        help="report scores + deltas but never block (PR leg, default)",
    )
    parser.set_defaults(enforce=False)


def main(argv: list[str] | None = None) -> int:
    """Run or re-score the T-MUT tier against the committed ratchet floors (TIER-R6)."""
    parser = argparse.ArgumentParser(prog="python -m tools.mutation_gate")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run the mutation campaign, then gate on the floors")
    _add_mode_flag(run_p)
    score_p = sub.add_parser("score", help="gate on an existing mutmut results listing")
    _add_mode_flag(score_p)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_proc = subprocess.run(
            ["uv", "run", "mutmut", "run"],  # noqa: S607 - PATH-resolved uv, dev/CI tooling
            check=False,
        )
        if run_proc.returncode not in (0, 2):  # mutmut exits non-zero on survivors
            print("mutmut run failed to execute", file=sys.stderr)
            return 1
    return _gate(_results_listing(), enforce=args.enforce)


if __name__ == "__main__":
    raise SystemExit(main())
