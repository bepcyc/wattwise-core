"""T-MUT mutation-score gate (TIER-R6 / CI-R1 item 17): per-package floors over mutmut.

``python -m tools.mutation_gate run`` drives ``mutmut run`` over the correctness core
(the analytics package and the ASBO→GBO adapter layer, ``[tool.mutmut]`` in
``pyproject.toml``), then parses ``mutmut results --all true`` and enforces the
declared per-package mutation-score floors:

* ``wattwise_core/analytics``      — killed/generated >= **0.90**
* ``wattwise_core/ingestion/adapters`` — killed/generated >= **0.85**

``python -m tools.mutation_gate score`` re-checks an existing results listing without
re-running the (expensive) mutation campaign. The per-mutant listing + the aggregate
scores are written to ``reports/mutation-report.txt`` / ``reports/mutation-scores.json``
as retained CI artifacts (CI-R6). T-MUT runs nightly and on PRs touching the mutated
packages (the ``test-mut-pr`` recipe decides FROM THE DIFF, keeping the gate logic in
the Justfile/tools layer, CI-R0) — never on every push.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Declared mutation-score floors (TIER-R6): analytics 90%, adapter layer 85%.
FLOORS: dict[str, float] = {
    "wattwise_core.analytics": 0.90,
    "wattwise_core.ingestion.adapters": 0.85,
}

#: Mutant statuses that count as NOT killed. ``killed`` and ``timeout`` (the mutant
#: broke the suite's time budget — detected) count as caught; a survivor or a
#: suspicious/skipped mutant does not.
_CAUGHT = frozenset({"killed", "timeout"})
# "no tests" counts as a GENERATED-but-uncaught mutant (no covering test selected is a
# survivor for floor purposes); "not checked" (an interrupted run) is excluded so a
# partial campaign never reads as a fake pass/fail tally.
_COUNTED = frozenset({"killed", "timeout", "survived", "suspicious", "no_tests"})

_RESULT_LINE = re.compile(r"^(?P<name>[\w.]+__mutmut_\d+):\s*(?P<status>[a-z_ ]+)$")


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


def parse_results(listing: str) -> dict[str, PackageScore]:
    """Tally per-package mutation scores from a ``mutmut results --all true`` listing."""
    scores = {pkg: PackageScore(pkg, floor) for pkg, floor in FLOORS.items()}
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
            "passed": tally.passed,
        }
        for pkg, tally in scores.items()
    }
    (reports / "mutation-scores.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _gate(listing: str) -> int:
    scores = parse_results(listing)
    _write_artifacts(listing, scores)
    failed = False
    for tally in scores.values():
        status = "PASS" if tally.passed else "FAIL"
        print(
            f"[{status}] {tally.package}: mutation score {tally.score:.3f} "
            f"({tally.killed}/{tally.total} caught; floor {tally.floor:.2f})"
        )
        if not tally.passed:
            failed = True
    if failed:
        print("T-MUT gate FAILED: a package is below its mutation-score floor", file=sys.stderr)
        return 1
    print("T-MUT gate PASSED")
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


def main(argv: list[str] | None = None) -> int:
    """Run or re-score the T-MUT tier and enforce the per-package floors (TIER-R6)."""
    parser = argparse.ArgumentParser(prog="python -m tools.mutation_gate")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run the mutation campaign, then gate on the floors")
    sub.add_parser("score", help="gate on an existing mutmut results listing")
    args = parser.parse_args(argv)
    if args.command == "run":
        run_proc = subprocess.run(
            ["uv", "run", "mutmut", "run"],  # noqa: S607 - PATH-resolved uv, dev/CI tooling
            check=False,
        )
        if run_proc.returncode not in (0, 2):  # mutmut exits non-zero on survivors
            print("mutmut run failed to execute", file=sys.stderr)
            return 1
    return _gate(_results_listing())


if __name__ == "__main__":
    raise SystemExit(main())
