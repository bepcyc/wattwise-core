"""Aggregating runner for the custom lint pack (CI-R1 items 13/14/21, ARCH-R21/R22).

Collects findings from every custom AST/static linter and partitions them by
severity: BLOCKING findings make the process exit non-zero (a real gate failure);
ADVISORY findings (the QUAL-R11(b) non-English-prose heuristic) are printed for
review but never fail the build. Output is one ``path:line: ...`` line per finding
so CI logs and editors can jump straight to the offending location.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from tools.lint import (
    content_copy,
    english_only,
    import_direction,
    no_vendor_sql,
    size_limits,
    test_docstrings,
)
from tools.lint.core import Severity, Violation

# Each registered linter is a pure ``paths -> [Violation]`` function.
_LINTERS: tuple[Callable[[Iterable[Path]], list[Violation]], ...] = (
    size_limits.check_paths,
    test_docstrings.check_paths,
    english_only.check_paths,
    content_copy.check_paths,
    no_vendor_sql.check_paths,
    import_direction.check_paths,
)


def collect(paths: Sequence[Path]) -> list[Violation]:
    """Run every linter over `paths` and return all findings sorted for stable output."""
    materialized = list(paths)
    findings: list[Violation] = []
    for linter in _LINTERS:
        findings.extend(linter(materialized))
    return sorted(findings, key=lambda v: (str(v.path), v.line, v.rule))


def run(argv: Sequence[str]) -> int:
    """Execute the lint pack; return process exit code (non-zero on a BLOCKING finding).

    Default target paths are ``src`` and ``tools`` when no explicit paths are given,
    matching the validation command in the build (``python -m tools.lint src tools``).
    """
    raw_paths = list(argv) or ["src", "tools"]
    paths = [Path(p) for p in raw_paths]
    findings = collect(paths)

    blocking = [v for v in findings if v.severity is Severity.BLOCKING]
    advisory = [v for v in findings if v.severity is Severity.ADVISORY]

    for finding in findings:
        stream = sys.stderr if finding.severity is Severity.BLOCKING else sys.stdout
        print(finding.render(), file=stream)

    if advisory:
        print(
            f"lint: {len(advisory)} advisory finding(s) (non-blocking, review-only)",
            file=sys.stdout,
        )
    if blocking:
        print(
            f"lint: FAILED with {len(blocking)} blocking violation(s)",
            file=sys.stderr,
        )
        return 1
    print("lint: OK (no blocking violations)", file=sys.stdout)
    return 0
