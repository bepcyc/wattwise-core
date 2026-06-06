"""Shared types and file discovery for the custom lint pack.

Defines the typed `Violation` value object every linter returns, the `Severity`
enum that separates BLOCKING gate failures from ADVISORY review warnings
(QUAL-R11(b): the non-English prose heuristic must NOT hard-fail), and the
path-walking helpers used to enumerate the Python sources a linter inspects.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


class Severity(enum.Enum):
    """A finding's gate impact.

    BLOCKING findings make the runner exit non-zero (a real gate failure);
    ADVISORY findings are surfaced for human review only and never fail CI
    (QUAL-R11(b) heuristic prose detection — avoids a flaky gate, R11).
    """

    BLOCKING = "blocking"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class Violation:
    """One lint finding, anchored to a `file:line` location for review.

    `rule` is the short linter id (e.g. ``size-limits``); `requirement` cites the
    governing spec ID so a reviewer can trace the gate to its mandate.
    """

    path: Path
    line: int
    rule: str
    message: str
    requirement: str
    severity: Severity = Severity.BLOCKING

    def render(self) -> str:
        """Format the finding as a single ``path:line: [SEV] rule (REQ) message`` line."""
        tag = "ERROR" if self.severity is Severity.BLOCKING else "warning"
        return (
            f"{self.path}:{self.line}: {tag} [{self.rule}] "
            f"({self.requirement}) {self.message}"
        )


# Directory names that are never engine source: skipped wholesale during discovery.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".hypothesis",
        "node_modules",
        "build",
        "dist",
    }
)


def iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield every ``*.py`` file under the given paths, skipping caches/VCS dirs.

    A path that is itself a ``.py`` file is yielded directly; a directory is walked
    recursively. Results are de-duplicated and emitted in a stable sorted order so
    linter output is deterministic (QUAL-R4 reproducibility).
    """
    seen: set[Path] = set()
    for raw in paths:
        root = raw.resolve()
        if root.is_file():
            if root.suffix == ".py" and root not in seen:
                seen.add(root)
            continue
        for candidate in sorted(root.rglob("*.py")):
            if any(part in _SKIP_DIRS for part in candidate.parts):
                continue
            if candidate not in seen:
                seen.add(candidate)
    yield from sorted(seen)
