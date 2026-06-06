#!/usr/bin/env python3
"""Requirement-ID -> test-ID traceability map (DOD-R6).

Cites: doc 80 DOD-R6 ("every shipped requirement ID maps to >=1 test ID, mapping
machine-checkable; a requirement with no covering test is not done; a requirement-IDs->
test-IDs coverage report SHOULD be produced and SHOULD fail CI if a 'shipped'
requirement has no covering test") and ROAD-R3 (automated traceability gating becomes a
required check in the hardening phase).

This module statically scans the test tree, extracts every spec requirement ID cited in
a test's docstring or marker (the convention the suite already follows — see QUAL-R10b:
each test docstring states the behavioural contract and cites the requirement it covers),
and builds a ``requirement -> [test node ids]`` map. It is import-safe (pure AST, no test
execution, no DB, no network) so it can run as a fast CI gate, and it is runnable directly:

    uv run python scripts/traceability.py            # human-readable coverage report
    uv run python scripts/traceability.py --json     # machine-readable map (CI artifact)

Exit code is 0 when every headline requirement ID is covered, 1 otherwise (fail-closed),
so the same command gates locally and in CI (CI-R0: local command == CI command).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

# A spec requirement ID: an uppercase namespace, ``-R``, a number, an optional dotted
# sub-number, and an optional trailing letter — e.g. ``DOD-R6``, ``PRIV-R8``,
# ``SEC-R2.1``, ``API-R11c``. ``-AC`` acceptance suffixes are normalized onto their base
# ID so an acceptance test counts toward its requirement.
_REQ_ID = re.compile(r"\b([A-Z][A-Z0-9]+-R[0-9]+(?:\.[0-9]+)?[a-z]?)(?:-AC)?\b")

# Headline requirement IDs this gate insists are covered (the close-out spine: traceability,
# erasure, the planted-secret gate, and portable persistence). These MUST have a covering
# test or the gate fails — they are the requirements this close-out task ships.
HEADLINE_REQUIREMENTS: tuple[str, ...] = (
    "DOD-R6",
    "PRIV-R8",
    "SEC-R12",
    "BOOT-R3",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = _REPO_ROOT / "tests"

# Marker names that, when applied, also imply the requirement family they gate (so a test
# tagged ``@pytest.mark.portability`` counts toward the portability requirement even if its
# docstring is terse). Kept narrow and explicit.
_MARKER_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "portability": ("BOOT-R3",),
}


def _iter_test_files(tests_dir: Path) -> Iterator[Path]:
    """Yield every ``test_*.py`` file under the test tree in stable sorted order."""
    yield from sorted(p for p in tests_dir.rglob("test_*.py") if "__pycache__" not in p.parts)


def _mark_attr(node: ast.expr) -> str | None:
    """Return the ``pytest.mark.<name>`` marker name for a node, or None if not a marker.

    ``pytest.mark.<name>`` parses to ``Attribute(attr=name, value=Attribute(attr='mark'))``
    (the decorator may also be a ``Call`` when the marker takes arguments).
    """
    target = node.func if isinstance(node, ast.Call) else node
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
    ):
        return target.attr
    return None


def _marker_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect the ``pytest.mark.<name>`` marker names decorating a test function."""
    return {attr for deco in node.decorator_list if (attr := _mark_attr(deco)) is not None}


def _module_marker_names(tree: ast.Module) -> set[str]:
    """Collect module-level ``pytestmark = pytest.mark.<name>`` (single or list)."""
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        candidates = node.value.elts if isinstance(node.value, ast.List) else [node.value]
        names |= {attr for cand in candidates if (attr := _mark_attr(cand)) is not None}
    return names


def _ids_from_text(text: str) -> set[str]:
    """Extract normalized requirement IDs from a blob of text (docstring / marker arg)."""
    return {m.group(1) for m in _REQ_ID.finditer(text)}


def _node_id(path: Path, func_name: str) -> str:
    """A pytest-style node id ``tests/<rel>::<func>`` for the map values."""
    rel = path.relative_to(_REPO_ROOT)
    return f"{rel.as_posix()}::{func_name}"


def build_traceability(tests_dir: Path = _TESTS_DIR) -> dict[str, list[str]]:
    """Build the ``requirement-ID -> sorted[test node id]`` map over the test tree (DOD-R6).

    A test covers a requirement when the ID appears in the test's docstring or its marker
    arguments, or when a module/function marker implies it (``_MARKER_REQUIREMENTS``). The
    scan is pure AST — no test is imported or executed.
    """
    mapping: dict[str, set[str]] = defaultdict(set)
    for path in _iter_test_files(tests_dir):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_markers = _module_marker_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            node_id = _node_id(path, node.name)
            ids: set[str] = set()
            doc = ast.get_docstring(node)
            if doc:
                ids |= _ids_from_text(doc)
            for marker in _marker_names(node) | module_markers:
                ids |= set(_MARKER_REQUIREMENTS.get(marker, ()))
            for req in ids:
                mapping[req].add(node_id)
    return {req: sorted(nodes) for req, nodes in sorted(mapping.items())}


def missing_headline(mapping: Mapping[str, list[str]]) -> list[str]:
    """Headline requirement IDs (DOD-R6 spine) that have no covering test."""
    return [req for req in HEADLINE_REQUIREMENTS if not mapping.get(req)]


def _render_report(mapping: Mapping[str, list[str]], missing: Iterable[str]) -> str:
    """Human-readable coverage report: per-requirement test counts + headline status."""
    lines = [
        "requirement -> test coverage (DOD-R6)",
        f"  requirements covered: {len(mapping)}",
        f"  test references:      {sum(len(v) for v in mapping.values())}",
        "",
        "headline requirements:",
    ]
    for req in HEADLINE_REQUIREMENTS:
        tests = mapping.get(req, [])
        status = "OK " if tests else "MISSING"
        lines.append(f"  [{status}] {req}: {len(tests)} test(s)")
    missing_list = list(missing)
    if missing_list:
        lines += ["", f"FAIL: uncovered headline requirements: {', '.join(missing_list)}"]
    else:
        lines += ["", "PASS: every headline requirement has a covering test."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Build the map, print the report (or JSON), and exit non-zero on any uncovered headline."""
    parser = argparse.ArgumentParser(description="Requirement->test traceability map (DOD-R6).")
    parser.add_argument("--json", action="store_true", help="emit the raw map as JSON")
    args = parser.parse_args(argv)

    mapping = build_traceability()
    missing = missing_headline(mapping)
    if args.json:
        print(json.dumps(mapping, indent=2, sort_keys=True))
    else:
        print(_render_report(mapping, missing))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
