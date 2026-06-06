"""Test-docstring linter (QUAL-R10(b), CI-R1 item 14).

Every ``test_*`` function MUST carry a short docstring as its first statement
stating, in plain English, the behavioural contract under test. A test function
with NO docstring, or one that merely echoes its own name (e.g.
``def test_ctl_rises(): \"\"\"test ctl rises\"\"\"``), fails this rule. Only files
that look like test modules are inspected (``test_*.py`` / ``conftest.py`` /
under a ``tests/`` tree) so production code is never flagged here.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from tools.lint.core import Violation, iter_python_files

_RULE = "test-docstrings"
_REQ = "QUAL-R10b"


def _is_test_module(path: Path) -> bool:
    """True when a file is a pytest test module (where the rule applies)."""
    if path.name.startswith("test_") or path.name == "conftest.py":
        return True
    return "/tests/" in f"/{path.as_posix()}"


def _normalize(text: str) -> str:
    """Reduce a name/docstring to comparable tokens (lowercase, no separators)."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _is_echo(name: str, docstring: str) -> bool:
    """True when the docstring only restates the test's name (adds no intent).

    The function name with its ``test_`` prefix stripped is compared, separator-
    insensitively, against the docstring; an exact normalized match is an echo
    (QUAL-R10(b): "or one that only echoes its name").
    """
    bare = name[len("test_") :] if name.startswith("test_") else name
    norm_doc = _normalize(docstring)
    return norm_doc == _normalize(name) or norm_doc == _normalize(bare)


def _check_source(path: Path, source: str) -> list[Violation]:
    """Flag every ``test_*`` function lacking a meaningful first-statement docstring."""
    violations: list[Violation] = []
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        doc = ast.get_docstring(node)
        if doc is None or not doc.strip():
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    rule=_RULE,
                    requirement=_REQ,
                    message=(
                        f"test '{node.name}' has no docstring; state the behavioural "
                        f"contract it asserts as the first statement"
                    ),
                )
            )
            continue
        if _is_echo(node.name, doc):
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    rule=_RULE,
                    requirement=_REQ,
                    message=(
                        f"test '{node.name}' docstring only echoes its name; describe "
                        f"the invariant/behaviour under test"
                    ),
                )
            )
    return violations


def check_paths(paths: Iterable[Path]) -> list[Violation]:
    """Run the test-docstring linter over every test module under `paths`."""
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        if not _is_test_module(path):
            continue
        violations.extend(_check_source(path, path.read_text(encoding="utf-8")))
    return violations
