"""Module / function / class size-ceiling linter (QUAL-R9(a), CI-R1 item 14).

Spec-cited ceilings:
  * module:   <= 400 non-blank / non-comment lines  (QUAL-R9(a))
  * function: <= 60 lines (span)                     (QUAL-R9(a))
  * class:    <= 200 non-blank / non-comment lines   (derived guard; QUAL-R9 notes
              "a 200-line module mixing unrelated concerns is still a violation",
              so 200 is the review-judgement anchor applied to a single class body)

Test modules and generated migration files are EXEMPT from the ceiling
(QUAL-R9(a): "Test modules and generated migration files are exempt"). An
individual justified over-limit case carries a per-node suppression comment
``# noqa: size-limits`` on the def/class line (QUAL-R9: "a documented suppression
comment", "never by blanket per-file suppression").
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from tools.lint.core import Severity, Violation, iter_python_files

MODULE_MAX_LINES = 400
FUNCTION_MAX_LINES = 60
CLASS_MAX_LINES = 200

_RULE = "size-limits"
_REQ = "QUAL-R9"
_SUPPRESS = "# noqa: size-limits"

# Path SEGMENTS (directory names) whose files are exempt from the size ceiling
# (QUAL-R9(a)). Matched against path parts, never as raw substrings — a tmp dir
# named "test_..." in the absolute path must NOT accidentally exempt a real module.
_EXEMPT_SEGMENTS = frozenset({"tests", "migrations"})


def _is_exempt(path: Path) -> bool:
    """True for test modules and generated migrations (QUAL-R9(a) exemption)."""
    name = path.name
    if name.startswith("test_") or name == "conftest.py":
        return True
    return any(part in _EXEMPT_SEGMENTS for part in path.parts)


def _significant_line_count(source: str) -> int:
    """Count physical lines that are neither blank nor a pure ``#`` comment.

    This is the QUAL-R9 "non-blank / non-comment lines" measure for a module.
    String-literal content is intentionally counted (it is real source weight);
    only whitespace-only lines and full-line comments are excluded.
    """
    count = 0
    for raw in source.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def _significant_span(lines: list[str], start: int, end: int) -> int:
    """Significant (non-blank/non-comment) line count for a ``[start, end]`` span.

    `start`/`end` are 1-based inclusive line numbers (ast `lineno`/`end_lineno`).
    """
    count = 0
    for raw in lines[start - 1 : end]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def _docstring_line_range(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    """1-based line numbers occupied by the function's docstring (empty if none).

    Docstrings are EXCLUDED from the function size measure so QUAL-R10(a)-mandated
    explanatory documentation (formula derivations, references, invariants) does not
    push an otherwise-reasonable function over the ceiling.
    """
    body = node.body
    if not body:
        return set()
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        end = first.end_lineno or first.lineno
        return set(range(first.lineno, end + 1))
    return set()


def _function_code_lines(lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count a function's significant CODE lines (non-blank, non-comment, non-docstring).

    This is the function analogue of the module's non-blank/non-comment measure
    (QUAL-R9(a)); it deliberately ignores the docstring span so well-documented
    functions are not penalised for documentation alone.
    """
    end = node.end_lineno or node.lineno
    doc_range = _docstring_line_range(node)
    count = 0
    for lineno in range(node.lineno, end + 1):
        raw = lines[lineno - 1].strip()
        if not raw or raw.startswith("#") or lineno in doc_range:
            continue
        count += 1
    return count


def _node_is_suppressed(lines: list[str], node: ast.AST) -> bool:
    """True if the def/class header line carries the per-case suppression comment."""
    lineno = getattr(node, "lineno", 0)
    if not (1 <= lineno <= len(lines)):
        return False
    return _SUPPRESS in lines[lineno - 1]


def _check_source(path: Path, source: str) -> list[Violation]:
    """Apply the module/function/class ceilings to one already-read source file."""
    violations: list[Violation] = []
    lines = source.splitlines()

    module_lines = _significant_line_count(source)
    if module_lines > MODULE_MAX_LINES:
        violations.append(
            Violation(
                path=path,
                line=1,
                rule=_RULE,
                requirement=_REQ,
                message=(
                    f"module has {module_lines} significant lines "
                    f"(ceiling {MODULE_MAX_LINES}); decompose into focused modules"
                ),
                severity=Severity.BLOCKING,
            )
        )

    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            code_lines = _function_code_lines(lines, node)
            if code_lines > FUNCTION_MAX_LINES and not _node_is_suppressed(lines, node):
                violations.append(
                    Violation(
                        path=path,
                        line=node.lineno,
                        rule=_RULE,
                        requirement=_REQ,
                        message=(
                            f"function '{node.name}' has {code_lines} code lines "
                            f"(ceiling {FUNCTION_MAX_LINES}); extract helpers"
                        ),
                    )
                )
        elif isinstance(node, ast.ClassDef):
            end = node.end_lineno or node.lineno
            body_lines = _significant_span(lines, node.lineno, end)
            if body_lines > CLASS_MAX_LINES and not _node_is_suppressed(lines, node):
                violations.append(
                    Violation(
                        path=path,
                        line=node.lineno,
                        rule=_RULE,
                        requirement=_REQ,
                        message=(
                            f"class '{node.name}' has {body_lines} significant lines "
                            f"(ceiling {CLASS_MAX_LINES}); split responsibilities"
                        ),
                    )
                )
    return violations


def check_paths(paths: Iterable[Path]) -> list[Violation]:
    """Run the size-ceiling linter over every non-exempt ``*.py`` under `paths`."""
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        if _is_exempt(path):
            continue
        violations.extend(_check_source(path, path.read_text(encoding="utf-8")))
    return violations
