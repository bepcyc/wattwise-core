"""English-only source linter (QUAL-R11, CI-R1 item 14).

Two checks of DIFFERENT severity:

  (a) Non-ASCII IDENTIFIER check  -> BLOCKING / mechanical (QUAL-R11(a)).
      Any non-ASCII character in a module/function/class/variable/parameter name
      outside the exemption zones fails the build. This is purely mechanical and
      has no false positives.

  (b) Non-English PROSE detection -> ADVISORY / review-assisted (QUAL-R11(b)).
      A heuristic flags non-English natural-language prose in comments, docstrings,
      and structured-log ``message=`` strings. It is reported as a review WARNING
      and NEVER fails the build (automated language detectors produce false
      positives on technical jargon and short strings; a hard gate would be flaky).

Three exemption zones (excluded from BOTH checks, QUAL-R11):
  (a) i18n / locale content under a ``locale/`` or ``i18n/`` path;
  (b) external LLM prompt / persona / skill / grounding-rule config (``prompt``/
      ``persona``/``skill``/``playbook`` path segments — these live in the runtime
      config bundle, DELIV-R2);
  (c) multilingual test fixtures / eval datasets (``fixture``/``fixtures``/
      ``eval`` dataset paths, QA-EVAL-R1 / GOLD-R1 / INJ-R1).

Per-line suppression: append ``# i18n-ok`` (or ``# noqa: english-only``) to a line
to silence both checks for that physical line (an intentional non-English token).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

from tools.lint.core import Severity, Violation, iter_python_files

_RULE_ID = "english-only-identifier"
_RULE_PROSE = "english-only-prose"
_REQ = "QUAL-R11"

_SUPPRESS_TOKENS = ("# i18n-ok", "# noqa: english-only")

# Path segments that mark an exemption zone (QUAL-R11 zones a/b/c).
_EXEMPT_SEGMENTS = frozenset(
    {
        "locale",
        "locales",
        "i18n",
        "prompt",
        "prompts",
        "persona",
        "personas",
        "skill",
        "skills",
        "playbook",
        "playbooks",
        "fixture",
        "fixtures",
        "datasets",
    }
)

_NON_ASCII = re.compile(r"[^\x00-\x7f]")

# A handful of Cyrillic/CJK ranges are a reliable "definitely not English" signal
# for the ADVISORY prose heuristic. Latin-with-diacritics (e.g. German umlauts) is
# deliberately NOT treated as non-English prose here — too false-positive-prone
# (QUAL-R11(b)); the blocking identifier check still catches non-ASCII identifiers.
_NON_ENGLISH_PROSE = re.compile(r"[Ѐ-ӿ぀-ヿ一-鿿가-힯]")  # i18n-ok: literal range table


def _is_exempt_path(path: Path) -> bool:
    """True when a file sits inside any QUAL-R11 exemption zone (by path segment)."""
    return any(part in _EXEMPT_SEGMENTS for part in path.parts)


def _line_suppressed(lines: list[str], lineno: int) -> bool:
    """True when the given 1-based physical line carries a per-line suppression token."""
    if not (1 <= lineno <= len(lines)):
        return False
    raw = lines[lineno - 1]
    return any(token in raw for token in _SUPPRESS_TOKENS)


def _check_identifiers(path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
    """BLOCKING: flag any non-ASCII identifier (QUAL-R11(a))."""
    violations: list[Violation] = []
    for node in ast.walk(tree):
        for name, lineno in _identifier_sites(node):
            if not _NON_ASCII.search(name):
                continue
            if _line_suppressed(lines, lineno):
                continue
            bad = _NON_ASCII.search(name)
            char = bad.group(0) if bad else "?"
            violations.append(
                Violation(
                    path=path,
                    line=lineno,
                    rule=_RULE_ID,
                    requirement=_REQ,
                    message=(
                        f"identifier {name!r} contains non-ASCII character {char!r}; "
                        f"engine identifiers MUST be English/ASCII"
                    ),
                    severity=Severity.BLOCKING,
                )
            )
    return violations


def _identifier_sites(node: ast.AST) -> list[tuple[str, int]]:
    """Yield ``(name, lineno)`` for every binding-identifier introduced by `node`.

    Covers module/function/class names, parameters, and assignment/loop targets —
    the names QUAL-R11(a) governs. Attribute and reference loads are skipped (a
    reference to an external non-ASCII name would be a third-party import problem,
    not an identifier *we* declared).
    """
    sites: list[tuple[str, int]] = []
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        sites.append((node.name, node.lineno))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            for arg in (
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                *([args.vararg] if args.vararg else []),
                *([args.kwarg] if args.kwarg else []),
            ):
                sites.append((arg.arg, arg.lineno))
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        sites.append((node.id, node.lineno))
    return sites


def _string_nodes(tree: ast.AST) -> list[ast.Constant]:
    """Collect string-literal constant nodes (docstring + log-message candidates)."""
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _check_prose(path: Path, source: str, tree: ast.AST, lines: list[str]) -> list[Violation]:
    """ADVISORY: heuristically flag non-English prose in comments + strings (R11(b))."""
    violations: list[Violation] = []

    for idx, raw in enumerate(lines, start=1):
        comment = _comment_text(raw)
        if comment and _NON_ENGLISH_PROSE.search(comment) and not _line_suppressed(lines, idx):
            violations.append(_prose_violation(path, idx))

    for node in _string_nodes(tree):
        value = node.value
        if not isinstance(value, str):  # narrowing for the type checker
            continue
        if not _NON_ENGLISH_PROSE.search(value):
            continue
        lineno = node.lineno
        if _line_suppressed(lines, lineno):
            continue
        violations.append(_prose_violation(path, lineno))
    return violations


def _comment_text(raw: str) -> str:
    """Return the inline-comment portion of a physical line, or '' if none.

    Naive ``#`` split: good enough for the ADVISORY heuristic. A ``#`` inside a
    string literal may yield a false comment, but since this layer never blocks the
    build (QUAL-R11(b)) the imprecision is acceptable and documented.
    """
    hash_pos = raw.find("#")
    return raw[hash_pos + 1 :] if hash_pos != -1 else ""


def _prose_violation(path: Path, line: int) -> Violation:
    """Build an ADVISORY non-English-prose finding for `path:line`."""
    return Violation(
        path=path,
        line=line,
        rule=_RULE_PROSE,
        requirement=_REQ,
        message=(
            "possible non-English prose in a comment/docstring/log message; "
            "resolve in review (advisory, not a build failure)"
        ),
        severity=Severity.ADVISORY,
    )


def check_paths(paths: Iterable[Path]) -> list[Violation]:
    """Run both English-only checks over every non-exempt ``*.py`` under `paths`."""
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        if _is_exempt_path(path):
            continue
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        violations.extend(_check_identifiers(path, tree, lines))
        violations.extend(_check_prose(path, source, tree, lines))
    return violations
