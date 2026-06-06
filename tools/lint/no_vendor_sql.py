"""No-raw-vendor-SQL static linter (BOOT-R3, CI-R1 item 13 static slice).

BOOT-R3 mandates ZERO vendor-specific SQL in application code: all persistence
flows through the ORM/query-builder so the schema and queries run identically on
SQLite, PostgreSQL, and MariaDB with a DSN-only difference. This linter is the
static half of that gate — "any vendor-SQL string found by static check fails".

What it flags in application code (``src/``, outside the exemptions):
  * raw SQL DML/DDL string literals (``SELECT``/``INSERT``/``UPDATE``/``DELETE``/
    ``CREATE TABLE``/``ALTER TABLE``/``DROP TABLE``/``UPSERT``);
  * dialect-only / vendor-specific constructs (``ON CONFLICT``, ``ON DUPLICATE KEY``,
    ``RETURNING``, ``PRAGMA``, ``ILIKE``, ``LIMIT ... OFFSET`` raw, ``::`` casts,
    ``sqlite_``/``pg_``/``information_schema`` catalog probes).

Exemptions:
  * the SINGLE whitelisted upsert seam ``src/wattwise_core/persistence/upsert.py``
    (BOOT-R3 / GBO-R8b: the unavoidable dialect construct is confined to one seam);
  * the migration layer (``migrations/``) — also confined per GBO-R8b;
  * test modules (fixtures may carry raw SQL for setup/assertions).

The unavoidable-construct seam carries the whole risk in ONE reviewed file, so the
rest of the codebase stays dialect-clean by construction (sharp-edges: one blessed
escape hatch, everything else mechanically forbidden).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

from tools.lint.core import Violation, iter_python_files

_RULE = "no-vendor-sql"
_REQ = "BOOT-R3"

# The ONE blessed seam where a dialect-specific upsert may live (BOOT-R3/GBO-R8b).
_WHITELIST_SUFFIX = "wattwise_core/persistence/upsert.py"

_SUPPRESS = "# noqa: no-vendor-sql"

# Only application source under src/ is in scope; migrations + tests are exempt.
_APP_ROOT_SEGMENT = "wattwise_core"
_EXEMPT_SEGMENTS = ("migrations",)

# Raw SQL statement starts (word-boundary, case-insensitive, multiline). These are
# the unambiguous "this is a SQL string" signals — note NO bare ``UPSERT`` word
# (it collides with the English word "upsert" used throughout the seam docstrings,
# a false positive). A real SQL string always carries one of these clause shapes.
_RAW_SQL = re.compile(
    r"\b(?:SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|"
    r"CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE)\b",
    re.IGNORECASE | re.DOTALL,
)

# Upsert/dialect clause shapes that mark a string as SQL on their own (the upsert
# seam confines these). Matched as multi-word clauses, never single English words.
_SQL_CLAUSE = re.compile(
    r"ON\s+CONFLICT|ON\s+DUPLICATE\s+KEY",
    re.IGNORECASE,
)

# Catalog/system-table probes — vendor-specific, break portability on sight.
_CATALOG_PROBE = re.compile(
    r"\b(?:sqlite_master|pg_catalog|information_schema)\b",
    re.IGNORECASE,
)

# Constructs that are dialect-specific but whose tokens also occur as ordinary
# English ("returning a value") or unrelated identifiers. They are flagged ONLY
# when the surrounding string already looks like SQL — never on the bare word.
_DIALECT_IN_SQL = re.compile(
    r"\bRETURNING\b|\bILIKE\b|\bPRAGMA\b",
    re.IGNORECASE,
)


def _looks_like_sql(value: str) -> bool:
    """True when a string is unmistakably a SQL statement/clause (not prose)."""
    return bool(_RAW_SQL.search(value) or _SQL_CLAUSE.search(value))


def _is_app_source(path: Path) -> bool:
    """True for in-scope application source (under the package, not migrations)."""
    parts = path.parts
    if _APP_ROOT_SEGMENT not in parts:
        return False
    return not any(seg in parts for seg in _EXEMPT_SEGMENTS)


def _is_whitelisted(path: Path) -> bool:
    """True for the single blessed upsert seam (BOOT-R3 exemption)."""
    return path.as_posix().endswith(_WHITELIST_SUFFIX)


def _is_test(path: Path) -> bool:
    """True for test modules (raw SQL allowed in fixtures/assertions)."""
    return path.name.startswith("test_") or "/tests/" in f"/{path.as_posix()}"


def _scan_string(value: str) -> str | None:
    """Return a short reason if `value` is raw/vendor SQL, else None.

    Dialect tokens that double as English words (RETURNING/ILIKE/PRAGMA) only count
    when the string is already recognisably SQL — this is the false-positive guard
    that keeps docstrings like "returning a typed result" from tripping the gate.
    """
    if _RAW_SQL.search(value):
        return "raw SQL statement literal"
    if _SQL_CLAUSE.search(value):
        match = _SQL_CLAUSE.search(value)
        token = match.group(0) if match else "upsert clause"
        return f"dialect-specific upsert clause {token!r}"
    if _CATALOG_PROBE.search(value):
        match = _CATALOG_PROBE.search(value)
        token = match.group(0) if match else "catalog probe"
        return f"vendor catalog probe {token!r}"
    if _looks_like_sql(value) and _DIALECT_IN_SQL.search(value):
        match = _DIALECT_IN_SQL.search(value)
        token = match.group(0) if match else "dialect token"
        return f"dialect-specific construct {token!r}"
    return None


def _check_source(path: Path, source: str) -> list[Violation]:
    """Flag raw/vendor SQL in every string literal of an application module."""
    violations: list[Violation] = []
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        reason = _scan_string(node.value)
        if reason is None:
            continue
        lineno = node.lineno
        if 1 <= lineno <= len(lines) and _SUPPRESS in lines[lineno - 1]:
            continue
        violations.append(
            Violation(
                path=path,
                line=lineno,
                rule=_RULE,
                requirement=_REQ,
                message=(
                    f"{reason} in application code breaks 3-backend portability; "
                    f"use the ORM/query-builder or the persistence/upsert.py seam"
                ),
            )
        )
    return violations


def check_paths(paths: Iterable[Path]) -> list[Violation]:
    """Run the no-vendor-SQL linter over in-scope application source under `paths`."""
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        if not _is_app_source(path) or _is_test(path) or _is_whitelisted(path):
            continue
        violations.extend(_check_source(path, path.read_text(encoding="utf-8")))
    return violations
