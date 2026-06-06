"""Content / copy gate for athlete-facing strings (QUAL-R13(j), CI-R1 item 21).

This linter mechanically enforces the regex/allowlist-checkable rules of QUAL-R13
(c)-(g) over the externalized, keyed user-facing copy catalog, plus the two
cross-reference gates the spec names:

  * BANNED-BLAME words (QUAL-R13(e)):  invalid|illegal|forbidden|prohibited|
    incorrect|you forgot|you failed  (permitted only inside machine code names,
    which never appear in catalog VALUE text).
  * BANNED-EDGY / robotic-hacky tokens (QUAL-R13(f)):  oops|whoops|uh-oh|yikes|
    gotcha|pwned  + leetspeak/meme tone.
  * `please` / `sorry` over-apology in routine error copy (QUAL-R13(e)).
  * `!` in error/validation/empty-state copy (QUAL-R13(f); reserved for milestones).
  * INTERNALS LEAK (QUAL-R13(e) + E2E-R4):  Traceback/Exception/SQL/JWT/hex-code
    AND developer jargon — source-descriptor names, DB/store words, and raw
    FIDELITY-ENUM values (``summary_only``/``modeled``/``measured``/``derived``/
    ``estimated``/``sensor``) leaking into an athlete-facing string.
  * CATALOG-KEY cross-ref (QUAL-R13(c)):  every error entry resolves a STABLE key
    and references a machine ``code``; a value with no key is an orphan.
  * ERROR-CODE REGISTRY uniqueness (QUAL-R13(d)/(j)):  each error ``code`` is unique
    across the catalog (clients branch on the stable code, ERR-R3).

Catalog shape (the linter reads ``*.copy.toml`` / ``*.copy.json`` under a
``locale``/``i18n``/``copy`` path; tables keyed by catalog key). Each athlete-facing
entry is a table with at least ``text`` (the wording) and, for error entries,
``code`` (the stable machine code). Entries whose ``kind`` is ``error``/
``validation``/``empty_state`` get the `!` and apology checks; all entries get the
banned-word + leak checks.

The non-mechanical dimensions (read-aloud naturalness, "is the fix helpful",
warmth) are a human-review checklist item (DELIV-R3), deliberately NOT gated here.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tools.lint.core import Violation, iter_python_files

_RULE_BLAME = "copy-banned-blame"
_RULE_EDGY = "copy-banned-edgy"
_RULE_APOLOGY = "copy-over-apology"
_RULE_BANG = "copy-exclamation"
_RULE_LEAK = "copy-internals-leak"
_RULE_ORPHAN = "copy-orphan-literal"
_RULE_KEY = "copy-catalog-key"
_RULE_CODE = "copy-error-code"
_REQ = "QUAL-R13"

# Catalog file suffixes + the path segments under which they live.
_CATALOG_SUFFIXES = (".copy.toml", ".copy.json")
_CATALOG_SEGMENTS = frozenset({"locale", "locales", "i18n", "copy"})

_ERRORLIKE_KINDS = frozenset({"error", "validation", "empty_state"})

# QUAL-R13(e) banned-blame vocabulary (word-boundary, case-insensitive).
_BLAME = re.compile(
    r"\b(?:invalid|illegal|forbidden|prohibited|incorrect)\b"
    r"|you\s+forgot|you\s+failed",
    re.IGNORECASE,
)
# QUAL-R13(f) banned-edgy / hacker-tone tokens.
_EDGY = re.compile(
    r"\b(?:oops|whoops|uh-?oh|yikes|gotcha|pwned)\b",
    re.IGNORECASE,
)
# QUAL-R13(e) routine over-apology.
_APOLOGY = re.compile(r"\b(?:please|sorry)\b", re.IGNORECASE)

# QUAL-R13(e)/E2E-R4 internals-leak: machine/transport internals + dev jargon.
_LEAK = re.compile(
    r"Traceback|Exception\b|\bSQL\b|\bSELECT\b|\bINSERT\b|"
    r"eyJ[A-Za-z0-9_\-]+\."  # JWT-ish
    r"|0x[0-9a-fA-F]{4,}"  # hex error code
    r"|\bGBO\b|\bASBO\b|\bMCP\b|\badapter\b|\bendpoint\b|\bschema\b|\btoken\b"
    r"|\bcheckpoint\b|\bvector store\b|\bdatabase\b|\bfidelity\b",
    re.IGNORECASE,
)
# Raw fidelity-enum values must never surface to an athlete (E2E-R4).
_FIDELITY_ENUM = re.compile(
    r"\b(?:summary_only|modeled|measured|derived|estimated|sensor)\b"
)


def _is_catalog(path: Path) -> bool:
    """True for a user-facing copy catalog file (by suffix + path segment)."""
    name = path.name
    if not any(name.endswith(suffix) for suffix in _CATALOG_SUFFIXES):
        return False
    return any(part in _CATALOG_SEGMENTS for part in path.parts)


def _load(path: Path) -> dict[str, Any]:
    """Parse a catalog file (TOML or JSON) into a dict of entries."""
    raw = path.read_text(encoding="utf-8")
    if path.name.endswith(".toml"):
        return tomllib.loads(raw)
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _scan_text(path: Path, key: str, entry: dict[str, Any]) -> list[Violation]:
    """Apply every word/leak rule to one catalog entry's athlete-facing text."""
    text = str(entry.get("text", ""))
    kind = str(entry.get("kind", "")).lower()
    errorlike = kind in _ERRORLIKE_KINDS
    out: list[Violation] = []

    def add(rule: str, msg: str) -> None:
        out.append(Violation(path=path, line=1, rule=rule, requirement=_REQ, message=msg))

    if _BLAME.search(text):
        add(_RULE_BLAME, f"key '{key}' uses blame language; the system/state is the subject")
    if _EDGY.search(text):
        add(_RULE_EDGY, f"key '{key}' uses edgy/hacky tone banned in user copy")
    if _LEAK.search(text) or _FIDELITY_ENUM.search(text):
        add(_RULE_LEAK, f"key '{key}' leaks internals/jargon/fidelity-enum to the athlete")
    if errorlike and _APOLOGY.search(text):
        add(_RULE_APOLOGY, f"key '{key}' over-apologises ('please'/'sorry') in a routine error")
    if errorlike and "!" in text:
        add(_RULE_BANG, f"key '{key}' uses '!' in error/validation/empty-state copy")
    return out


def _scan_registry(path: Path, catalog: dict[str, Any]) -> list[Violation]:
    """Cross-ref: error entries carry a key + a code; codes are unique (QUAL-R13d/j)."""
    out: list[Violation] = []
    seen_codes: dict[str, str] = {}
    for key, entry in catalog.items():
        if not isinstance(entry, dict):
            out.append(
                Violation(
                    path=path, line=1, rule=_RULE_KEY, requirement=_REQ,
                    message=(
                        f"key '{key}' is not a keyed table; user copy MUST "
                        f"resolve via a stable key"
                    ),
                )
            )
            continue
        out.extend(_scan_text(path, key, entry))
        kind = str(entry.get("kind", "")).lower()
        if kind not in _ERRORLIKE_KINDS:
            continue
        code = entry.get("code")
        if not code:
            out.append(
                Violation(
                    path=path, line=1, rule=_RULE_CODE, requirement=_REQ,
                    message=(
                        f"error key '{key}' has no stable machine 'code' (clients "
                        f"branch on the code, not the sentence)"
                    ),
                )
            )
            continue
        code = str(code)
        if code in seen_codes:
            out.append(
                Violation(
                    path=path, line=1, rule=_RULE_CODE, requirement=_REQ,
                    message=(
                        f"duplicate error code '{code}' (keys '{seen_codes[code]}' "
                        f"and '{key}'); each code MUST be unique in the registry"
                    ),
                )
            )
        else:
            seen_codes[code] = key
    return out


# --- inline orphan-literal detection (QUAL-R13(c): no user-facing literal in logic) ---

# Keyword args at an ATHLETE-FACING emission site that must resolve a catalog key
# rather than carry an inline sentence. Scoped deliberately to the RFC-9457 API
# problem fields (``title``/``detail``) plus explicit user-message kwargs. NOTE:
# analytics/domain ``Unavailable(detail=...)`` is a DEVELOPER-facing diagnostic
# field (QUAL-R13(a): the standard "does NOT govern internal logs, machine codes,
# or developer-facing strings"), so the scan is restricted to the API layer below
# to avoid flagging internal reason text — a false positive we explicitly reject.
_USERFACING_KWARGS = frozenset({"detail", "title", "user_message", "athlete_message"})
_APP_SEGMENT = "wattwise_core"
# Only these engine subpackages emit athlete-facing API copy through these kwargs.
_USERFACING_LAYERS = frozenset({"api"})


def _is_app_source(path: Path) -> bool:
    """True for the athlete-facing API layer where orphan user literals are forbidden.

    Restricted to ``api/**`` (the RFC-9457 problem surface). Analytics/domain/agent
    internal ``detail=`` reason fields are developer-facing and intentionally NOT in
    scope (QUAL-R13(a) excludes developer-facing strings).
    """
    parts = path.parts
    if _APP_SEGMENT not in parts:
        return False
    if path.name.startswith("test_") or "/tests/" in f"/{path.as_posix()}":
        return False
    if any(seg in parts for seg in _CATALOG_SEGMENTS):
        return False
    idx = parts.index(_APP_SEGMENT)
    subpackage = parts[idx + 1] if idx + 1 < len(parts) else ""
    return subpackage in _USERFACING_LAYERS


def _looks_like_sentence(value: str) -> bool:
    """Heuristic: a multi-word human sentence (vs. a catalog key or short token)."""
    return " " in value.strip() and len(value.strip()) >= 12


def _scan_orphans(path: Path, source: str) -> list[Violation]:
    """Flag athlete-facing keyword args assigned an inline sentence literal."""
    out: list[Violation] = []
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in _USERFACING_KWARGS:
                continue
            value = kw.value
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                continue
            if not _looks_like_sentence(value.value):
                continue
            out.append(
                Violation(
                    path=path,
                    line=value.lineno,
                    rule=_RULE_ORPHAN,
                    requirement=_REQ,
                    message=(
                        f"athlete-facing '{kw.arg}=' is an inline literal; user copy "
                        f"MUST resolve through the externalized keyed catalog, not logic"
                    ),
                )
            )
    return out


def check_paths(paths: Iterable[Path]) -> list[Violation]:
    """Run the content/copy gate over copy catalogs + scan app source for orphan literals."""
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        if _is_app_source(path):
            violations.extend(_scan_orphans(path, path.read_text(encoding="utf-8")))
    for path in _iter_catalogs(paths):
        violations.extend(_scan_registry(path, _load(path)))
    return violations


def _iter_catalogs(paths: Iterable[Path]) -> list[Path]:
    """Discover copy-catalog files (``*.copy.toml`` / ``*.copy.json``) under `paths`."""
    found: set[Path] = set()
    for raw in paths:
        root = raw.resolve()
        if root.is_file():
            if _is_catalog(root):
                found.add(root)
            continue
        for suffix in _CATALOG_SUFFIXES:
            for candidate in root.rglob(f"*{suffix}"):
                if _is_catalog(candidate):
                    found.add(candidate)
    return sorted(found)
