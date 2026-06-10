"""Golden/snapshot tests for the custom lint pack (false-positive control).

Each linter is fed a KNOWN-BAD and a KNOWN-GOOD sample written into a temp tree;
the test asserts the linter flags EXACTLY the bad one and stays silent on the good
one. This pins both directions — a missed violation (false negative) and a spurious
one (false positive) — which is the whole point of a lint gate (QUAL-R9/R10/R11/R13,
ARCH-R21/R22, BOOT-R3). Tests are tier T-UNIT (offline, fixture-only).

These tests intentionally embed banned/edgy/vendor-SQL/non-ASCII content as STRING
fixtures inside helper-written files; the lint pack exempts ``tests/`` paths, so the
fixtures here never trip the gate on the real tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The custom lint pack lives in the repo-root ``tools/`` namespace (not an installed
# package and not under ``src/``), so ensure the repo root is importable when pytest
# collects this module regardless of its rootdir/import-mode. ``tools`` is run in
# production as ``python -m tools.lint`` with the repo root already on ``sys.path``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.lint import (  # noqa: E402  (import after sys.path bootstrap, intentional)
    content_copy,
    content_leak,
    core,
    english_only,
    import_direction,
    no_vendor_sql,
    size_limits,
    test_docstrings,
)
from tools.lint.core import Severity  # noqa: E402
from tools.lint.runner import collect  # noqa: E402

from wattwise_core.ingestion.registry import load_registry  # noqa: E402

pytestmark = pytest.mark.unit


def _write(base: Path, rel: str, body: str) -> Path:
    """Write `body` to ``base/rel`` (creating parents) and return the file path."""
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _rules(violations: list[core.Violation]) -> set[str]:
    """Collapse a violation list to the set of rule ids it contains."""
    return {v.rule for v in violations}


# --------------------------------------------------------------------------- size


def test_size_limits_flags_oversized_function_only(tmp_path: Path) -> None:
    """size-limits flags a >60-line function but leaves a compact one untouched."""
    long_body = "\n".join(f"    x{i} = {i}" for i in range(80))
    bad = _write(
        tmp_path / "src" / "wattwise_core" / "analytics",
        "huge.py",
        f'"""Mod."""\n\n\ndef compute() -> int:\n{long_body}\n    return x0\n',
    )
    good = _write(
        tmp_path / "src" / "wattwise_core" / "analytics",
        "small.py",
        '"""Mod."""\n\n\ndef compute() -> int:\n    """Doc."""\n    return 1\n',
    )
    found = size_limits.check_paths([tmp_path])
    flagged = {v.path for v in found}
    assert bad in flagged
    assert good not in flagged
    assert all(v.severity is Severity.BLOCKING for v in found)


def test_size_limits_exempts_test_modules(tmp_path: Path) -> None:
    """An oversized function inside a test module is exempt from the size ceiling."""
    long_body = "\n".join(f"    x{i} = {i}" for i in range(80))
    _write(
        tmp_path / "tests",
        "test_big.py",
        f'"""Mod."""\n\n\ndef test_thing() -> None:\n    """Doc."""\n{long_body}\n',
    )
    assert size_limits.check_paths([tmp_path]) == []


def test_size_limits_honors_suppression_comment(tmp_path: Path) -> None:
    """A documented per-case suppression comment silences one over-limit function."""
    long_body = "\n".join(f"    x{i} = {i}" for i in range(80))
    _write(
        tmp_path / "src" / "wattwise_core" / "analytics",
        "ok_big.py",
        f'"""Mod."""\n\n\ndef compute() -> int:  # noqa: size-limits\n{long_body}\n    return x0\n',
    )
    assert size_limits.check_paths([tmp_path]) == []


# ---------------------------------------------------------------------- docstrings


def test_test_docstrings_flags_missing_and_echo_only(tmp_path: Path) -> None:
    """test-docstrings flags a no-docstring and an echo-only test, passes a real one."""
    real = 'def test_real() -> None:\n    """CTL rises when load appended (ANL-T)."""\n    pass\n'
    module = (
        '"""Mod."""\n\n\n'
        "def test_no_doc() -> None:\n    assert True\n\n\n"
        'def test_echo() -> None:\n    """test echo"""\n    assert True\n\n\n' + real
    )
    path = _write(tmp_path / "tests", "test_sample.py", module)
    found = test_docstrings.check_paths([tmp_path])
    lines = {v.line for v in found if v.path == path}
    # test_no_doc starts at line 4, test_echo at line 8, test_real at line 13.
    assert 4 in lines  # missing docstring
    assert 8 in lines  # echo-only docstring
    assert 13 not in lines  # real docstring -> not flagged


def test_test_docstrings_ignores_non_test_functions(tmp_path: Path) -> None:
    """A non-test helper function without a docstring is not flagged by the rule."""
    _write(
        tmp_path / "tests",
        "test_helpers.py",
        '"""Mod."""\n\n\ndef helper() -> int:\n    return 1\n\n\n'
        'def test_uses_helper() -> None:\n    """Helper returns one (sanity)."""\n'
        "    assert helper() == 1\n",
    )
    assert test_docstrings.check_paths([tmp_path]) == []


# ------------------------------------------------------------------- english-only


def test_english_only_blocks_non_ascii_identifier(tmp_path: Path) -> None:
    """A non-ASCII identifier is a BLOCKING violation (QUAL-R11a, mechanical)."""
    # Intentional Cyrillic letter in the function name (the thing under test).
    bad = _write(
        tmp_path / "src" / "wattwise_core" / "domain",
        "bad.py",
        '"""Mod."""\n\n\ndef сompute() -> int:\n    """Doc."""\n    return 1\n',  # noqa: RUF001
    )
    found = english_only.check_paths([tmp_path])
    blocking = [v for v in found if v.severity is Severity.BLOCKING]
    assert any(v.path == bad and v.rule == "english-only-identifier" for v in blocking)


def test_english_only_prose_is_advisory_not_blocking(tmp_path: Path) -> None:
    """Non-English prose in a comment is ADVISORY only — never fails the gate (R11b)."""
    _write(
        tmp_path / "src" / "wattwise_core" / "domain",
        "prose.py",
        '"""Mod."""\n\n\ndef compute() -> int:\n'
        "    # this comment is intentionally written in russian below\n"
        "    # русский комментарий\n"
        '    """Doc."""\n    return 1\n',
    )
    found = english_only.check_paths([tmp_path])
    prose = [v for v in found if v.rule == "english-only-prose"]
    assert prose
    assert all(v.severity is Severity.ADVISORY for v in prose)
    assert [v for v in found if v.severity is Severity.BLOCKING] == []


def test_english_only_exempts_locale_and_prompt_zones(tmp_path: Path) -> None:
    """Non-ASCII identifiers/prose in locale + prompt zones are fully exempt (R11)."""
    body = (
        '"""Mod."""\n\n\ndef сompute() -> int:\n'  # noqa: RUF001
        "    # русский\n"
        '    """Doc."""\n    return 1\n'
    )
    _write(tmp_path / "src" / "wattwise_core" / "config" / "locale", "de.py", body)
    _write(tmp_path / "src" / "wattwise_core" / "agent" / "prompts", "persona.py", body)
    assert english_only.check_paths([tmp_path]) == []


def test_english_only_honors_line_suppression(tmp_path: Path) -> None:
    """A per-line `# i18n-ok` token silences both checks for that physical line."""
    _write(
        tmp_path / "src" / "wattwise_core" / "domain",
        "supp.py",
        '"""Mod."""\n\n\ndef compute() -> int:\n'
        '    table = {"µs": 1}  # i18n-ok: unit symbol\n'
        '    """Doc."""\n    return table["µs"]\n',
    )
    found = english_only.check_paths([tmp_path])
    # The micro-sign string is Latin-1 non-ASCII but in a literal, not an identifier;
    # and the line is suppressed regardless — assert no BLOCKING finding here.
    assert [v for v in found if v.severity is Severity.BLOCKING] == []


# ----------------------------------------------------------------- no-vendor-sql


def test_no_vendor_sql_flags_raw_and_dialect_sql(tmp_path: Path) -> None:
    """no-vendor-sql flags a raw SELECT and an ON CONFLICT dialect construct."""
    bad = _write(
        tmp_path / "src" / "wattwise_core" / "domain",
        "queries.py",
        '"""Mod."""\n\n\n'
        'RAW = "SELECT id FROM activity WHERE athlete_id = :a"\n'
        'UP = "INSERT INTO daily VALUES (1) ON CONFLICT DO NOTHING"\n',
    )
    found = no_vendor_sql.check_paths([tmp_path])
    assert {v.line for v in found if v.path == bad} == {4, 5}


def test_no_vendor_sql_whitelists_upsert_seam(tmp_path: Path) -> None:
    """The single blessed persistence/upsert.py seam may carry dialect SQL (BOOT-R3)."""
    _write(
        tmp_path / "src" / "wattwise_core" / "persistence",
        "upsert.py",
        '"""Upsert seam."""\n\n\n'
        'PG = "INSERT INTO daily VALUES (1) ON CONFLICT (id) DO UPDATE SET v = 2"\n',
    )
    assert no_vendor_sql.check_paths([tmp_path]) == []


def test_no_vendor_sql_ignores_orm_code(tmp_path: Path) -> None:
    """Portable ORM/query-builder usage (no raw SQL string) is not flagged."""
    _write(
        tmp_path / "src" / "wattwise_core" / "domain",
        "repo.py",
        '"""Mod."""\n\n\n'
        "def recent(session: object) -> list[int]:\n"
        '    """Return recent ids via the ORM."""\n'
        "    return [1, 2, 3]\n",
    )
    assert no_vendor_sql.check_paths([tmp_path]) == []


# -------------------------------------------------------------- import-direction


def test_import_direction_flags_outward_import(tmp_path: Path) -> None:
    """import-direction flags an L5 analytics module importing an L6 api module."""
    pkg = tmp_path / "src" / "wattwise_core"
    bad = _write(
        pkg / "analytics",
        "calc.py",
        '"""Mod."""\n\nfrom wattwise_core.api import routers\n\n\n'
        'def compute() -> int:\n    """Doc."""\n    return routers.x\n',
    )
    found = import_direction.check_paths([tmp_path])
    assert any(v.path == bad and v.rule == "import-direction" for v in found)


def test_import_direction_allows_inward_import(tmp_path: Path) -> None:
    """An L6 api module importing L5 analytics is inward and allowed."""
    pkg = tmp_path / "src" / "wattwise_core"
    _write(
        pkg / "api",
        "router.py",
        '"""Mod."""\n\nfrom wattwise_core.analytics import calc\n\n\n'
        'def handler() -> int:\n    """Doc."""\n    return calc.x\n',
    )
    found = import_direction.check_paths([tmp_path])
    assert [v for v in found if v.rule == "import-direction"] == []


def test_import_direction_flags_source_specific_adapter_import(tmp_path: Path) -> None:
    """A consumer importing a named source adapter violates the no-source-branch rule."""
    pkg = tmp_path / "src" / "wattwise_core"
    bad = _write(
        pkg / "ingestion",
        "service.py",
        '"""Mod."""\n\n'
        "from wattwise_core.ingestion.adapters import intervals_icu\n\n\n"
        'def run() -> int:\n    """Doc."""\n    return intervals_icu.x\n',
    )
    found = import_direction.check_paths([tmp_path])
    assert any(v.path == bad and v.rule == "source-name-import" for v in found)


def test_import_direction_allows_adapter_registry_import(tmp_path: Path) -> None:
    """Importing the neutral adapter base/registry (not a named source) is allowed."""
    pkg = tmp_path / "src" / "wattwise_core"
    _write(
        pkg / "ingestion",
        "service.py",
        '"""Mod."""\n\n'
        "from wattwise_core.ingestion.adapters import base\n\n\n"
        'def run() -> int:\n    """Doc."""\n    return base.x\n',
    )
    found = import_direction.check_paths([tmp_path])
    assert [v for v in found if v.rule == "source-name-import"] == []


# ------------------------------------------------- source-name literal scan (ARCH-R2/R22)


def _registered_source_name() -> str:
    """A real registered source name (registry-derived, never hardcoded — CFG-R1a)."""
    keys = load_registry().source_keys()
    assert keys, "the OSS registry must expose at least one registered source name"
    return keys[0]


def test_source_literal_flags_control_flow_branch_on_registered_name(tmp_path: Path) -> None:
    """A consumer branching on a REGISTERED source-name literal in logic is flagged (ARCH-R2)."""
    name = _registered_source_name()
    pkg = tmp_path / "src" / "wattwise_core"
    bad = _write(
        pkg / "analytics",
        "service.py",
        '"""Mod."""\n\n\n'
        "def pick(source_key: str) -> int:\n"
        '    """Doc."""\n'
        f'    if source_key == "{name}":\n'
        "        return 1\n"
        "    return 0\n",
    )
    found = import_direction.check_paths([tmp_path])
    assert any(v.path == bad and v.rule == "source-name-literal" for v in found)
    hit = next(v for v in found if v.path == bad and v.rule == "source-name-literal")
    assert hit.requirement == "ARCH-R2"


def test_source_literal_clean_when_no_registered_name_literal(tmp_path: Path) -> None:
    """A source-agnostic consumer (selects via the registry/key argument) is silent."""
    pkg = tmp_path / "src" / "wattwise_core"
    _write(
        pkg / "analytics",
        "service.py",
        '"""Mod."""\n\n\n'
        "def pick(source_key: str, registry: object) -> object:\n"
        '    """Doc."""\n'
        "    return registry.get(source_key)\n",
    )
    found = import_direction.check_paths([tmp_path])
    assert [v for v in found if v.rule == "source-name-literal"] == []


def test_source_literal_allows_unregistered_string(tmp_path: Path) -> None:
    """A generic/unregistered string equal to no registered source name is allowed."""
    pkg = tmp_path / "src" / "wattwise_core"
    _write(
        pkg / "analytics",
        "service.py",
        '"""Mod."""\n\n\n'
        "def pick(kind: str) -> int:\n"
        '    """Doc."""\n'
        '    if kind == "definitely_not_a_registered_source":\n'
        "        return 1\n"
        "    return 0\n",
    )
    found = import_direction.check_paths([tmp_path])
    assert [v for v in found if v.rule == "source-name-literal"] == []


def test_source_literal_allows_adapter_package(tmp_path: Path) -> None:
    """An L2 adapter module MAY embed its own source name — scan is OUTSIDE adapters (ARCH-R2)."""
    name = _registered_source_name()
    pkg = tmp_path / "src" / "wattwise_core"
    _write(
        pkg / "ingestion" / "adapters",
        "some_source.py",
        '"""Mod."""\n\n\n'
        "def claim(source_key: str) -> int:\n"
        '    """Doc."""\n'
        f'    if source_key == "{name}":\n'
        "        return 1\n"
        "    return 0\n",
    )
    found = import_direction.check_paths([tmp_path])
    assert [v for v in found if v.rule == "source-name-literal"] == []


def test_source_literal_exempts_connections_sync_datahealth_and_registry(tmp_path: Path) -> None:
    """The AUTH-R15 surfaces carry source identity as runtime DATA — exempt (ARCH-R2/R22)."""
    name = _registered_source_name()
    pkg = tmp_path / "src" / "wattwise_core"
    body = (
        '"""Mod."""\n\n\n'
        "def reach(source_key: str) -> str:\n"
        '    """Doc."""\n'
        f'    if source_key == "{name}":\n'
        f'        return "we could not reach {name}"\n'
        '    return "ok"\n'
    )
    exempt = [
        _write(pkg / "api" / "routers", "connections.py", body),
        _write(pkg / "api" / "routers", "sync.py", body),
        _write(pkg / "api" / "routers", "data_health.py", body),
        _write(pkg / "api", "connection_catalog.py", body),
        _write(pkg / "ingestion", "registry.py", body),
    ]
    found = import_direction.check_paths([tmp_path])
    flagged = {v.path for v in found if v.rule == "source-name-literal"}
    assert flagged.isdisjoint(set(exempt))


def test_source_literal_ignores_docstring_mention_of_registered_name(tmp_path: Path) -> None:
    """A registered name in a docstring is prose, not control flow/logic — not flagged."""
    name = _registered_source_name()
    pkg = tmp_path / "src" / "wattwise_core"
    _write(
        pkg / "analytics",
        "service.py",
        f'"""Mod that documents the {name} source as an example."""\n\n\n'
        "def pick(source_key: str, registry: object) -> object:\n"
        f'    """The built-in {name} importer registers one descriptor."""\n'
        "    return registry.get(source_key)\n",
    )
    found = import_direction.check_paths([tmp_path])
    assert [v for v in found if v.rule == "source-name-literal"] == []


# ------------------------------------------------------------------ content-copy


def _catalog(base: Path, name: str, body: str) -> Path:
    """Write a copy-catalog file under a ``locale/`` zone and return its path."""
    return _write(base / "src" / "wattwise_core" / "config" / "locale", name, body)


def test_content_copy_flags_blame_edgy_and_leak(tmp_path: Path) -> None:
    """content-copy flags blame words, edgy tokens, and an internals/jargon leak."""
    body = (
        '[bad_blame]\nkind = "error"\ncode = "E_BLAME"\n'
        'text = "You failed to enter a valid date."\n\n'
        '[bad_edgy]\nkind = "error"\ncode = "E_EDGY"\n'
        'text = "Oops, that broke."\n\n'
        '[bad_leak]\nkind = "error"\ncode = "E_LEAK"\n'
        'text = "The database adapter returned fidelity summary_only."\n'
    )
    _catalog(tmp_path, "en.copy.toml", body)
    rules = _rules(content_copy.check_paths([tmp_path]))
    assert "copy-banned-blame" in rules
    assert "copy-banned-edgy" in rules
    assert "copy-internals-leak" in rules


def test_content_copy_flags_apology_and_exclamation(tmp_path: Path) -> None:
    """content-copy flags routine over-apology and a `!` in error/empty-state copy."""
    body = (
        '[apology]\nkind = "error"\ncode = "E_AP"\n'
        'text = "Sorry, we could not save your ride."\n\n'
        '[bang]\nkind = "validation"\ncode = "E_BANG"\n'
        'text = "Enter a date in the past!"\n'
    )
    _catalog(tmp_path, "en.copy.toml", body)
    rules = _rules(content_copy.check_paths([tmp_path]))
    assert "copy-over-apology" in rules
    assert "copy-exclamation" in rules


def test_content_copy_flags_duplicate_error_code(tmp_path: Path) -> None:
    """content-copy flags a duplicate machine error code in the registry (uniqueness)."""
    body = (
        '[first]\nkind = "error"\ncode = "E_DUP"\ntext = "We could not reach the source."\n\n'
        '[second]\nkind = "error"\ncode = "E_DUP"\ntext = "We could not load your week."\n'
    )
    _catalog(tmp_path, "en.copy.toml", body)
    rules = _rules(content_copy.check_paths([tmp_path]))
    assert "copy-error-code" in rules


def test_content_copy_flags_missing_error_code(tmp_path: Path) -> None:
    """An error entry with no stable machine code is flagged (clients branch on code)."""
    body = '[no_code]\nkind = "error"\ntext = "We could not save your ride right now."\n'
    _catalog(tmp_path, "en.copy.toml", body)
    rules = _rules(content_copy.check_paths([tmp_path]))
    assert "copy-error-code" in rules


def test_content_copy_passes_clean_catalog(tmp_path: Path) -> None:
    """A calm, jargon-free, uniquely-coded catalog produces no findings (no false pos)."""
    body = (
        '[source_unreachable]\nkind = "error"\ncode = "E_SOURCE_UNREACHABLE"\n'
        'text = "We could not reach your training source. '
        'Your numbers still work with what we have."\n\n'
        '[date_in_future]\nkind = "validation"\ncode = "E_DATE_FUTURE"\n'
        'text = "Choose a date that has already happened, then try again."\n\n'
        '[empty_week]\nkind = "empty_state"\ncode = "E_EMPTY_WEEK"\n'
        'text = "No rides yet this week. Connect a source to see your training load."\n'
    )
    _catalog(tmp_path, "en.copy.toml", body)
    assert content_copy.check_paths([tmp_path]) == []


def test_content_copy_flags_orphan_inline_user_literal(tmp_path: Path) -> None:
    """An inline athlete-facing sentence literal in logic is an orphan (QUAL-R13c)."""
    _write(
        tmp_path / "src" / "wattwise_core" / "api",
        "errors.py",
        '"""Mod."""\n\n\n'
        "def boom() -> object:\n"
        '    """Build a problem response."""\n'
        '    return make_problem(detail="We could not save your ride right now.")\n',
    )
    rules = _rules(content_copy.check_paths([tmp_path]))
    assert "copy-orphan-literal" in rules


def test_content_copy_allows_catalog_key_reference(tmp_path: Path) -> None:
    """Resolving copy via a catalog-key reference (not an inline sentence) is clean."""
    _write(
        tmp_path / "src" / "wattwise_core" / "api",
        "errors.py",
        '"""Mod."""\n\n\n'
        "def boom() -> object:\n"
        '    """Build a problem response."""\n'
        '    return make_problem(detail=catalog["source_unreachable"])\n',
    )
    orphans = [v for v in content_copy.check_paths([tmp_path]) if v.rule == "copy-orphan-literal"]
    assert orphans == []


# ----------------------------------------------------------------- content-leak (ARCH-R29)


def test_content_leak_flags_inline_named_prompt_body(tmp_path: Path) -> None:
    """ARCH-R29: a prompt body assigned to a behavior-asset NAME in engine source is flagged."""
    _write(
        tmp_path / "src" / "wattwise_core" / "agent",
        "leaky.py",
        '"""Mod."""\n\n\n'
        "_REFLECT_SYSTEM = (\n"
        '    "You are the coaching agent\'s reflection step. Decide the next move over the "\n'
        '    "closed verdict set and never invent a capability."\n'
        ")\n",
    )
    leaks = [v for v in content_leak.check_paths([tmp_path]) if v.rule == "content-leak-prompt"]
    assert leaks, "an inline named prompt body must be flagged (ARCH-R29)"
    assert leaks[0].requirement == "ARCH-R29"


def test_content_leak_flags_unnamed_persona_prose_literal(tmp_path: Path) -> None:
    """ARCH-R29: a long persona/prompt prose literal under an innocuous name is still flagged."""
    _write(
        tmp_path / "src" / "wattwise_core" / "agent",
        "sneaky.py",
        '"""Mod."""\n\n\n'
        "X = \"You are the athlete's endurance coach; answer only from the canonical data and "
        'never call this a readiness score, return only the structured narration."\n',
    )
    rules = _rules(content_leak.check_paths([tmp_path]))
    assert "content-leak-prompt" in rules


def test_content_leak_flags_short_multiline_named_persona_body(tmp_path: Path) -> None:
    """ARCH-R29: even a SHORT (<40-char) multi-line body under a behaviour-asset NAME is flagged.

    Guards the false-negative the bare-length threshold missed: a multi-line persona body whose
    stripped length is under the char threshold is still a leaked body (multi-line signal trips).
    """
    body = "Be warm.\nGround it.\n"  # stripped length < 40 chars, but multi-line
    assert len(body.strip()) < 40
    _write(
        tmp_path / "src" / "wattwise_core" / "agent",
        "tripled.py",
        '"""Mod."""\n\n\n_COACH_PERSONA = """' + body + '"""\n',
    )
    rules = _rules(content_leak.check_paths([tmp_path]))
    assert "content-leak-prompt" in rules


def test_content_leak_clean_engine_source_is_silent(tmp_path: Path) -> None:
    """A module with NO inline prompt/persona body (empty/short defaults only) is clean."""
    _write(
        tmp_path / "src" / "wattwise_core" / "agent",
        "clean.py",
        '"""A focused module."""\n\n\n'
        "class Bundle:\n"
        '    """Holds a loaded prompt (from config, never inline)."""\n\n'
        '    system_prompt: str = ""\n'
        '    plan_system: str = ""\n',
    )
    leaks = [v for v in content_leak.check_paths([tmp_path]) if v.rule == "content-leak-prompt"]
    assert leaks == [], "empty/short config-field defaults are not a leaked prompt body"


def test_content_leak_real_engine_source_has_no_inline_prompts() -> None:
    """The REAL engine source embeds NO inline prompt/persona body (ARCH-R29 over src)."""
    src = Path(__file__).resolve().parents[2] / "src"
    leaks = [v for v in content_leak.check_paths([src]) if v.rule == "content-leak-prompt"]
    assert leaks == [], f"engine source leaks a prompt/persona body: {[v.render() for v in leaks]}"


# ------------------------------------------------------------------------ runner


def test_runner_collects_and_orders_findings(tmp_path: Path) -> None:
    """The aggregating runner returns findings from multiple linters, stably ordered."""
    pkg = tmp_path / "src" / "wattwise_core"
    _write(
        pkg / "analytics",
        "calc.py",
        '"""Mod."""\n\nfrom wattwise_core.api import routers\n\n\n'
        'def compute() -> int:\n    """Doc."""\n    return routers.x\n',
    )
    _write(
        pkg / "domain",
        "queries.py",
        '"""Mod."""\n\n\nRAW = "SELECT 1 FROM activity"\n',
    )
    findings = collect([tmp_path])
    rules = _rules(findings)
    assert "import-direction" in rules
    assert "no-vendor-sql" in rules
    ordered = [(str(v.path), v.line, v.rule) for v in findings]
    assert ordered == sorted(ordered)


def test_runner_clean_tree_has_no_blocking(tmp_path: Path) -> None:
    """A clean engine module yields zero blocking findings from the full pack."""
    _write(
        tmp_path / "src" / "wattwise_core" / "domain",
        "clean.py",
        '"""A focused module."""\n\n\n'
        "def add(a: int, b: int) -> int:\n"
        '    """Return the sum of two integers."""\n'
        "    return a + b\n",
    )
    findings = collect([tmp_path])
    assert [v for v in findings if v.severity is Severity.BLOCKING] == []
