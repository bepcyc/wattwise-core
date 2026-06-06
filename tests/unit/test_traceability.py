"""The requirement->test traceability map is non-empty for the headline IDs (DOD-R6).

Cites: doc 80 DOD-R6 (machine-checkable requirement-ID -> test-ID coverage; a requirement with no
covering test is not done; the report SHOULD fail CI when a shipped requirement is uncovered).

This exercises ``scripts/traceability.py`` as a library: it builds the map over the real test tree
and asserts the close-out spine requirements (traceability itself, GDPR erasure, the planted-secret
gate, and portable persistence) each resolve to at least one covering test node — so the gate is
real, not a stub that always passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "traceability.py"


def _load_traceability() -> ModuleType:
    """Import scripts/traceability.py by path (scripts/ is not an installed package)."""
    spec = importlib.util.spec_from_file_location("ww_traceability", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


traceability = _load_traceability()


@pytest.mark.unit
def test_map_is_non_empty_for_headline_requirements() -> None:
    """Each DOD-R6 headline requirement maps to at least one covering test node."""
    mapping = traceability.build_traceability()
    assert mapping, "traceability map is empty — no requirement IDs found in any test"
    for req in traceability.HEADLINE_REQUIREMENTS:
        nodes = mapping.get(req)
        assert nodes, f"requirement {req} has no covering test (DOD-R6)"
        assert all(node.startswith("tests/") for node in nodes)


@pytest.mark.unit
def test_no_headline_requirement_is_missing() -> None:
    """``missing_headline`` returns empty: every headline requirement is covered (gate is green)."""
    mapping = traceability.build_traceability()
    assert traceability.missing_headline(mapping) == []


@pytest.mark.unit
def test_main_exits_zero_when_headlines_covered(capsys: pytest.CaptureFixture[str]) -> None:
    """Running the script as a CI gate exits 0 and prints a PASS report when coverage holds."""
    rc = traceability.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out
    assert "DOD-R6" in out


@pytest.mark.unit
def test_json_mode_emits_a_parseable_map(capsys: pytest.CaptureFixture[str]) -> None:
    """``--json`` emits a machine-readable map (DOD-R6 machine-checkable mapping)."""
    rc = traceability.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert isinstance(payload, dict)
    # values are lists of pytest-style node ids.
    assert all(isinstance(v, list) for v in payload.values())
    assert payload.get("DOD-R6")


@pytest.mark.unit
def test_acceptance_suffix_is_normalized_onto_its_base_id() -> None:
    """An ``-AC`` citation counts toward its base requirement (SEC-R12-AC -> SEC-R12)."""
    ids = traceability._ids_from_text("covers SEC-R12-AC and PRIV-R8-AC and API-R11c")
    assert "SEC-R12" in ids
    assert "PRIV-R8" in ids
    assert "API-R11c" in ids
    # the raw acceptance form is NOT kept as a distinct key.
    assert "SEC-R12-AC" not in ids
