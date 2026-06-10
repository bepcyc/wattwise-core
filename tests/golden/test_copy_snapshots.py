"""Per-locale golden snapshots of rendered user-facing copy (QUAL-R13(j)).

Pins every catalog entry's rendered text, per locale, against the committed snapshot
fixture — so ANY copy change is a visible, reviewed diff (never an accidental drift),
and pins a representative rendered API problem ``errors[]`` body so the catalog-driven
wiring (not just the catalog file) is covered. Also asserts the i18n completeness rule:
every key exists in every supported locale (QUAL-R13(g)).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wattwise_core.api.copy import SUPPORTED_LOCALES, message
from wattwise_core.api.problems import parameter_invalid, range_reversed

pytestmark = pytest.mark.golden

_SNAPSHOT = Path(__file__).parent / "fixtures" / "copy_snapshots.json"


def _snapshot() -> dict[str, dict[str, str]]:
    loaded: dict[str, dict[str, str]] = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    return loaded


def test_every_locale_renders_exactly_the_committed_snapshot() -> None:
    """Each catalog key renders byte-identically to the committed per-locale snapshot:
    a copy change is an explicit reviewed fixture edit (QUAL-R13(j) / GOLD-R2)."""
    snapshot = _snapshot()
    assert set(snapshot) == set(SUPPORTED_LOCALES)
    for locale, entries in snapshot.items():
        for key, expected in entries.items():
            assert message(key, locale) == expected, f"{locale}/{key} drifted"


def test_every_key_exists_in_every_locale() -> None:
    """No locale is missing a key — assembled from whole entries, never mixed (R13(g))."""
    snapshot = _snapshot()
    reference = set(snapshot["en"])
    for locale in SUPPORTED_LOCALES:
        assert set(snapshot[locale]) == reference


def test_rendered_problem_body_matches_catalog_copy() -> None:
    """The problem builders RENDER the catalog copy (the wiring, not only the file):
    the errors[] message equals the keyed entry and carries the stable machine code."""
    body = range_reversed().errors[0].to_dict()
    assert body["message"] == message("validation.range_reversed")
    assert body["code"] == "out_of_range"
    fallback = parameter_invalid("from").errors[0].to_dict()
    assert fallback["message"] == message("validation.check_value")
    assert fallback["code"] == "invalid"
