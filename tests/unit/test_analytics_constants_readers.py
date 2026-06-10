"""Unit tests for the fail-closed ``[analytics]`` defaults readers (CFG-R1a).

The constants module sources every analytics tunable from the packaged dead config
file (``defaults.toml``); an absent or mistyped key is a config DEFECT and must fail
closed at import-time read, never fall back to a code literal or truthiness-coerce.
"""

from __future__ import annotations

import pytest

from wattwise_core.analytics.constants import _analytics_default, _analytics_default_bool


@pytest.mark.unit
def test_analytics_default_absent_key_fails_closed() -> None:
    """CFG-R1a: a float key absent from [analytics] raises, never a code-literal fallback."""
    with pytest.raises(RuntimeError, match="fail-closed"):
        _analytics_default("nonexistent_tunable_key")


@pytest.mark.unit
def test_analytics_default_bool_absent_key_fails_closed() -> None:
    """CFG-R1a: a bool key absent from [analytics] raises, never a code-literal fallback."""
    with pytest.raises(RuntimeError, match="fail-closed"):
        _analytics_default_bool("nonexistent_bool_tunable_key")


@pytest.mark.unit
def test_analytics_default_bool_rejects_non_boolean_value() -> None:
    """CFG-R1a: a present but non-boolean value fails closed, never truthiness-coerced."""
    with pytest.raises(RuntimeError, match="absent or non-boolean"):
        # endurance_score_ctl_full_scale exists but is a float, not a bool.
        _analytics_default_bool("endurance_score_ctl_full_scale")


@pytest.mark.unit
def test_analytics_default_bool_reads_packaged_value() -> None:
    """The ES-R2 partial-policy boolean reads back as a real bool from defaults.toml."""
    assert isinstance(_analytics_default_bool("endurance_score_allow_partial"), bool)
