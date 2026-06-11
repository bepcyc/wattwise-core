"""Unit tests for the removed/renamed-config-key fail-closed guard (CFG-R1a).

The endurance-score "durability" component was renamed to "curve_shape" in this release
(issue #26). The three operator-facing settings keys were renamed with it. Because
``Settings.model_config`` sets ``extra="ignore"``, a stale operator override of an old key
would otherwise be SILENTLY dropped; the guard refuses to boot with an actionable message
naming the replacement so the rename never silently changes behavior (the CHANGELOG documents
this as a boot-time fail-closed change).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wattwise_core.config import load_settings
from wattwise_core.config.settings import ConfigError

pytestmark = pytest.mark.unit

_DSN = "sqlite+aiosqlite:///:memory:"

_LEGACY_ENV_KEYS = (
    "WATTWISE_ANALYTICS__ENDURANCE_SCORE_WEIGHT_DURABILITY",
    "WATTWISE_ANALYTICS__ENDURANCE_SCORE_DURABILITY_FLOOR",
    "WATTWISE_ANALYTICS__ENDURANCE_SCORE_DURABILITY_CEILING",
)


def test_clean_config_boots_without_legacy_keys() -> None:
    """The current key names load normally (no false positive from the guard)."""
    settings = load_settings(app__environment="development", database_dsn=_DSN)
    # The renamed field exists and carries the configured value.
    assert settings.analytics__endurance_score_weight_curve_shape >= 0.0


@pytest.mark.parametrize("legacy_env", _LEGACY_ENV_KEYS)
def test_legacy_env_override_fails_closed_with_actionable_message(
    legacy_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale ``WATTWISE_*DURABILITY*`` env override refuses the boot, naming the new key."""
    monkeypatch.setenv(legacy_env, "0.3")
    with pytest.raises(ConfigError) as exc:
        load_settings(app__environment="development", database_dsn=_DSN)
    message = str(exc.value)
    # The error must name the new key so the operator knows the fix (not a generic schema error).
    assert "curve_shape" in message
    assert "renamed" in message.lower()


def test_legacy_operator_file_key_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy key in the operator config FILE also fails closed (file layer, not just env)."""
    op_file = tmp_path / "operator.toml"
    op_file.write_text("[analytics]\nendurance_score_durability_floor = 0.4\n")
    monkeypatch.setenv("WATTWISE_CONFIG_FILE", str(op_file))
    with pytest.raises(ConfigError) as exc:
        load_settings(app__environment="development", database_dsn=_DSN)
    assert "curve_shape" in str(exc.value)
