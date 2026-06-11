"""Fail-closed guard for removed/renamed configuration keys (CFG-R1a).

Split off :mod:`wattwise_core.config.settings` to keep that module under the QUAL-R9 size
ceiling. When a settings key is renamed, a stale operator override of the OLD key (env var or
operator config file) must not be silently ignored — ``Settings.model_config`` sets
``extra="ignore"``, so without this guard the operator would believe their value is applied
when it is not. The guard detects the legacy key at the env or file layer and the settings
model refuses to boot with an actionable message naming the replacement.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# Settings keys removed/renamed in this release, mapped old -> new. Keys are the flat
# ``section__key`` form, lower-cased, as produced by the file source's flatten and by the env
# source's nested delimiter; detection is case-insensitive (env vars are upper-cased).
#
# This release: the endurance-score "durability" component was renamed to "curve_shape" (the
# name "durability" now belongs to the distinct work-conditioned durability metric, issue #26),
# so its three operator-facing settings keys were renamed with it.
RENAMED_SETTINGS_KEYS: Mapping[str, str] = {
    "analytics__endurance_score_weight_durability": (
        "analytics__endurance_score_weight_curve_shape"
    ),
    "analytics__endurance_score_durability_floor": "analytics__endurance_score_curve_shape_floor",
    "analytics__endurance_score_durability_ceiling": (
        "analytics__endurance_score_curve_shape_ceiling"
    ),
}

_EXAMPLE = (
    "WATTWISE_ANALYTICS__ENDURANCE_SCORE_WEIGHT_CURVE_SHAPE instead of "
    "WATTWISE_ANALYTICS__ENDURANCE_SCORE_WEIGHT_DURABILITY"
)


def _find_in_mapping(data: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Return present legacy keys -> replacement found in a raw settings mapping.

    Handles BOTH the flat ``section__key`` shape (operator file / layered source) and the
    nested-mapping shape the env source produces (``{"analytics": {"endurance_score_...":
    ...}}``) by recursing and re-joining with ``__``. Matching is case-insensitive.
    """
    found: dict[str, str] = {}
    for key, val in data.items():
        composite = f"{prefix}__{key}" if prefix else str(key)
        new = RENAMED_SETTINGS_KEYS.get(composite.lower())
        if new is not None:
            found[composite.lower()] = new
        elif isinstance(val, Mapping):
            found.update(_find_in_mapping(val, composite))
    return found


def _find_in_env() -> dict[str, str]:
    """Return legacy keys -> replacement overridden via ``WATTWISE_`` env vars.

    The env settings source silently DROPS variables that do not match a declared field, so a
    stale ``WATTWISE_*DURABILITY*`` never reaches the model's ``before`` validator payload;
    the environment is scanned directly (case-insensitive) so the rename fails the boot closed.
    """
    found: dict[str, str] = {}
    for old, new in RENAMED_SETTINGS_KEYS.items():
        if f"WATTWISE_{old}".upper() in os.environ:
            found[old] = new
    return found


def detect_renamed_keys(data: Any) -> dict[str, str]:
    """All present legacy keys (env + file layers) mapped to their replacement key."""
    found: dict[str, str] = {}
    if isinstance(data, Mapping):
        found.update(_find_in_mapping(data))
    found.update(_find_in_env())
    return found


def renamed_keys_error_message(found: Mapping[str, str]) -> str:
    """A fail-closed, actionable boot-error message naming each old key and its replacement."""
    lines = "; ".join(f"{old!r} was renamed to {new!r}" for old, new in sorted(found.items()))
    # S608 is a false positive: this is a human-readable boot error message, not SQL — the
    # "instead of" phrasing + parentheses merely trip ruff's string-built-SQL heuristic.
    return (
        "fail-closed: removed/renamed configuration key(s) present "  # noqa: S608
        f"({lines}). These keys were renamed in this release and are no longer recognized; "
        f"update your environment / operator config to the new key name(s) (e.g. set {_EXAMPLE})."
    )


def guard_renamed_keys(cls: type, data: Any) -> Any:
    """Fail the boot closed on removed/renamed config keys (CFG-R1a).

    Bound onto :class:`~wattwise_core.config.settings.Settings` as a
    ``model_validator(mode="before")``. ``mode="before"`` is required: ``extra="ignore"`` would
    otherwise drop a stale legacy key before an after-validator could see it, silently discarding
    the operator's intent instead of refusing to boot. ``ConfigError`` is imported lazily to avoid
    an import cycle with the settings module that defines it.
    """
    found = detect_renamed_keys(data)
    if found:
        # Lazy import breaks the import cycle: settings defines ConfigError and imports this guard.
        from wattwise_core.config.settings import ConfigError  # noqa: PLC0415

        raise ConfigError(renamed_keys_error_message(found))
    return data
