"""Recorded-eval cassette pins resolved from the layered config (QA-EVAL-R12(a)).

Split out of ``settings.py`` (QUAL-R9 module-size ceiling): this is the eval tier's
narrow, secret-free view of the layered config — the pinned model id plus a digest of
every coach prompt/persona/language content key the recorded fixtures were captured
under. It deliberately reuses the SAME defaults->file->env layering helpers as the
full settings loader so the pins can never drift from what the engine actually loads.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from wattwise_core.config.settings import _DEFAULTS_PATH, ConfigError, _flatten, _read_toml

# The config keys whose values the recorded eval fixtures were captured under
# (QA-EVAL-R12(a)): the pinned model identifier plus every coach prompt/persona/language
# content key. A change to ANY of these without re-recording the fixtures is the
# "stale cassette" condition the eval gate fails on.
_RECORDED_PIN_PREFIXES: tuple[str, ...] = (
    "agent__coach__system_prompt",
    "agent__coach__prompts",
    "agent__coach__languages",
)


def load_recorded_pins() -> dict[str, str]:
    """Resolve the prompt/model pins the recorded eval fixtures depend on (QA-EVAL-R12(a)).

    Returns ``{"model": <pinned model id>, "prompt_sha256": <digest>}`` where the digest
    covers the resolved compose system prompt, every prompt fragment, and every language
    pack — the content whose change invalidates recorded model outputs. Resolved from the
    SAME layered config as the full settings (defaults -> operator file -> env override),
    WITHOUT the secret fail-close (TIER-R1: the offline eval tier carries no secrets).
    A missing model pin fails closed (CFG-R1a).
    """
    config_file_env = os.environ.get("WATTWISE_CONFIG_FILE")
    config_file = Path(config_file_env) if config_file_env else None
    merged: dict[str, Any] = {}
    _flatten("", _read_toml(_DEFAULTS_PATH), merged)
    if config_file is not None:
        if not config_file.is_file():
            raise ConfigError(f"WATTWISE_CONFIG_FILE does not exist: {config_file}")
        _flatten("", _read_toml(config_file), merged)
    model = os.environ.get("WATTWISE_AGENT__MODEL", merged.get("agent__model"))
    if not model:
        raise ConfigError(
            "fail-closed: required config is missing: WATTWISE_AGENT__MODEL / [agent].model "
            "(the recorded-fixture model pin, QA-EVAL-R12(a))"
        )
    content = {key: merged[key] for key in sorted(merged) if key.startswith(_RECORDED_PIN_PREFIXES)}
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {"model": str(model), "prompt_sha256": digest}


__all__ = ["load_recorded_pins"]
