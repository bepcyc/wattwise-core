"""Layered, fail-closed configuration for wattwise-core.

Config layering (CFG-R*, doc 10): packaged ``defaults.toml`` -> optional operator
file (``WATTWISE_CONFIG_FILE``) -> environment variables (``WATTWISE_*``). Each
later layer overrides an earlier one.

Schema-only code (CFG-R1a): this module declares ONLY the typed schema and its
validation constraints (``ge``/``le``/``gt``/enum). It carries **no** concrete
configuration value — not even as a field default. Every non-secret value lives in
the packaged ``defaults.toml`` (the lowest config layer), overridable by the
operator file / environment. A non-secret field is therefore *required*: a
``Field`` with constraints but no ``default=`` (pydantic treats it as required),
satisfied by ``defaults.toml`` for a clean dev boot. If a key is absent from every
layer, validation fails **closed** rather than reintroducing a hardcoded fallback.

Secret handling (CFG-R2, BOOT-R4, SEC-R*): the service/infra secrets — the database
DSN, the LLM provider key, the token signing key, and the encryption root key — are
the sole exception to CFG-R1a: they come ONLY from the environment / a secret
manager, never from ``defaults.toml`` or any committed config, and are never baked
into images. A required secret being absent fails the boot **closed** (RUN-R4.1):
:func:`load_settings` raises rather than starting in an undefined or insecure state.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

_DEFAULTS_PATH = Path(__file__).with_name("defaults.toml")


class Environment(StrEnum):
    """Deployment environment; governs which fail-closed rules are strict."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"


class ConfigError(RuntimeError):
    """A fail-closed configuration error (RUN-R4.1).

    Raised when configuration is missing, contradictory, or insecure for the
    target environment. The engine must refuse to boot rather than continue in
    an undefined state.
    """


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


# TOML tables whose VALUE is itself dict-shaped config (a free-form key->value map the
# engine consumes whole, e.g. the metric-equivalence aliases of §16) rather than a nested
# set of scalar settings. The flattener stops at these and assigns the whole sub-mapping to
# the single ``section__key`` field, so a free-form alias table is one ``dict[str, str]``
# setting (CFG-R1a) instead of one settings key per alias.
_LEAF_TABLE_KEYS: frozenset[str] = frozenset({"agent__metric_aliases"})


def _flatten(prefix: str, value: Mapping[str, Any], out: dict[str, Any]) -> None:
    """Flatten a nested TOML table into ``section__key`` settings keys.

    A table whose composite key is a declared leaf table (:data:`_LEAF_TABLE_KEYS`) is NOT
    recursed into: its whole sub-mapping becomes one dict-valued setting, so a free-form
    map (metric aliases, §16) is a single typed ``dict`` field, not one key per entry.
    """
    for key, val in value.items():
        composite = f"{prefix}__{key}" if prefix else key
        if isinstance(val, Mapping) and composite.lower() not in _LEAF_TABLE_KEYS:
            _flatten(composite, val, out)
        else:
            out[composite.lower()] = dict(val) if isinstance(val, Mapping) else val


class _LayeredFileSource(PydanticBaseSettingsSource):
    """Settings source: packaged defaults overlaid by an optional operator file."""

    def __init__(self, settings_cls: type[BaseSettings], config_file: Path | None) -> None:
        super().__init__(settings_cls)
        merged: dict[str, Any] = {}
        _flatten("", _read_toml(_DEFAULTS_PATH), merged)
        if config_file is not None:
            if not config_file.is_file():
                raise ConfigError(f"WATTWISE_CONFIG_FILE does not exist: {config_file}")
            _flatten("", _read_toml(config_file), merged)
        self._data = merged

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Unused: the whole mapping is returned from __call__ instead.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class Settings(BaseSettings):
    """Resolved, validated engine configuration.

    Fields are populated from the layered sources in :meth:`settings_customise_sources`.
    Nested config keys use the ``section__key`` form (e.g. ``api__port``); the same
    form is the env-var suffix (``WATTWISE_API__PORT``).
    """

    model_config = SettingsConfigDict(
        env_prefix="WATTWISE_",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
    )

    # Non-secret fields below carry NO value default (CFG-R1a): only the type +
    # validation constraints. Each is required and is resolved from defaults.toml
    # (overridable by the operator file / env). Absence from every layer fails closed.

    # --- app ---
    app__environment: Environment
    app__log_level: str

    # --- api ---
    api__host: str  # value (incl. bind-all 0.0.0.0) lives in defaults.toml, not here
    api__port: int = Field(ge=1, le=65535)
    api__rate_limit_per_minute: int = Field(ge=1)
    api__request_max_bytes: int = Field(ge=1)

    # --- object store (verbatim original-file retention, RAW-R*) ---
    object_store__kind: str
    object_store__local_root: Path
    # s3_* are genuinely optional (only used when kind='s3'); TOML cannot express
    # null, so absence == not-configured is modelled as the ``None`` sentinel rather
    # than a concrete value default (CFG-R1a is about VALUES, not absence).
    object_store__s3_endpoint: str | None = None
    object_store__s3_bucket: str | None = None

    # --- analytics (doc 40 constants) ---
    analytics__ctl_time_constant_days: float = Field(gt=0)
    analytics__atl_time_constant_days: float = Field(gt=0)

    # --- agent (model-routing seam, grounding) ---
    agent__base_url: str
    # 2026 default model + budget live in defaults.toml (MODEL-R5a); the budget is
    # sized for reasoning models (reasoning tokens are billed against the completion
    # budget and emitted before the answer), so a small allowance starves the answer.
    agent__model: str
    agent__temperature: float = Field(ge=0.0, le=2.0)
    agent__max_output_tokens: int = Field(ge=1)
    agent__grounding_min_coverage: float = Field(ge=0.0, le=1.0)
    agent__request_timeout_seconds: float = Field(gt=0)
    # First-party URL allow-list (GROUND-R4): the exact hosts whose links the grounder may keep.
    # Loaded policy content (CFG-R1a), never a host literal baked into code.
    agent__allowed_hosts: list[str]
    # The OSS default coach-config bundle (SKILL-R1 / §16): the compose-node system prompt,
    # the numeric-grounding thresholds, the canonical-value display precision, the dateless-claim
    # lookback window, and the metric-equivalence (natural term -> canonical key) aliases the
    # grounder resolves through (GROUND-R2/-R7). All are loaded CONTENT, never inline engine
    # literals (CFG-R1a) — values live in defaults.toml, overridable by the operator/private bundle.
    agent__coach__system_prompt: str
    agent__coach__grounding_rel_tolerance: float = Field(ge=0.0)
    agent__coach__grounding_abs_tolerance: float = Field(ge=0.0)
    agent__coach__grounding_display_decimals: int = Field(ge=0, le=6)
    agent__coach__latest_lookback_days: int = Field(ge=1)
    agent__metric_aliases: dict[str, str]

    # --- retention ---
    retention__raw_file_days: int = Field(ge=0)

    # --- SECRETS (env / secret-manager only; BOOT-R4) ---
    database_dsn: SecretStr | None = None
    encryption_root_key: SecretStr | None = None
    token_signing_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _fail_closed(self) -> Settings:
        """Refuse to boot when required secrets are absent in a real environment.

        In development the engine may run without external secrets (it uses an
        ephemeral key path), but in staging/production every load-bearing secret
        MUST be present or the boot fails closed (RUN-R4.1).
        """
        strict = self.app__environment is not Environment.DEVELOPMENT
        missing: list[str] = []
        if self.database_dsn is None:
            missing.append("WATTWISE_DATABASE_DSN")
        if strict:
            if self.encryption_root_key is None:
                missing.append("WATTWISE_ENCRYPTION_ROOT_KEY")
            if self.token_signing_key is None:
                missing.append("WATTWISE_TOKEN_SIGNING_KEY")
        if missing:
            raise ConfigError(
                "fail-closed: required configuration is missing: "
                + ", ".join(sorted(missing))
                + " (must be provided via the environment / a secret manager; BOOT-R4)"
            )
        if self.object_store__kind not in {"local", "s3"}:
            raise ConfigError(
                f"object_store.kind must be 'local' or 's3', got {self.object_store__kind!r}"
            )
        if self.object_store__kind == "s3" and not (
            self.object_store__s3_endpoint and self.object_store__s3_bucket
        ):
            raise ConfigError("object_store.kind='s3' requires s3_endpoint and s3_bucket")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence (highest first): explicit init kwargs, env vars, then the
        # layered packaged-defaults + operator-file source. NO dotenv source is
        # registered: the engine does not read .env files at runtime (TASK §5);
        # secrets arrive via the environment / secret manager only.
        config_file_env = os.environ.get("WATTWISE_CONFIG_FILE")
        config_file = Path(config_file_env) if config_file_env else None
        return (init_settings, env_settings, _LayeredFileSource(settings_cls, config_file))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide resolved settings (cached).

    Raises :class:`ConfigError` (fail-closed) if configuration is invalid.
    """
    return load_settings()


def load_settings(**overrides: Any) -> Settings:
    """Load and validate settings fresh, applying any explicit overrides."""
    return Settings(**overrides)
