"""Layered, fail-closed configuration for wattwise-core.

Config layering (CFG-R*, doc 10): packaged ``defaults.toml`` -> optional operator
file (``WATTWISE_CONFIG_FILE``) -> environment variables (``WATTWISE_*``). Each
later layer overrides an earlier one.

Secret handling (BOOT-R4, SEC-R*): the service/infra secrets — the database DSN,
the LLM provider key, the token signing key, and the encryption root key — come
ONLY from the environment / a secret manager. They are never baked into images or
committed config. A required secret being absent fails the boot **closed**
(RUN-R4.1): :func:`load_settings` raises rather than starting in an undefined or
insecure state.
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


def _flatten(prefix: str, value: Mapping[str, Any], out: dict[str, Any]) -> None:
    """Flatten a nested TOML table into ``section__key`` settings keys."""
    for key, val in value.items():
        composite = f"{prefix}__{key}" if prefix else key
        if isinstance(val, Mapping):
            _flatten(composite, val, out)
        else:
            out[composite.lower()] = val


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

    # --- app ---
    app__environment: Environment = Environment.PRODUCTION
    app__log_level: str = "INFO"

    # --- api ---
    api__host: str = "0.0.0.0"  # noqa: S104 (bind-all is intended for the containerized service)
    api__port: int = Field(default=8000, ge=1, le=65535)
    api__rate_limit_per_minute: int = Field(default=60, ge=1)
    api__request_max_bytes: int = Field(default=33_554_432, ge=1)

    # --- object store (verbatim original-file retention, RAW-R*) ---
    object_store__kind: str = "local"
    object_store__local_root: Path = Path("/var/lib/wattwise/objects")
    object_store__s3_endpoint: str | None = None
    object_store__s3_bucket: str | None = None

    # --- analytics (doc 40 constants) ---
    analytics__ctl_time_constant_days: float = Field(default=42.0, gt=0)
    analytics__atl_time_constant_days: float = Field(default=7.0, gt=0)

    # --- agent (model-routing seam, grounding) ---
    agent__base_url: str = "https://openrouter.ai/api/v1"
    agent__model: str = "deepseek/deepseek-chat"
    agent__temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    agent__max_output_tokens: int = Field(default=1024, ge=1)
    agent__grounding_min_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    agent__request_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- retention ---
    retention__raw_file_days: int = Field(default=0, ge=0)

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
