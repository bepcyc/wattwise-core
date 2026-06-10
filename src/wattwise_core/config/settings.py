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

#: SEC-R3 signing-key entropy floor: >= 256 bits (32 bytes) of key material, mirroring the
#: ``encryption_root_key`` 32-byte floor in ``security/crypto.py``. A key shorter than this
#: is refused in a real environment.
_SIGNING_KEY_MIN_BYTES = 32
#: A trivially-weak-key guard: a key built from fewer than this many DISTINCT bytes (e.g.
#: ``"k" * 64`` — long but degenerate) carries no real entropy and is refused (SEC-R3).
_SIGNING_KEY_MIN_DISTINCT_BYTES = 8


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
_LEAF_TABLE_KEYS: frozenset[str] = frozenset(
    {
        "agent__metric_aliases",
        # The externalized coach-config behavior-asset tables (§16 / SKILL-R1/-R3): each is a
        # free-form name->content map (prompt fragments, grounding rules) or a single closed
        # record (the bundle manifest) the engine consumes WHOLE, not a nested set of scalar
        # settings. Stopping the flattener here keeps each as ONE dict-valued setting (CFG-R1a),
        # so a fragment/rule is addressed by name inside the loaded map, not by a per-entry
        # settings key. (``agent__coach__skills`` is a TOML array-of-tables → already a list.)
        "agent__coach__prompts",
        "agent__coach__grounding_rules",
        "agent__coach__manifest",
    }
)


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

    # --- security: CORS / allowed-host / transport headers (SEC-R10/.1/.2, CFG-R1a) ---
    # Config-driven, never hardcoded origins/hosts (SEC-R10.2 "no per-deployment values
    # baked into code"). The OSS defaults are first-party-client-correct out of the box;
    # an operator overrides them via the operator file / env. A TOML array flattens to one
    # list setting.
    security__cors_allow_origins: list[str]
    security__cors_allow_credentials: bool
    security__cors_allow_methods: list[str]
    security__cors_allow_headers: list[str]
    security__allowed_hosts: list[str]
    # Security-header values (SEC-R10.1): HSTS max-age, the Referrer-Policy, and the CSP
    # for any HTML surface. Loaded content (CFG-R1a), never a code literal.
    security__hsts_max_age_seconds: int = Field(ge=0)
    security__referrer_policy: str
    security__content_security_policy: str

    # --- entitlement: the OSS default plan's non-monetary local guards (AGT-ENT-R4) ---
    # The single all-permissive OSS plan carries NO monetary budget (COMM-R20); these are
    # the per-request NON-monetary bounds it carries with generous, configurable defaults
    # (loaded content, CFG-R1a — never hardcoded in the gate/graph, AGT-ENT-R1). An
    # operator MAY raise them. They are strictly positive (a 0/negative bound is a
    # degenerate plan rejected fail-closed at load by ``entitlement.validate_plan``).
    entitlement__node_visit_ceiling: int = Field(ge=1)
    entitlement__max_output_tokens: int = Field(ge=1)
    entitlement__wall_clock_seconds: float = Field(gt=0)
    entitlement__max_tool_iterations: int = Field(ge=1)
    entitlement__request_rate_per_minute: int = Field(ge=1)

    # --- rate-limit: the READ / MUTATING per-minute request ceilings (LIMIT-R2, CFG-R1a) ---
    # The per-athlete per-minute request ceilings for the read + mutating endpoint classes
    # (LIMIT-R2). Loaded content (CFG-R1a) — never a code literal; the production RateLimiter is
    # built from these (+ the AGENT class from ``entitlement__request_rate_per_minute``, the
    # entitlement-governed bound) so NO rate value is baked into code. Strictly positive.
    ratelimit__read_per_minute: int = Field(ge=1)
    ratelimit__mutating_per_minute: int = Field(ge=1)

    # --- object store (verbatim original-file retention, RAW-R*) ---
    object_store__kind: str
    object_store__local_root: Path
    # s3_* are genuinely optional (only used when kind='s3'); TOML cannot express
    # null, so absence == not-configured is modelled as the ``None`` sentinel rather
    # than a concrete value default (CFG-R1a is about VALUES, not absence).
    object_store__s3_endpoint: str | None = None
    object_store__s3_bucket: str | None = None

    # --- ingestion (bulk-upsert batching, PERF-R1 / ING-UPS-R1/R3) ---
    # Candidates are landed in batches of this size: each batch is one bounded multi-row
    # round-trip (PERF-R1) committed atomically, so a committed batch survives even if a
    # later batch fails (ING-UPS-R3). Strictly positive; the value lives here (CFG-R1a),
    # never a code literal.
    ingestion__batch_size: int = Field(ge=1)

    # --- adapters: Intervals.icu outbound-client resilience (CLI-R6/R10/R11, CFG-R1a) ---
    # The typed client's per-source retry budget (CLI-R6) + client-side token-bucket limiter
    # (CLI-R10/R11) are BUILT from these settings (IntervalsIcuClient.from_settings) so NO
    # resilience value is code-baked (CFG-R1a). Schema + constraints only — the VALUES live in
    # defaults.toml, overridable by the operator file / env. The budget caps attempts AND total
    # elapsed wall time; the bucket reduce_factor is a fraction in (0,1) and min_rate floors the
    # adaptive 429 issue-rate reduction (CLI-R11). All rates/counts/durations strictly positive.
    adapters__intervals_icu__budget_max_attempts: int = Field(ge=1)
    adapters__intervals_icu__budget_max_elapsed_s: float = Field(gt=0)
    adapters__intervals_icu__budget_base_backoff_s: float = Field(ge=0)
    adapters__intervals_icu__budget_max_backoff_s: float = Field(ge=0)
    adapters__intervals_icu__bucket_rate_per_s: float = Field(gt=0)
    adapters__intervals_icu__bucket_capacity: float = Field(gt=0)
    adapters__intervals_icu__bucket_reduce_factor: float = Field(gt=0, lt=1)
    adapters__intervals_icu__bucket_min_rate: float = Field(gt=0)
    adapters__intervals_icu__http_timeout_s: float = Field(gt=0)

    # --- analytics (doc 40 constants) ---
    analytics__ctl_time_constant_days: float = Field(gt=0)
    analytics__atl_time_constant_days: float = Field(gt=0)
    # DEGR-R2 substitution confidence multiplier (in (0,1]); the VALUE lives in defaults.toml
    # (CFG-R1a), this declares only the typed schema + range constraint.
    analytics__training_load_confidence_penalty: float = Field(gt=0, le=1)

    # --- agent (model-routing seam, grounding) ---
    agent__base_url: str
    # 2026 default model + budget live in defaults.toml (MODEL-R5a); the budget is
    # sized for reasoning models (reasoning tokens are billed against the completion
    # budget and emitted before the answer), so a small allowance starves the answer.
    agent__model: str
    # The observability tier + reasoning effort the run is tagged with on every model span
    # (AGT-OBS-R2). The OSS engine runs ONE model (MODEL-R4: no escalation), so these are the
    # single configured tier/effort labels — loaded content (CFG-R1a), never a code literal; a
    # commercial deployment with multiple tiers overrides per route through the same fields.
    agent__tier: str
    agent__reasoning_effort: str
    # Per-token pricing (USD per million tokens) the AGT-OBS-R2 per-span/per-run cost is COMPUTED
    # from. Loaded config content (CFG-R1a) — the engine bakes no price into code; a deployment
    # sets its provider's real rates. Non-negative; 0 reports cost as 0 (a free local model).
    agent__cost__input_per_million_usd: float = Field(ge=0.0)
    agent__cost__output_per_million_usd: float = Field(ge=0.0)
    agent__temperature: float = Field(ge=0.0, le=2.0)
    agent__max_output_tokens: int = Field(ge=1)
    agent__grounding_min_coverage: float = Field(ge=0.0, le=1.0)
    agent__request_timeout_seconds: float = Field(gt=0)
    # AGT-SEC-R4 provider-send PII policy ("where policy requires"): when true the model seam
    # masks the outbound system + untrusted-data regions through the central redactor before
    # they reach the third-party provider. Loaded policy (CFG-R1a) — no code-baked default;
    # absence from every layer fails closed at load.
    agent__redact_provider_payloads: bool
    # CKPT-R4 idempotency dedup window (seconds): a re-submitted SAME turn within this window
    # resolves to the SAME thread/run instead of starting a duplicate. Loaded config (CFG-R1a),
    # never a code literal; ``0`` disables time-bucketing (every turn is its own bucket).
    agent__idempotency_dedup_window_seconds: int = Field(ge=0)
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
    # The externalized skill/prompt bundle (SKILL-R1..R4 / CFG-R3): EVERY system/agent prompt is
    # loaded CONTENT here, never inline engine code (ARCH-R29). ``prompts`` is the name->fragment
    # map (the verbatim system prompts the verdict/compose nodes drive); ``grounding_rules`` the
    # name->policy-text map a skill's ``grounding_refs`` resolve against; ``manifest`` the closed
    # bundle identity/schema-version record (SKILL-R3b); ``skills`` the array of named/versioned
    # composable skill records (SKILL-R2/-R3a). These are SHAPE-only here (typed maps/list); the
    # CLOSED skill/manifest schema + cross-reference resolution + fail-closed validation are owned
    # by ``CoachManifest.load`` (SKILL-R4 / CFG-R6), not by this settings schema.
    agent__coach__prompts: dict[str, str]
    agent__coach__grounding_rules: dict[str, str]
    agent__coach__manifest: dict[str, str]
    agent__coach__skills: list[dict[str, Any]]
    # Offline-eval cost/latency budgets (QA-EVAL-R8): the median cost-per-task and p95
    # latency the eval gate enforces, plus the price per 1k tokens used to cost a recorded
    # (network-free) run. Loaded content (CFG-R1a), never a gate hardcode; strictly positive.
    agent__eval__median_cost_usd: float = Field(gt=0)
    agent__eval__p95_latency_ms: float = Field(gt=0)
    agent__eval__cost_per_1k_tokens_usd: float = Field(gt=0)
    agent__metric_aliases: dict[str, str]

    # --- retention ---
    retention__raw_file_days: int = Field(ge=0)
    # CKPT-R8 / PRIV-R7 agent-state retention window (days): durable run checkpoints (threads,
    # writes, interrupts) older than this are expired by the retention sweeper. Loaded config
    # (CFG-R1a), never a code literal; ``0`` = retain indefinitely (no sweep), mirroring
    # ``retention__raw_file_days``.
    retention__agent_state_days: int = Field(ge=0)

    # --- SECRETS (env / secret-manager only; BOOT-R4) ---
    database_dsn: SecretStr | None = None
    encryption_root_key: SecretStr | None = None
    token_signing_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _fail_closed(self) -> Settings:
        """Refuse to boot when required secrets are absent/weak or config is insecure.

        In development the engine may run without external secrets (it uses an
        ephemeral key path), but in staging/production every load-bearing secret
        MUST be present or the boot fails closed (RUN-R4.1). The CORS configuration-cliff
        guard (SEC-R10-AC) is enforced in EVERY environment — a wildcard origin combined
        with credentials is always rejected.
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
        if strict:
            self._require_strong_signing_key()
        if self.object_store__kind not in {"local", "s3"}:
            raise ConfigError(
                f"object_store.kind must be 'local' or 's3', got {self.object_store__kind!r}"
            )
        if self.object_store__kind == "s3" and not (
            self.object_store__s3_endpoint and self.object_store__s3_bucket
        ):
            raise ConfigError("object_store.kind='s3' requires s3_endpoint and s3_bucket")
        self._reject_wildcard_cors_with_credentials()
        return self

    def _require_strong_signing_key(self) -> None:
        """Reject a weak/short token signing key in a real environment (SEC-R3 / RUN-R4.1).

        Mirrors the ``encryption_root_key`` 32-byte floor in ``security/crypto.py``: the
        signing/verification key material MUST carry at least 256 bits (32 bytes) of
        entropy and MUST NOT be trivially weak. A key shorter than 32 bytes, or one whose
        material is degenerate (a single repeated byte / a tiny distinct-character set —
        no real entropy), is refused so the service never signs tokens under a guessable
        key. In development the key may be absent (an ephemeral path) or a short test
        value, so this floor is enforced only in the strict (staging/production)
        environments — exactly where the presence check above already requires it.
        """
        key = self.token_signing_key
        if key is None:  # presence already enforced above for strict; defensive guard
            return
        raw = key.get_secret_value().encode("utf-8")
        if len(raw) < _SIGNING_KEY_MIN_BYTES:
            raise ConfigError(
                "fail-closed: WATTWISE_TOKEN_SIGNING_KEY carries insufficient entropy "
                f"(needs >= {_SIGNING_KEY_MIN_BYTES} bytes / 256 bits, got {len(raw)}); "
                "the service refuses to start with a weak signing key (SEC-R3)"
            )
        if len(set(raw)) < _SIGNING_KEY_MIN_DISTINCT_BYTES:
            raise ConfigError(
                "fail-closed: WATTWISE_TOKEN_SIGNING_KEY is trivially weak "
                "(too few distinct bytes — a repeated/degenerate value, not real entropy); "
                "the service refuses to start (SEC-R3)"
            )

    def _reject_wildcard_cors_with_credentials(self) -> None:
        """Reject the wildcard-origin + credentials configuration cliff (SEC-R10 / SEC-R10-AC).

        A configuration that allows the wildcard origin ``*`` AND sets
        ``allow_credentials=true`` MUST be rejected at startup with a clear configuration
        error (SEC-R10-AC) — it is an always-insecure combination the browser would refuse
        anyway, so the service fails closed rather than starting in an undefined CORS state.
        Enforced in every environment (a configuration-cliff guard, not an env-strict rule).
        """
        if self.security__cors_allow_credentials and "*" in self.security__cors_allow_origins:
            raise ConfigError(
                "fail-closed: CORS must not combine a wildcard origin '*' with "
                "allow_credentials=true (SEC-R10); the service refuses to start"
            )

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


# The offline-eval budget keys (QA-EVAL-R8), resolved from the SAME layered config
# (defaults.toml -> operator file -> env) as the full settings but WITHOUT the secret
# fail-close — so the network-free, secret-free offline eval tier (TIER-R1) can read its
# cost/latency budgets. Each key is required: absence from every layer fails closed (CFG-R1a).
_EVAL_BUDGET_KEYS: tuple[str, ...] = (
    "agent__eval__median_cost_usd",
    "agent__eval__p95_latency_ms",
    "agent__eval__cost_per_1k_tokens_usd",
)


def load_eval_budget() -> dict[str, float]:
    """Resolve the QA-EVAL-R8 cost/latency budget values from layered config (CFG-R1a).

    Reads the ``[agent.eval]`` budgets from the layered packaged-defaults + operator file,
    overlaid by any ``WATTWISE_AGENT__EVAL__*`` env override, WITHOUT instantiating the
    secret-validated :class:`Settings` (the offline eval tier carries no operator secrets,
    TIER-R1). A budget key absent from every layer fails closed (CFG-R1a). Values must be
    strictly positive.
    """
    config_file_env = os.environ.get("WATTWISE_CONFIG_FILE")
    config_file = Path(config_file_env) if config_file_env else None
    merged: dict[str, Any] = {}
    _flatten("", _read_toml(_DEFAULTS_PATH), merged)
    if config_file is not None:
        if not config_file.is_file():
            raise ConfigError(f"WATTWISE_CONFIG_FILE does not exist: {config_file}")
        _flatten("", _read_toml(config_file), merged)
    out: dict[str, float] = {}
    for key in _EVAL_BUDGET_KEYS:
        env_val = os.environ.get(f"WATTWISE_{key.upper()}")
        raw = env_val if env_val is not None else merged.get(key)
        if raw is None:
            raise ConfigError(
                f"fail-closed: required eval-budget config is missing: WATTWISE_{key.upper()} "
                "(defaults.toml [agent.eval] / operator file / env; QA-EVAL-R8, CFG-R1a)"
            )
        value = float(raw)
        if value <= 0:
            raise ConfigError(f"fail-closed: eval-budget {key} must be > 0 (got {value})")
        out[key] = value
    return out
