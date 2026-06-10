"""Per-write-domain DB-role DSN resolution + the structural grant matrix (DEPLOY-R4 / ARCH-R3).

DEPLOY-R4 requires database access roles be DISTINCT per write domain — FOUR of them. The
engine exposes that as four resolved DSNs — canonical-write (Ingestion, ARCH-R3a),
master-data-write (the API's master-data endpoints, ARCH-R3b), read-only (Domain/Analytics),
and agent-state-write (Orchestrator) — each backed by its own optional secret. These tests pin:

* when a deployment configures the distinct master-data / read-only / agent-state DSNs, each
  helper returns ITS role's DSN (the per-role split is honoured, not collapsed to canonical);
* when they are unset, all fall back to the canonical DSN (a single-operator self-host runs
  on one credential — the role split is an opt-in deploy choice, not a hard requirement);
* with NO canonical DSN reachable, every helper resolves to ``None`` so the engine builders
  fail closed (BOOT-R4) rather than fabricating a credential;
* the API factory binds the master-data endpoints' sessions to the master-data-write
  Database — its OWN engine when the role DSN is distinct (ARCH-R3b wiring);
* the pure :func:`grant_plan` realizes the FULL 4-role reciprocal-denial matrix (ARCH-R24):
  on SQLite the role enforcement is structural (separate DSN/role config + this plan-level
  assertion); the real privilege denial is proven on PostgreSQL by the integration leg.
"""

from __future__ import annotations

import pytest

from wattwise_core.api.app import create_app
from wattwise_core.config import Settings, load_settings
from wattwise_core.persistence.roles import DbRole, grant_plan

_CANON = "postgresql+asyncpg://canon_writer:pw@db:5432/wattwise"
_MASTER = "postgresql+asyncpg://master_writer:pw@db:5432/wattwise"
_READ = "postgresql+asyncpg://read_only:pw@db:5432/wattwise"
_STATE = "postgresql+asyncpg://state_writer:pw@db:5432/agentstate"


def _settings(**overrides: object):  # type: ignore[no-untyped-def]
    return load_settings(app__environment="development", **overrides)


def test_distinct_role_dsns_are_honoured() -> None:
    """Each helper returns its OWN per-role DSN when configured (DEPLOY-R4, all four)."""
    settings = _settings(
        database_dsn=_CANON,
        database_master_data_dsn=_MASTER,
        database_read_dsn=_READ,
        agent_state_dsn=_STATE,
    )
    assert settings.canonical_write_dsn() == _CANON
    assert settings.master_data_write_dsn() == _MASTER
    assert settings.read_only_dsn() == _READ
    assert settings.agent_state_write_dsn() == _STATE
    # The four resolved DSNs are genuinely distinct (the per-role split, not a collapse).
    assert len({_CANON, _MASTER, _READ, _STATE}) == 4


def test_all_roles_fall_back_to_canonical_when_unset() -> None:
    """Unset per-role DSNs fall back to the canonical DSN (single-operator self-host)."""
    settings = _settings(database_dsn=_CANON)
    assert settings.canonical_write_dsn() == _CANON
    assert settings.master_data_write_dsn() == _CANON
    assert settings.read_only_dsn() == _CANON
    assert settings.agent_state_write_dsn() == _CANON


def test_each_role_dsn_is_independent_of_the_others() -> None:
    """One configured role DSN is honoured without coupling to the other roles' DSNs."""
    settings = _settings(database_dsn=_CANON, database_read_dsn=_READ)
    assert settings.read_only_dsn() == _READ  # distinct read role used
    assert settings.master_data_write_dsn() == _CANON  # master-data still falls back
    assert settings.agent_state_write_dsn() == _CANON  # agent-state still falls back

    settings = _settings(database_dsn=_CANON, database_master_data_dsn=_MASTER)
    assert settings.master_data_write_dsn() == _MASTER  # distinct master-data role used
    assert settings.read_only_dsn() == _CANON
    assert settings.agent_state_write_dsn() == _CANON


def test_blank_per_role_dsn_coerces_to_canonical_fallback() -> None:
    """A BLANK per-role DSN (the compose ``${VAR:-}`` empty-string form) falls back, not breaks.

    The compose env map cannot omit a key, so an unset deploy var arrives as ``""``. Settings
    MUST treat that as "not configured" and fall back to the canonical DSN — never build an
    empty, invalid per-role DSN (DEPLOY-R4 fail-closed).
    """
    settings = _settings(
        database_dsn=_CANON,
        database_master_data_dsn="",
        database_read_dsn="",
        agent_state_dsn="",
    )
    assert settings.database_master_data_dsn is None
    assert settings.database_read_dsn is None
    assert settings.agent_state_dsn is None
    assert settings.master_data_write_dsn() == _CANON
    assert settings.read_only_dsn() == _CANON
    assert settings.agent_state_write_dsn() == _CANON


def test_all_dsns_none_when_unconfigured_fails_closed() -> None:
    """No canonical DSN reachable -> every helper resolves to ``None`` (fail-closed, BOOT-R4).

    ``load_settings`` requires ``database_dsn`` in every environment, so the defensive
    ``None`` path is reached by clearing the secret on an already-validated Settings instance
    (``model_construct`` bypasses re-validation) — proving the helpers never fabricate a DSN
    when none is configured.
    """
    settings = Settings.model_construct(
        database_dsn=None,
        database_master_data_dsn=None,
        database_read_dsn=None,
        agent_state_dsn=None,
    )
    assert settings.canonical_write_dsn() is None
    assert settings.master_data_write_dsn() is None
    assert settings.read_only_dsn() is None
    assert settings.agent_state_write_dsn() is None


def test_app_binds_master_data_database_distinct_when_configured(tmp_path: object) -> None:
    """The factory gives the master-data surface its OWN Database under a per-role deploy.

    ARCH-R3b wiring: with a distinct ``database_master_data_dsn`` configured, the app binds
    ``app.state.master_data_database`` to a SEPARATE engine on that role's DSN (the credential
    the goals / user-settings / athlete routers' sessions open on); without it, the surface
    shares the canonical Database (one credential, one pool — no phantom second store).
    """
    canon = f"sqlite+aiosqlite:///{tmp_path}/canon.db"
    master = f"sqlite+aiosqlite:///{tmp_path}/master.db"
    split = create_app(_settings(database_dsn=canon, database_master_data_dsn=master))
    assert split.state.master_data_database is not split.state.database
    assert "master.db" in str(split.state.master_data_database.engine.url)

    shared = create_app(_settings(database_dsn=canon))
    assert shared.state.master_data_database is shared.state.database


# --- the FULL 4-role reciprocal-denial matrix, asserted structurally on the grant plan ---

_ROLE_NAMES = {role: f"ww_{role.value}" for role in DbRole}
_PASSWORDS = dict.fromkeys(DbRole, "pw")
_CANON_T = "s.activity"
_MASTER_T = "s.goal"
_STATE_T = "s.checkpoint"


def _plan() -> list[str]:
    return grant_plan(
        schema="s",
        canonical_tables=[_CANON_T],
        master_data_tables=[_MASTER_T],
        agent_state_tables=[_STATE_T],
        role_names=_ROLE_NAMES,
        passwords=_PASSWORDS,
    )


def _write_grant(table: str, role: DbRole) -> str:
    return f'GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO "{_ROLE_NAMES[role]}"'


def _read_grant(table: str, role: DbRole) -> str:
    return f'GRANT SELECT ON {table} TO "{_ROLE_NAMES[role]}"'


def test_grant_plan_full_reciprocal_denial_matrix() -> None:
    """ARCH-R24/DEPLOY-R4: each role can write ONLY its own domain — every other cell is denied.

    Structural assertion of the 4x3 write matrix (role x table-partition) on the pure grant
    plan: exactly ONE write grant per partition, to its owning role; reads outside the write
    domain are SELECT-only on the canonical partitions; the agent-state store grants NOTHING
    to any canonical-store role; and the plan opens fail-closed (PUBLIC revoked first).
    """
    plan = _plan()
    assert plan[0] == 'REVOKE ALL ON SCHEMA "s" FROM PUBLIC'  # fail-closed default

    # Positive diagonal: each write-owning role gets full DML on ITS partition only.
    assert _write_grant(_CANON_T, DbRole.CANONICAL_WRITE) in plan
    assert _write_grant(_MASTER_T, DbRole.MASTER_DATA_WRITE) in plan
    assert _write_grant(_STATE_T, DbRole.AGENT_STATE_WRITE) in plan

    # Reciprocal denial: every OFF-diagonal write grant is absent from the plan.
    denied = [
        (_CANON_T, DbRole.MASTER_DATA_WRITE),
        (_CANON_T, DbRole.READ_ONLY),
        (_CANON_T, DbRole.AGENT_STATE_WRITE),
        (_MASTER_T, DbRole.CANONICAL_WRITE),
        (_MASTER_T, DbRole.READ_ONLY),
        (_MASTER_T, DbRole.AGENT_STATE_WRITE),
        (_STATE_T, DbRole.CANONICAL_WRITE),
        (_STATE_T, DbRole.MASTER_DATA_WRITE),
        (_STATE_T, DbRole.READ_ONLY),
    ]
    for table, role in denied:
        assert _write_grant(table, role) not in plan, (table, role)

    # Read-only OUTSIDE the write domain (ARCH-R3): SELECT-only on the canonical partitions
    # for the cross-domain canonical roles and the read-only role...
    for table in (_CANON_T, _MASTER_T):
        assert _read_grant(table, DbRole.READ_ONLY) in plan
    assert _read_grant(_MASTER_T, DbRole.CANONICAL_WRITE) in plan
    assert _read_grant(_CANON_T, DbRole.MASTER_DATA_WRITE) in plan
    # ...and the agent-state store grants NOTHING (not even SELECT) to the canonical roles,
    # nor the canonical store anything to the agent-state role (store separation, ARCH-R13).
    for stmt in plan:
        if _STATE_T in stmt:
            assert f'"{_ROLE_NAMES[DbRole.AGENT_STATE_WRITE]}"' in stmt
        if (
            f'"{_ROLE_NAMES[DbRole.AGENT_STATE_WRITE]}"' in stmt
            and stmt.startswith("GRANT")
            and "ON s." in stmt
        ):
            assert _STATE_T in stmt


def test_grant_plan_requires_every_role_configured() -> None:
    """A missing role name/password fails closed — never a partial, silently-open plan."""
    incomplete = {k: v for k, v in _ROLE_NAMES.items() if k is not DbRole.MASTER_DATA_WRITE}
    with pytest.raises(ValueError, match="master_data_write"):
        grant_plan(
            schema="s",
            canonical_tables=[_CANON_T],
            master_data_tables=[_MASTER_T],
            agent_state_tables=[_STATE_T],
            role_names=incomplete,
            passwords=_PASSWORDS,
        )
