"""Per-layer database roles — the canonical write-paths are structural, not conventional.

ARCH-R3 partitions the canonical store into two write domains: (a) **source-derived
canonical data** is written **only** by the Ingestion/Sync service, and (b) **athlete-
authored master-data** (profile, zone definitions, fitness signature/thresholds, goals,
the canonical-backed ``/v1/user-settings/*`` settings) is written **only** through the
API's distinct **master-data-write** role. Every layer's access to data *outside* its own
write domain MUST be read-only at the data-access-credential level — "a structural, not
merely conventional, guarantee". DEPLOY-R4 fixes the FOUR roles this implies and requires
they be DISTINCT per write domain:

1. **canonical-write** — the Ingestion/Sync service against the source-derived canonical
   tables ONLY (ARCH-R3(a)); rejected on the master-data tables and the agent-state store;
2. **master-data-write** — the API's master-data endpoints against the athlete-authored
   master-data tables ONLY (ARCH-R3(b)); rejected on the source-derived canonical tables
   and the agent-state store; the ONLY write surface for master-data;
3. **read-only** — the Domain/Analytics services (used by both API and MCP/agent) against
   the canonical store (BOTH partitions), no write anywhere;
4. **agent-state-write** — the Agent Orchestrator against the agent-state store ONLY; it
   MUST NOT write the canonical store (either partition) and MUST be distinct from the
   other three.

This enforces ARCH-R3 (no source-derived canonical writes outside Ingestion; master-data
only via the master-data-write role) and ARCH-R13 (the canonical store and the agent-state
store never share a write credential) at the infrastructure level, verified by the
ARCH-R24 reciprocal write-denial role test.

This module is the provisioning helper a deploy uses to create the four roles and grant
each one ONLY its domain's privileges (a privilege-minimal GRANT/REVOKE routine), plus the
:class:`DbRole` identities the engine layer maps each layer's session to. It carries NO
configuration value (CFG-R1a): role NAMES, the schema, the table sets, and the role
credentials are all passed in by the caller, resolved from defaults/env at the deploy seam.
The privilege model is PostgreSQL's role/GRANT system (the recommended production backend,
ARCH-R13/DEPLOY-R2); SQLite/MariaDB have no equivalent per-connection role and the structural
guarantee is therefore a PostgreSQL property (the role test runs only there). The grant
PLAN itself (:func:`grant_plan`) is pure and backend-free, so the full reciprocal-denial
matrix is also asserted structurally in the default (SQLite) gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class DbRole(StrEnum):
    """The four distinct per-write-domain database roles (DEPLOY-R4).

    The enum VALUE is a stable, layer-naming token (not a deployment role name — the actual
    role names are supplied by configuration). It identifies which privilege class a layer's
    session connects under, so the engine wiring is explicit about the access each layer holds.
    """

    #: Ingestion/Sync (L3) — the ONLY writer to the SOURCE-DERIVED canonical tables (ARCH-R3a).
    CANONICAL_WRITE = "canonical_write"
    #: The API's master-data endpoints — the ONLY writer to the athlete-authored master-data
    #: tables (ARCH-R3b: profile, zones, fitness signature, goals, canonical user-settings).
    MASTER_DATA_WRITE = "master_data_write"
    #: Domain/Analytics (L5), consumed by both the REST API and the MCP/agent — read-only
    #: against the canonical store, BOTH partitions (ARCH-R3 "outside the write domain is
    #: read-only at the credential level").
    READ_ONLY = "read_only"
    #: Agent Orchestrator (L6) — writes the agent-state store ONLY; cannot write canonical.
    AGENT_STATE_WRITE = "agent_state_write"


def _quote_ident(identifier: str) -> str:
    """Quote a SQL identifier, rejecting an embedded quote (no role-name SQL injection)."""
    if '"' in identifier:
        raise ValueError(f"invalid SQL identifier (embedded quote): {identifier!r}")
    return f'"{identifier}"'


def _quote_literal(value: str) -> str:
    """Quote a SQL string literal (role password) by doubling single quotes."""
    return "'" + value.replace("'", "''") + "'"


def _create_role_sql(name_q: str, name_lit: str, pw_lit: str) -> str:
    """An idempotent ``CREATE ROLE ... LOGIN`` guarded so re-provisioning is a no-op.

    Pre-quoted inputs (``name_q`` an identifier via :func:`_quote_ident`; ``name_lit`` /
    ``pw_lit`` literals via :func:`_quote_literal`) make the string-built DDL injection-safe.
    PostgreSQL-only role DDL (no portable ORM equivalent), outside the BOOT-R3 data path.
    """
    exists = f"SELECT 1 FROM pg_roles WHERE rolname = {name_lit}"  # noqa: S608  # noqa: no-vendor-sql
    return (
        f"DO $$ BEGIN IF NOT EXISTS ({exists}) "
        f"THEN CREATE ROLE {name_q} LOGIN PASSWORD {pw_lit}; END IF; END $$"
    )


def grant_plan(
    *,
    schema: str,
    canonical_tables: Sequence[str],
    master_data_tables: Sequence[str],
    agent_state_tables: Sequence[str],
    role_names: Mapping[DbRole, str],
    passwords: Mapping[DbRole, str],
) -> list[str]:
    """The ordered, privilege-minimal GRANT/REVOKE statement plan for the four roles.

    Pure (no I/O), so the DEPLOY-R4 reciprocal-denial matrix is asserted structurally by
    the unit gate on any backend:

    * **canonical-write** — full DML on the source-derived canonical tables; SELECT-only on
      the master-data tables (read-only outside its write domain, ARCH-R3); NO grant on the
      agent-state tables;
    * **master-data-write** — full DML on the master-data tables ONLY; SELECT-only on the
      source-derived canonical tables; NO grant on the agent-state tables;
    * **read-only** — SELECT on BOTH canonical partitions; NO write grant anywhere, and NO
      grant on the agent-state tables;
    * **agent-state-write** — full DML on the agent-state tables only; NO grant on either
      canonical partition (it cannot even read the canonical store — DEPLOY-R4/ARCH-R13).

    The default ``PUBLIC`` privileges on the schema are revoked first so a role holds ONLY
    what is granted here (fail-closed: absence of a grant is a denial, not an open default).
    Role names / passwords / schema / table sets are caller-supplied (CFG-R1a).
    """
    missing = [role for role in DbRole if role not in role_names or role not in passwords]
    if missing:
        raise ValueError(f"grant_plan requires a name + password for every role: {missing}")

    schema_q = _quote_ident(schema)
    statements: list[str] = [
        # Fail-closed default: PUBLIC gets nothing on this schema; each role holds ONLY its
        # explicit grants below. (USAGE on the schema is granted per-role where needed.)
        f"REVOKE ALL ON SCHEMA {schema_q} FROM PUBLIC",
    ]
    for role in DbRole:
        name_q = _quote_ident(role_names[role])
        name_lit = _quote_literal(role_names[role])
        pw_lit = _quote_literal(passwords[role])
        # CREATE ROLE is not idempotent; guard it in a DO block so re-provisioning is safe.
        # This module is INHERENTLY PostgreSQL-specific role provisioning (CREATE ROLE / GRANT
        # have no portable ORM equivalent and never run on SQLite/MariaDB); it is NOT a canonical
        # data-path query, so the BOOT-R3 portability rule does not apply. Identifiers go through
        # _quote_ident (rejects embedded quotes) and literals through _quote_literal, so the
        # string-built DDL is injection-safe despite the static SQL-string warnings.
        statements.append(_create_role_sql(name_q, name_lit, pw_lit))
        statements.append(f"GRANT USAGE ON SCHEMA {schema_q} TO {name_q}")

    cw_q = _quote_ident(role_names[DbRole.CANONICAL_WRITE])
    mdw_q = _quote_ident(role_names[DbRole.MASTER_DATA_WRITE])
    ro_q = _quote_ident(role_names[DbRole.READ_ONLY])
    asw_q = _quote_ident(role_names[DbRole.AGENT_STATE_WRITE])

    for table in canonical_tables:
        # canonical-write: full DML on its OWN domain; master-data-write + read-only: SELECT
        # only (read-only outside the write domain); agent-state-write: NOTHING (cannot even
        # reach the canonical store — DEPLOY-R4).
        statements.append(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {cw_q}")
        statements.append(f"GRANT SELECT ON {table} TO {mdw_q}")
        statements.append(f"GRANT SELECT ON {table} TO {ro_q}")
    for table in master_data_tables:
        # master-data-write: full DML on its OWN domain (the ONLY master-data write surface,
        # ARCH-R3b); canonical-write + read-only: SELECT only; agent-state-write: NOTHING.
        statements.append(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {mdw_q}")
        statements.append(f"GRANT SELECT ON {table} TO {cw_q}")
        statements.append(f"GRANT SELECT ON {table} TO {ro_q}")
    for table in agent_state_tables:
        # agent-state-write: full DML on agent-state; the canonical-store roles: NOTHING
        # (reciprocal denial — the two stores never share a write credential, ARCH-R13).
        statements.append(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {asw_q}")
    return statements


async def provision_roles(
    admin_engine: AsyncEngine,
    *,
    schema: str,
    canonical_tables: Sequence[str],
    master_data_tables: Sequence[str],
    agent_state_tables: Sequence[str],
    role_names: Mapping[DbRole, str],
    passwords: Mapping[DbRole, str],
) -> None:
    """Create the four per-domain roles and grant each ONLY its privileges (DEPLOY-R4).

    Runs under an ADMIN connection (a role that may ``CREATE ROLE`` / ``GRANT``), executing
    the pure :func:`grant_plan` — see it for the privilege matrix. Idempotent: each role is
    created only if absent. Role names / passwords / schema / table sets are caller-supplied
    (CFG-R1a) — this routine bakes none of them.
    """
    statements = grant_plan(
        schema=schema,
        canonical_tables=canonical_tables,
        master_data_tables=master_data_tables,
        agent_state_tables=agent_state_tables,
        role_names=role_names,
        passwords=passwords,
    )
    async with admin_engine.begin() as conn:
        for stmt in statements:
            await conn.execute(text(stmt))


async def teardown_roles(
    admin_engine: AsyncEngine,
    *,
    schema: str,
    role_names: Mapping[DbRole, str],
) -> None:
    """Drop the provisioned roles and their schema (test/throwaway cleanup).

    Drops the schema CASCADE (removing the grants that would otherwise block ``DROP ROLE``),
    then drops each role if present. Idempotent and best-effort — intended for throwaway test
    databases, never for a production store.
    """
    schema_q = _quote_ident(schema)
    async with admin_engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_q} CASCADE"))
        for role in DbRole:
            name = role_names.get(role)
            if name is None:
                continue
            # Reassign/drop owned objects first would be needed for owned data; the throwaway
            # roles here own nothing outside the dropped schema, so a guarded DROP ROLE suffices.
            await conn.execute(text(f"DROP ROLE IF EXISTS {_quote_ident(name)}"))


__all__ = ["DbRole", "grant_plan", "provision_roles", "teardown_roles"]
