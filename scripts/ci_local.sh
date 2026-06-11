#!/usr/bin/env bash
# Full-fidelity LOCAL mirror of the GitHub/Forgejo CI pipeline (CI-R0: the justfile is the single
# source of gate truth — this script runs the SAME recipe per CI job, in the SAME serial order,
# with the SAME service shape: throwaway PostgreSQL 16 + MariaDB 11 on non-default localhost
# ports, per-run generated boot secrets, the /healthz bootstrap-acceptance probe, the package
# build, and (unless WW_SKIP_IMAGE=1) the container image scan + SBOM.
#
# Purpose: a change that passes `just ci-local` and still fails CI is an environment finding to
# investigate, never a "worked on my machine" surprise. Hand-rolled gate subsets are forbidden.
#
# Knobs:
#   WW_SKIP_IMAGE=1     skip image build + scan/SBOM (the slowest leg; CI still runs it)
#   WW_REPEAT_DB=N      repeat the db-portable leg N times (flake hunting for races; default 1)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Tooling exactly as CI provisions it (scripts/ci_tools.sh installs into ~/.local/bin).
export PATH="$HOME/.local/bin:$PATH"
command -v gitleaks >/dev/null 2>&1 && command -v trivy >/dev/null 2>&1 \
  || bash scripts/ci_tools.sh gitleaks trivy

PG_PORT="${WW_CI_PG_PORT:-55461}"
# CI's runner binds the app on 8000; a dev box may have 8000 occupied (e.g. a docker-published
# service) — that is an environment artifact, not a finding, so the local probe picks a free
# port by default while staying overridable for exact-CI replication (WW_CI_APP_PORT=8000).
APP_PORT="${WW_CI_APP_PORT:-$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')}"
MD_PORT="${WW_CI_MD_PORT:-53361}"
PG_NAME="wwci-pg-$$"
MD_NAME="wwci-mariadb-$$"
REPEAT_DB="${WW_REPEAT_DB:-1}"

log()  { printf '\n[ci-local] ===== %s =====\n' "$*"; }
fail() { printf '[ci-local][RED] %s\n' "$*" >&2; exit 1; }

cleanup() {
  docker rm -f "$PG_NAME" "$MD_NAME" >/dev/null 2>&1 || true
  [ -n "${UV_PID:-}" ] && kill "$UV_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- per-run generated boot secrets (mirrors the workflow step; BOOT-R4) ---
# CI grants these ONLY to the service-booting jobs (integration / db-portability / e2e);
# the fast-stage jobs run WITHOUT them — so they are NOT exported globally here either
# (a global export already masked a real wiring-test divergence once). BOOT_ENV expands
# them per-leg exactly like the workflow step does.
ROOT_KEY="$(python3 -c 'import secrets,base64;print(base64.b64encode(secrets.token_bytes(32)).decode())')"
SIGNING_KEY="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
boot_env() { env WATTWISE_ENCRYPTION_ROOT_KEY="$ROOT_KEY" WATTWISE_TOKEN_SIGNING_KEY="$SIGNING_KEY" "$@"; }

# ----------------------------------------------------------------- FAST STAGE
# NOTE: legs are SEPARATE statements, never `a && b` chains — under `set -e` a failure
# in any non-final segment of an AND-list does NOT abort the script (bash exempts && / ||
# list members), which once let a red lint leg fall through to a green banner.
log "Lint (just lint + fmt-check)";            just lint
just fmt-check
log "Type check (just type)";                  just type
log "Conventional commits (just lint-commits)"; just lint-commits || fail "lint-commits"
log "Pre-commit config validity";              uv run pre-commit validate-config
log "Fast tiers"
just test-unit
just test-property
just test-golden
just test-contract
just test-fuzz
just test-logging
log "Coverage gate (just cov)";                just cov
log "Agent eval gate (just eval)";             just eval
log "Injection gate (just test-inject)";       just test-inject
log "Secret + dependency scan (just scan)";    just scan
log "Forge portability";                       just test-forge-portable

# ----------------------------------------------------------------- SLOW STAGE
log "starting throwaway PostgreSQL 16 + MariaDB 11 (tmpfs, --rm, 127.0.0.1, non-default ports)"
docker run -d --rm --name "$PG_NAME" --tmpfs /var/lib/postgresql/data \
  -p "127.0.0.1:${PG_PORT}:5432" -e POSTGRES_USER=wattwise -e POSTGRES_PASSWORD=wattwise \
  -e POSTGRES_DB=wattwise postgres:16 >/dev/null
docker run -d --rm --name "$MD_NAME" --tmpfs /var/lib/mysql \
  -p "127.0.0.1:${MD_PORT}:3306" -e MARIADB_USER=wattwise -e MARIADB_PASSWORD=wattwise \
  -e MARIADB_DATABASE=wattwise -e MARIADB_ROOT_PASSWORD=rootwattwise mariadb:11 >/dev/null
for i in $(seq 1 60); do docker exec "$PG_NAME" pg_isready -U wattwise >/dev/null 2>&1 && break; sleep 1; done
for i in $(seq 1 90); do docker exec "$MD_NAME" healthcheck.sh --connect --innodb_initialized >/dev/null 2>&1 && break; sleep 2; done

PG_DSN="postgresql+asyncpg://wattwise:wattwise@127.0.0.1:${PG_PORT}/wattwise"
MD_DSN="mysql+aiomysql://wattwise:wattwise@127.0.0.1:${MD_PORT}/wattwise"

log "Integration (T-INT): migrate + test-integration on PostgreSQL"
boot_env env WATTWISE_DATABASE_DSN="$PG_DSN" just migrate
boot_env env WATTWISE_DATABASE_DSN="$PG_DSN" just test-integration

log "Bootstrap & DB portability (x${REPEAT_DB})"
for n in $(seq 1 "$REPEAT_DB"); do
  [ "$REPEAT_DB" -gt 1 ] && log "db-portable round ${n}/${REPEAT_DB}"
  boot_env env WATTWISE_PG_DSN="$PG_DSN" WATTWISE_MARIADB_DSN="$MD_DSN" just test-db-portable
done

log "Bootstrap acceptance (/healthz serves) — mirrors the workflow step"
rm -f ./.wattwise-ci.sqlite
boot_env env WATTWISE_DATABASE_DSN="sqlite+aiosqlite:///./.wattwise-ci.sqlite" just migrate
boot_env env WATTWISE_DATABASE_DSN="sqlite+aiosqlite:///./.wattwise-ci.sqlite" \
  uv run uvicorn --factory wattwise_core.api.app:create_app --host 127.0.0.1 --port "$APP_PORT" &
UV_PID=$!
ok=""
for i in $(seq 1 30); do curl -fsS http://127.0.0.1:${APP_PORT}/healthz >/dev/null 2>&1 && ok=1 && break; sleep 1; done
kill "$UV_PID" >/dev/null 2>&1 || true; UV_PID=""
rm -f ./.wattwise-ci.sqlite
[ -n "$ok" ] || fail "bootstrap acceptance: /healthz never served"

log "E2E smoke (just test-e2e)";               boot_env just test-e2e
log "Package build (uv build)";                just build

if [ "${WW_SKIP_IMAGE:-0}" != "1" ]; then
  log "Container image scan + SBOM (just sbom)"; just sbom
else
  log "WW_SKIP_IMAGE=1 — image scan/SBOM SKIPPED (CI will still run it)"
fi

log "ALL CI-MIRROR LEGS GREEN"
