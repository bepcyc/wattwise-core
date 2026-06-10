# wattwise-core — canonical developer/ops task runner (RUN-R3.3, CI-R0).
#
# This Justfile is the SINGLE SOURCE OF TRUTH for every gate. CI (GitHub Actions
# AND Forgejo Actions, CI-R9/DELIV-R8) are thin schedulers that invoke ONLY these
# recipes — there is ZERO gate logic in the workflow YAML. The exact command a
# contributor runs locally (`just lint`, `just type`, `just test`, ...) is the
# command CI runs (CI-R0: no CI-only logic).
#
# It is NOT a product CLI (DELIV-R4 / RUN-R3.3): no `coach` console, no end-user
# runtime. Every recipe is a thin shell wrapper around a tool, resolved through
# `uv run` so dependencies come from the uv-managed, pinned lockfile (RUN-R3.2).
#
# Recipes that depend on a module a sibling agent still owns carry a `TODO:`
# pointing at the owning path; they invoke that path so they go green the moment
# the module lands (no Justfile change needed).

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set dotenv-load := false   # secrets come from the real environment only (BOOT-R4); never auto-load .env

# --- shared knobs (override on the command line, e.g. `just deploy VERSION=v1.2.3`) ---

# OSS package + image coordinates.
package := "wattwise_core"
image_name := "wattwise-core"

# Forge selector for release/deploy recipes (CI-R10/CI-R12): github | forgejo.
FORGE_PROVIDER := env_var_or_default("FORGE_PROVIDER", "github")
# Dry-run guard: when "1", release/deploy print intent and touch NO network (CI-R10/CI-R12).
RELEASE_DRY_RUN := env_var_or_default("RELEASE_DRY_RUN", "0")

# Local dev DSN default (BOOT-R1): a file-backed SQLite so `just bootstrap` and
# the offline tiers run with zero external services. Production overrides this
# env var with a Postgres/MariaDB DSN (BOOT-R3).
WATTWISE_DATABASE_DSN := env_var_or_default("WATTWISE_DATABASE_DSN", "sqlite+aiosqlite:///./.wattwise-dev.sqlite")

# uv invocation prefix (RUN-R3.2). `--frozen` is added by reproducible recipes.
uv := "uv run"

# Default target: print the recipe catalog.
default:
    @just --list --unsorted

# =====================================================================
# 0. Environment
# =====================================================================

# Sync the dev toolchain from the pinned lockfile (CI-R5). Idempotent.
install:
    uv sync --frozen --all-extras

# =====================================================================
# 1. Static gates — lint, format, type (CI-R1 items 1, 2, 14)
# =====================================================================

# Ruff lint (CI-R1 item 1). Style/lint clean, zero errors.
lint: lint-ast lint-content lint-novendor-sql lint-arch
    {{uv}} ruff check src tests tools

# Ruff format (write). Use `just fmt-check` in CI for a non-mutating gate.
fmt:
    {{uv}} ruff format src tests tools

# Ruff format check (non-mutating) — the CI form of `fmt`.
fmt-check:
    {{uv}} ruff format --check src tests tools

# mypy --strict via committed pyproject config (CI-R1 item 2, QUAL-R8).
type:
    {{uv}} mypy

# --- code-craft AST lints (CI-R1 item 14) — owned by tools/lint/ (sibling agent) ---

# Module-/function-size ceilings, test-docstring rule, English-only identifier
# scan (QUAL-R9/R10b/R11). Implemented as a custom AST check, NOT a native ruff
# rule (CI-R1 item 14).
# TODO(tools/lint): module owned by the lint sibling — `tools/lint/__main__.py`.
lint-ast:
    {{uv}} python -m tools.lint ast src

# Content/copy gate (CI-R1 item 21, QUAL-R13(j)): banned-blame/edgy word lists,
# please/sorry + `!`-in-errors bans, internals-leak regexes, catalog-key
# resolution, error-code uniqueness, validation-specificity.
# TODO(tools/lint): `tools/lint/content.py` owned by the lint sibling.
lint-content:
    {{uv}} python -m tools.lint content src

# Static no-vendor-SQL check (BOOT-R3): zero dialect-specific SQL/DDL in app code.
# TODO(tools/lint): `tools/lint/novendor_sql.py` owned by the persistence/lint sibling.
lint-novendor-sql:
    {{uv}} python -m tools.lint novendor-sql src

# Import-direction / architecture gate (Principle A/B layering).
# TODO(tools/lint): `tools/lint/arch.py` owned by the lint sibling.
lint-arch:
    {{uv}} python -m tools.lint arch src

# =====================================================================
# 2. Test tiers (TIER-R3: one marker per test) — CI-R1 items 3, 4, 12, 16
# =====================================================================

# The full suite (all tiers). Inner-loop devs use the targeted recipes below.
test:
    {{uv}} pytest

# Fast offline tiers (TIER-R1: no network/credentials). CI-R1 item 3.
test-unit:
    {{uv}} pytest -m unit

test-property:
    {{uv}} pytest -m property

test-golden:
    {{uv}} pytest -m golden

test-contract:
    {{uv}} pytest -m contract

# Parser/decoder fuzzing — bounded, deterministic PR-gate mode (TIER-R5 (a)).
# CI-R1 item 16. Hypothesis-driven, ≤ 3 min, reproducible (no coverage engine).
# TODO(tests): requires the `fuzz` pytest marker (registered by Dev B in
# pyproject `[tool.pytest.ini_options].markers`) and the parser/decoder suites.
test-fuzz:
    {{uv}} pytest -m fuzz

# Integration tier against a real ephemeral master store (TIER-R2). CI-R1 item 4.
# DSN is taken from the environment; CI provisions ephemeral Postgres/MariaDB.
test-integration:
    {{uv}} pytest -m integration

# API-level E2E smoke over the built stack (E2E-R1, CI-R1 item 12).
test-e2e:
    {{uv}} pytest -m e2e

# Live-LLM tier (env-gated; never part of the offline gate, TIER-R1). Requires
# WATTWISE_LLM_API_KEY (and optionally WATTWISE_AGENT__MODEL); tests skip without it.
test-llm:
    {{uv}} pytest -m llm

# =====================================================================
# 3. Coverage gate (CI-R1 item 5)
# =====================================================================

# Combined coverage ≥ 80% overall AND a ≥ 95% line+branch FLOOR on the two
# correctness-critical packages — analytics + ingestion adapters (DOD-R1). The
# global run emits XML + a retained term report (CI-R6); the scoped run enforces
# the per-package floor. Both must pass for `cov` to go green.
cov: cov-critical
    {{uv}} pytest --cov={{package}} --cov-branch \
        --cov-report=term-missing \
        --cov-report=xml:reports/coverage.xml \
        --cov-fail-under=80

# Per-package ≥ 95% line+branch FLOOR for the analytics + ingestion-adapter
# packages (DOD-R1). Scoped `--cov` to exactly those two import roots so the
# `--cov-fail-under=95` threshold is measured against them ALONE (not diluted by
# the rest of the package, which the global ≥ 80% floor in `cov` covers). Branch
# coverage is on via [tool.coverage.run]. Report retained for CI artifacts (CI-R6).
cov-critical:
    {{uv}} pytest \
        --cov=wattwise_core.analytics \
        --cov=wattwise_core.ingestion.adapters \
        --cov-branch \
        --cov-report=term-missing \
        --cov-report=xml:reports/coverage-critical.xml \
        --cov-fail-under=95

# =====================================================================
# 4. Agent eval + injection tiers (CI-R1 items 6, 7)
# =====================================================================

# Agent eval gate in recorded-response mode (QA-EVAL-R6/R7, QA-EVAL-R9). CI-R1 item 6.
# TODO(src/wattwise_core/eval): harness owned by the agent/eval sibling
# (`wattwise_core.eval`); recorded-response cassettes are committed fixtures.
eval:
    {{uv}} python -m {{package}}.eval run --mode=recorded --scorecard=reports/eval-scorecard.json

# Refresh recorded-response cassettes (QA-EVAL-R12(a)). Reviewed change only.
# TODO(src/wattwise_core/eval): owned by the agent/eval sibling.
eval-record:
    {{uv}} python -m {{package}}.eval record

# Regenerate the committed baseline scorecard from a clean run (QA-EVAL-R7/-R12(c)).
# Rewrites src/wattwise_core/eval/baseline-scorecard.json; review the diff before commit.
eval-update-baseline:
    {{uv}} python -m {{package}}.eval update-baseline

# Prompt-injection corpus (INJ-R3, T-INJECT). CI-R1 item 7.
test-inject:
    {{uv}} pytest -m inject

# =====================================================================
# 5. Bootstrap + DB portability (CI-R1 item 13; BOOT-R1..R4)
# =====================================================================

# Clone -> running, health-serving instance (BOOT-R1). One documented command.
#
# Steps: install (frozen) -> apply ORM migrations from empty (BOOT-R2) against
# WATTWISE_DATABASE_DSN (defaults to a local SQLite dev DSN) -> start the ASGI
# app via uvicorn. Secrets come from the environment only (BOOT-R4); nothing
# manual beyond providing them.
#
# Entry point CHOSEN: the app factory `wattwise_core.api.app:create_app`
# (uvicorn `--factory`). The engine ships no product CLI / `python -m` runtime
# (DELIV-R4), so the ASGI factory is the single boot surface.
# TODO(src/wattwise_core/api): `wattwise_core.api.app:create_app` owned by the
# API sibling (Dev A). Until it lands this recipe fails at the uvicorn step.
bootstrap: install migrate
    @echo "bootstrap: starting wattwise-core against ${WATTWISE_DATABASE_DSN%%:*}... DSN (BOOT-R1)"
    WATTWISE_DATABASE_DSN="{{WATTWISE_DATABASE_DSN}}" \
        {{uv}} uvicorn --factory {{package}}.api.app:create_app \
        --host 127.0.0.1 --port "${WATTWISE_API__PORT:-8000}"

# Apply versioned ORM migrations from empty (BOOT-R2). Vendor-portable: the same
# Alembic revisions run on SQLite/PostgreSQL/MariaDB (only the DSN differs).
# TODO(migrations): Alembic env + revisions owned by the persistence sibling
# (Dev B); `alembic.ini` + `migrations/`.
migrate:
    WATTWISE_DATABASE_DSN="{{WATTWISE_DATABASE_DSN}}" {{uv}} alembic upgrade head

# Run the portability-marked integration suite across all three backends
# (BOOT-R3, CI-R1 item 13). Asserts identical canonical/analytic outputs with a
# DSN-only difference. The backend DSNs come from the environment so CI can point
# at its service containers; locally only the SQLite leg runs without services.
#
# WATTWISE_PG_DSN / WATTWISE_MARIADB_DSN are set by the CI matrix; when absent the
# corresponding leg is skipped (the SQLite leg always runs).
test-db-portable:
    @echo "portability: SQLite leg (always available)"
    WATTWISE_DATABASE_DSN="sqlite+aiosqlite:///./.wattwise-portable.sqlite" \
        {{uv}} pytest -m "portability or integration"
    @if [ -n "${WATTWISE_PG_DSN:-}" ]; then \
        echo "portability: PostgreSQL leg"; \
        WATTWISE_DATABASE_DSN="$WATTWISE_PG_DSN" {{uv}} pytest -m "portability or integration"; \
    else echo "portability: PostgreSQL leg SKIPPED (WATTWISE_PG_DSN unset)"; fi
    @if [ -n "${WATTWISE_MARIADB_DSN:-}" ]; then \
        echo "portability: MariaDB leg"; \
        WATTWISE_DATABASE_DSN="$WATTWISE_MARIADB_DSN" {{uv}} pytest -m "portability or integration"; \
    else echo "portability: MariaDB leg SKIPPED (WATTWISE_MARIADB_DSN unset)"; fi

# =====================================================================
# 6. Supply-chain + commits + logging (CI-R1 items 8, 9, 10, 11, 15)
# =====================================================================

# Secret scan + dependency/SCA scan (CI-R1 items 8, 9; SEC-R13.1/.2).
# Two fail-closed scripts, both owned by the security sibling — kept as scripts so
# the tool choice is not baked into YAML:
#   * `scripts/secret_scan.sh` — committed-secret scan over tree + diff with a
#     planted-canary self-test (SEC-R13.1 / SEC-R12-AC; gitleaks/trufflehog).
#   * `scripts/scan.sh`        — dependency/SCA + image vuln scan (SEC-R13.2 /
#     CONT-R1; trivy/grype). Secret scan runs FIRST so a leak fails fast.
# The PR-stage scan runs BEFORE any image is built, so it scans the secret surface +
# the repo filesystem (lockfile SCA) only; the image gate runs in `sbom`/release
# against a freshly built tag (WW_SCAN_TARGETS selects the target set).
scan:
    bash scripts/secret_scan.sh
    WW_SCAN_TARGETS=fs bash scripts/scan.sh

# Scan the runtime image (no Critical admitted) and emit an SBOM (CI-R1 item 10;
# CONT-R1). Composed from the two real security-sibling scripts so the tool choice
# stays out of YAML and the image gate matches the SCA gate:
#   1. `scripts/scan.sh` with WW_FAIL_SEVERITY=CRITICAL — image vuln scan that
#      admits NO Critical (CONT-R1). WW_IMAGE selects the built image to scan.
#   2. `scripts/sbom.sh` — syft SBOM (CycloneDX/SPDX) for that image.
# Both retain their reports under reports/ (CI-R6). Override WW_IMAGE to point at
# the freshly built tag, e.g. `WW_IMAGE=wattwise-core:v1.2.3 just sbom`.
# Build the runtime image locally so the image gate has a real target to scan.
image-build:
    docker build -t {{image_name}}:local .

sbom: image-build
    WW_IMAGE={{image_name}}:local WW_FAIL_SEVERITY=CRITICAL WW_SCAN_TARGETS=image bash scripts/scan.sh
    WW_IMAGE={{image_name}}:local bash scripts/sbom.sh

# Conventional-commits gate (CI-R1 item 15, QUAL-R12). Lints PR commit messages.
# TODO(scripts): `scripts/lint_commits.sh` owned by the delivery sibling
# (commitlint / a thin conventional-commit regex over the PR range).
lint-commits:
    bash scripts/lint_commits.sh

# Logging-contract gate (CI-R1 item 11, QA-LOG-R1): no app-written log files,
# structured JSON to stdout/stderr, central redaction. Asserted as a pytest tier.
test-logging:
    {{uv}} pytest -m logging

# =====================================================================
# 7. Forge portability + package-build gates (CI-R1 items 18, 20)
# =====================================================================

# Forge-portability gate (CI-R1 item 18, CI-R9/DELIV-R8): the .github and
# .forgejo workflow files reference an IDENTICAL set of `just` recipes, and both
# `just release` dry-runs (github + forgejo) succeed touching no network.
# TODO(scripts): `scripts/test_forge_portable.py` owned by the delivery sibling
# — it (a) parses both workflow YAMLs and asserts equal recipe sets, then
# (b) shells out to the two dry-runs below. Implemented as a script so there is
# ZERO gate logic in the YAML.
test-forge-portable:
    {{uv}} python scripts/test_forge_portable.py \
        --github .github/workflows \
        --forgejo .forgejo/workflows
    @echo "forge-portable: dry-run release on forgejo"
    RELEASE_DRY_RUN=1 FORGE_PROVIDER=forgejo VERSION=v0.0.0-forge-portable-dryrun just release
    @echo "forge-portable: dry-run release on github"
    RELEASE_DRY_RUN=1 FORGE_PROVIDER=github VERSION=v0.0.0-forge-portable-dryrun just release

# Package-build & install-boot gate (CI-R1 item 20, COMM-R12): `uv build`
# produces wheel+sdist, the wheel installs into a FRESH env (not the editable
# checkout), and the engine boots + passes offline tiers FROM the installed
# package.
# TODO(scripts): `scripts/install_boot_check.sh` owned by the delivery sibling
# (build a clean venv, `pip install dist/*.whl`, run offline tiers from it).
build:
    uv build

install-boot-check: build
    bash scripts/install_boot_check.sh

# =====================================================================
# 8. The umbrella gate — every DETERMINISTIC required check (offline)
# =====================================================================

# Run all deterministic gates that need no live services. This is the inner-loop
# "is my PR green?" command. The service-backed tiers (integration, db-portable,
# e2e, image scan) run in the slow CI stage and via their own recipes — they are
# intentionally NOT in `gate` so it stays fast and offline (CI-R3).
gate: lint fmt-check type lint-commits test-unit test-property test-golden test-contract test-fuzz test-logging eval test-inject cov
    @echo "gate: all deterministic offline required checks passed."

# =====================================================================
# 9. Release (CI-R10/CI-R12; E5-T1) — dual-forge, abort-on-red, dry-run-safe
# =====================================================================

# Automated release. Wired by the CI-R9 workflow files to the `v*`-tag push on
# BOTH forges. Runs IN ORDER (CI-R10):
#   1. all CI-R1 required checks — abort if any is red;
#   2. `uv build` -> wheel + sdist;
#   3. `just changelog` (CHANGELOG from the conforming commit log, QUAL-R12);
#   4. SBOM for the built wheel (CycloneDX/SPDX, CONT-R1);
#   5. build+scan+push the versioned container image (CI-R12);
#   6. create the forge release (github|forgejo via FORGE_PROVIDER) attaching
#      wheel+sdist+SBOM+changelog+image digest;
#   7. publish the wheel to the index (PyPI) via uv IF a token is present, else
#      skip cleanly.
#
# RELEASE_DRY_RUN=1 performs every step up to the forge-API/network boundary and
# PRINTS the intended actions, touching NO network (CI-R10/CI-R12). The whole
# orchestration lives in a script so there is no gate logic in the YAML and the
# abort-on-red + conditional-publish-on-token logic is unit-testable.
# TODO(scripts): `scripts/release.sh` owned by the delivery sibling.
release:
    RELEASE_DRY_RUN="{{RELEASE_DRY_RUN}}" FORGE_PROVIDER="{{FORGE_PROVIDER}}" \
        bash scripts/release.sh

# Update CHANGELOG.md from the conforming Conventional-Commits log (QUAL-R12(c)).
# TODO(scripts): `scripts/changelog.sh` owned by the delivery sibling
# (semantic-release / release-please / git-cliff — the tool is not baked in YAML).
changelog:
    bash scripts/changelog.sh

# =====================================================================
# 10. Deploy (CI-R7/DEPLOY-R9; E5-T8) — digest-pinned, isolated, rollback-safe
# =====================================================================

# Deploy a released version (E5-T8). Pulls the digest-pinned image for VERSION,
# brings it up with `docker compose` against an ISOLATED persistent-volume
# Postgres on a NON-DEFAULT host port with a DEDICATED named volume, runs the
# health + smoke checks, and supports a one-step rollback to the previous
# version. It NEVER runs `docker compose down -v` (that would destroy the
# persistent data volume) — rollback swaps the image tag and restarts only.
#
# Usage: `just deploy VERSION=v1.2.3`
# TODO(scripts): `scripts/deploy.sh` + `deploy/compose.yaml` owned by the ops
# sibling. The compose file MUST: pin the app image by digest recorded on the
# forge release (CI-R12), publish Postgres on a non-default port (e.g. 55432),
# use a dedicated named volume (e.g. `wattwise_pgdata`), and define a one-step
# rollback path. RELEASE_DRY_RUN=1 prints the intended pull/up without network.
deploy VERSION="":
    @test -n "{{VERSION}}" || { echo "deploy: VERSION is required, e.g. just deploy VERSION=v1.2.3"; exit 2; }
    RELEASE_DRY_RUN="{{RELEASE_DRY_RUN}}" FORGE_PROVIDER="{{FORGE_PROVIDER}}" \
        bash scripts/deploy.sh "{{VERSION}}"

# =====================================================================
# 11. Housekeeping
# =====================================================================

# Start a local dev server with autoreload (developer convenience; NOT a gate
# and NOT a product CLI — it just wraps uvicorn against the dev DSN).
# TODO(src/wattwise_core/api): `wattwise_core.api.app:create_app` (Dev A).
dev: migrate
    WATTWISE_APP__ENVIRONMENT=development \
    WATTWISE_DATABASE_DSN="{{WATTWISE_DATABASE_DSN}}" \
        {{uv}} uvicorn --factory {{package}}.api.app:create_app --reload \
        --host 127.0.0.1 --port "${WATTWISE_API__PORT:-8000}"

# Remove build/test/dev artifacts. Never touches committed sources or git state.
clean:
    rm -rf dist build reports .coverage .coverage.* coverage.xml .pytest_cache .ruff_cache .mypy_cache
    rm -f ./.wattwise-dev.sqlite ./.wattwise-portable.sqlite
    find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
