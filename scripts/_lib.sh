#!/usr/bin/env bash
# Shared helpers for the wattwise-core supply-chain / deploy scripts.
#
# Sourced by scan.sh / sbom.sh / secret_scan.sh / release.sh / deploy.sh / rollback.sh.
# Provides: strict-mode setup, logging, repo-root resolution, tool-presence checks that DEGRADE
# WITH A CLEAR MESSAGE (never silently pass — SEC-R13 gates must be honest), and a dry-run guard.

set -euo pipefail

# --- repo root (scripts/ lives directly under the repo root) -------------------------------------
WW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WW_REPO_ROOT="$(cd "${WW_SCRIPT_DIR}/.." && pwd)"
export WW_REPO_ROOT

# --- logging (stderr; stdout reserved for machine-readable artifacts) ----------------------------
ww_log()  { printf '[wattwise] %s\n'        "$*" >&2; }
ww_warn() { printf '[wattwise][WARN] %s\n'  "$*" >&2; }
ww_err()  { printf '[wattwise][ERROR] %s\n' "$*" >&2; }

# Fail closed with a clear message. Scanner-absence and gate breaches both route here so a CI gate
# is NEVER a silent green (SEC-R13: a gate that cannot run MUST fail, not pass).
ww_die() { ww_err "$*"; exit 1; }

# True if a binary is on PATH.
ww_have() { command -v "$1" >/dev/null 2>&1; }

# First available tool from a list, echoed; empty string if none. Used to pick e.g. trivy|grype.
ww_first_tool() {
  local t
  for t in "$@"; do
    if ww_have "$t"; then printf '%s' "$t"; return 0; fi
  done
  printf ''
}

# Require a tool or DIE with install guidance. A missing scanner must fail the gate (not skip it).
ww_require_tool() {
  local tool="$1" hint="${2:-}"
  if ! ww_have "$tool"; then
    ww_die "required tool '${tool}' is not installed — the gate cannot run and MUST NOT pass silently.${hint:+ }${hint}"
  fi
}

# Output dir for generated reports/SBOMs (kept out of the build context via .dockerignore).
WW_OUT_DIR="${WW_OUT_DIR:-${WW_REPO_ROOT}/.scan}"
ww_ensure_out() { mkdir -p "${WW_OUT_DIR}"; }

# Dry-run flag honored by release.sh / deploy.sh (RELEASE_DRY_RUN hits no network).
ww_is_dry_run() { [ "${RELEASE_DRY_RUN:-0}" = "1" ]; }

# Severity threshold for SCA/image gates (SEC-R13.2 / CONT-R1). Default HIGH,CRITICAL.
WW_FAIL_SEVERITY="${WW_FAIL_SEVERITY:-HIGH,CRITICAL}"

# Image reference to scan/build. Overridable; defaults to a local dev tag.
WW_IMAGE="${WW_IMAGE:-wattwise-core:local}"
