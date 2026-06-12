#!/usr/bin/env bash
# ci_tools.sh — install the CI toolchain binaries a workflow job needs (CI-R0/CI-R9).
#
# Both forges' runners (GitHub ubuntu-latest, Forgejo docker containers) ship WITHOUT
# `just`, `gitleaks`, `trivy`, or `syft`. This script is the single forge-portable
# plumbing step that installs the requested tools into ~/.local/bin from each tool's
# OFFICIAL installer/release, then exposes that dir to subsequent steps via GITHUB_PATH
# (supported by GitHub Actions AND Forgejo Actions runners alike).
#
# It is CI plumbing, NOT a gate (CI-R0: zero gate logic outside the Justfile) — the
# same role as actions/checkout or setup-uv, kept as a script so the tool set cannot
# diverge between the two forges' workflow files.
#
# Usage:  bash scripts/ci_tools.sh just [gitleaks] [trivy] [syft]

set -euo pipefail

BIN_DIR="${WW_CI_BIN:-${HOME}/.local/bin}"
mkdir -p "${BIN_DIR}"
export PATH="${BIN_DIR}:${PATH}"
# Make the dir visible to every later step in the job (no-op outside Actions).
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "${BIN_DIR}" >> "${GITHUB_PATH}"
fi

# Pinned fallback used only when the GitHub "latest release" API is unreachable.
GITLEAKS_FALLBACK_VERSION="8.21.2"

# Bounded-retry flags for EVERY external fetch (issue #37): a single transient 403 /
# timeout / CDN hiccup must not fail a job. `--retry-all-errors` makes curl retry on
# HTTP 4xx/5xx too (not just connection errors), `--retry-delay` + curl's own backoff
# space the attempts out, and `--fail` keeps a final hard failure visible (non-zero exit).
CURL_RETRY=(--retry 5 --retry-delay 3 --retry-all-errors --fail)

# When a GitHub token is present (GITHUB_TOKEN on the runner, or GH_TOKEN locally),
# authenticate GitHub release/API requests — authenticated calls get a far higher rate
# limit, which is the actual cause of the transient 403s. Absence of a token changes
# NOTHING: the array stays empty and requests go out unauthenticated exactly as before.
GH_TOKEN_VALUE="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

log() { printf '[ci-tools] %s\n' "$*" >&2; }

install_just() {
  curl "${CURL_RETRY[@]}" --proto '=https' --tlsv1.2 -fsSL https://just.systems/install.sh \
    | bash -s -- --to "${BIN_DIR}"
}

install_trivy() {
  curl "${CURL_RETRY[@]}" -fsSL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh \
    | sh -s -- -b "${BIN_DIR}"
}

install_syft() {
  curl "${CURL_RETRY[@]}" -fsSL https://raw.githubusercontent.com/anchore/syft/main/install.sh \
    | sh -s -- -b "${BIN_DIR}"
}

install_gitleaks() {
  local version auth=()
  if [ -n "${GH_TOKEN_VALUE}" ]; then
    auth=(-H "Authorization: Bearer ${GH_TOKEN_VALUE}")
  fi
  version="$(curl "${CURL_RETRY[@]}" "${auth[@]}" -fsSL https://api.github.com/repos/gitleaks/gitleaks/releases/latest 2>/dev/null \
    | grep -oE '"tag_name": *"v[^"]+"' | sed -E 's/.*"v([^"]+)".*/\1/' || true)"
  [ -n "${version}" ] || version="${GITLEAKS_FALLBACK_VERSION}"
  curl "${CURL_RETRY[@]}" "${auth[@]}" -fsSL "https://github.com/gitleaks/gitleaks/releases/download/v${version}/gitleaks_${version}_linux_x64.tar.gz" \
    | tar -xz -C "${BIN_DIR}" gitleaks
}

[ "$#" -gt 0 ] || { log "no tools requested — nothing to do."; exit 0; }

for tool in "$@"; do
  if command -v "${tool}" >/dev/null 2>&1; then
    log "${tool} already present: $(command -v "${tool}")"
    continue
  fi
  case "${tool}" in
    just)     install_just ;;
    trivy)    install_trivy ;;
    syft)     install_syft ;;
    gitleaks) install_gitleaks ;;
    *) log "unknown tool '${tool}' — supported: just trivy syft gitleaks"; exit 2 ;;
  esac
  command -v "${tool}" >/dev/null 2>&1 || { log "install of '${tool}' failed."; exit 1; }
  log "installed ${tool}: $("${tool}" --version 2>/dev/null | head -n1 || true)"
done
