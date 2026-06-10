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

log() { printf '[ci-tools] %s\n' "$*" >&2; }

install_just() {
  curl --proto '=https' --tlsv1.2 -fsSL https://just.systems/install.sh \
    | bash -s -- --to "${BIN_DIR}"
}

install_trivy() {
  curl -fsSL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh \
    | sh -s -- -b "${BIN_DIR}"
}

install_syft() {
  curl -fsSL https://raw.githubusercontent.com/anchore/syft/main/install.sh \
    | sh -s -- -b "${BIN_DIR}"
}

install_gitleaks() {
  local version
  version="$(curl -fsSL https://api.github.com/repos/gitleaks/gitleaks/releases/latest 2>/dev/null \
    | grep -oE '"tag_name": *"v[^"]+"' | sed -E 's/.*"v([^"]+)".*/\1/' || true)"
  [ -n "${version}" ] || version="${GITLEAKS_FALLBACK_VERSION}"
  curl -fsSL "https://github.com/gitleaks/gitleaks/releases/download/v${version}/gitleaks_${version}_linux_x64.tar.gz" \
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
