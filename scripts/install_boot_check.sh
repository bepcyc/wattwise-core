#!/usr/bin/env bash
# install_boot_check.sh — prove the PACKAGED artifact boots offline (CI-R1 item 20, COMM-R12).
#
# Invoked by the Justfile `install-boot-check` recipe (which depends on `build`, so dist/ is
# normally already populated; this script also (re)builds defensively if the wheel is missing).
#
# What this gate asserts — the wheel is a self-contained, installable, importable artifact, NOT
# just "it works in the editable checkout":
#   1. A wheel exists in dist/ (built via `uv build` if absent).
#   2. It installs into a CLEAN, THROWAWAY virtualenv — separate from the project's editable
#      `.venv`, with NO access to the `src/` source tree on sys.path.
#   3. From that installed package (verified to resolve INSIDE the venv's site-packages, never the
#      repo `src/`), a tiny OFFLINE smoke imports `wattwise_core` and loads the settings entrypoint
#      (`wattwise_core.config.get_settings`) — proving the packaged engine boots with no network.
#
# Fail-closed (SEC-R13 idiom): a missing tool, an absent/ambiguous wheel, an install failure, a
# source-tree leak, or a smoke-import failure all DIE loudly — the gate is never a silent green.
# The throwaway venv is always removed on exit (trap), so nothing leaks into the working tree.
#
# Usage:  scripts/install_boot_check.sh          # build-if-needed, install, smoke
#         WW_KEEP_VENV=1 scripts/install_boot_check.sh   # keep the venv for debugging

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

dist_dir="${WW_REPO_ROOT}/dist"
src_dir="${WW_REPO_ROOT}/src"

# --- 1. ensure a wheel exists (build defensively; the just recipe normally already did) ----------
shopt -s nullglob
wheels=("${dist_dir}"/wattwise_core-*.whl)
shopt -u nullglob

if [ "${#wheels[@]}" -eq 0 ]; then
  ww_log "no wheel in dist/ — building with 'uv build'"
  ww_require_tool uv "Install uv: https://docs.astral.sh/uv/. The package-build gate cannot run without it."
  ( cd "${WW_REPO_ROOT}" && uv build ) || ww_die "'uv build' failed — cannot produce a wheel to boot-check."
  shopt -s nullglob
  wheels=("${dist_dir}"/wattwise_core-*.whl)
  shopt -u nullglob
fi

[ "${#wheels[@]}" -ne 0 ] || ww_die "no wattwise_core wheel found in ${dist_dir} after build — failing the gate (COMM-R12)."
if [ "${#wheels[@]}" -gt 1 ]; then
  ww_die "multiple wheels in ${dist_dir} (${wheels[*]}) — ambiguous boot-check target; run 'just clean' then rebuild."
fi
wheel="${wheels[0]}"
ww_log "boot-checking wheel: ${wheel}"

# --- 2. clean throwaway venv (NOT the project's editable .venv) -----------------------------------
venv_dir="$(mktemp -d "${TMPDIR:-/tmp}/ww-install-boot.XXXXXX")"
cleanup() {
  if [ "${WW_KEEP_VENV:-0}" = "1" ]; then
    ww_warn "WW_KEEP_VENV=1 — leaving throwaway venv at ${venv_dir}"
  else
    rm -rf "${venv_dir}"
  fi
}
trap cleanup EXIT INT TERM

# Build the venv with uv if present (fast, honours .python-version), else stdlib venv.
if ww_have uv; then
  uv venv "${venv_dir}/.venv" >&2 || ww_die "failed to create throwaway venv with uv."
else
  ww_require_tool python3 "No uv and no python3 — cannot create a clean venv for the boot-check."
  python3 -m venv "${venv_dir}/.venv" >&2 || ww_die "failed to create throwaway venv with python3 -m venv."
fi

venv_python="${venv_dir}/.venv/bin/python"
[ -x "${venv_python}" ] || ww_die "throwaway venv python not found at ${venv_python}."

# --- 3. install the wheel into that clean env -----------------------------------------------------
ww_log "installing wheel into clean venv ${venv_dir}/.venv"
if ww_have uv; then
  # --python pins the install to the throwaway interpreter; no editable, no source tree.
  uv pip install --python "${venv_python}" "${wheel}" >&2 \
    || ww_die "installing the wheel into the clean venv failed — the packaged artifact is not installable (COMM-R12)."
else
  "${venv_python}" -m pip install "${wheel}" >&2 \
    || ww_die "installing the wheel into the clean venv failed — the packaged artifact is not installable (COMM-R12)."
fi

# --- 4. OFFLINE smoke: import + settings entrypoint, FROM THE INSTALLED PACKAGE -------------------
# Run from a neutral working dir (the venv tmp dir, never the repo root) so the repo's `src/` cannot
# leak onto sys.path via "" / CWD. The smoke also ASSERTS the loaded module resolves inside the
# venv site-packages and NOT inside the repo `src/` — that is the whole point of this gate.
#
# Settings boot fail-closed (BOOT-R4): in the default (production) environment the loader REQUIRES
# the DSN + encryption/signing secrets. We feed throwaway, obviously-fake values and an offline
# in-process SQLite DSN so the smoke exercises the real, strict config validator with NO network
# and NO live services — the secrets below are NOT real credentials and never leave this process.
ww_log "running offline boot smoke from the installed package"
WW_EXPECT_VENV="${venv_dir}/.venv" WW_REPO_SRC="${src_dir}" \
  WATTWISE_DATABASE_DSN="sqlite+aiosqlite:///:memory:" \
  WATTWISE_ENCRYPTION_ROOT_KEY="boot-check-not-a-real-key-do-not-use" \
  WATTWISE_TOKEN_SIGNING_KEY="boot-check-not-a-real-key-do-not-use" \
  "${venv_python}" - <<'PY' >&2 || ww_die "offline boot smoke FAILED — the packaged artifact does not boot (COMM-R12)."
import os
from pathlib import Path

# A clean install must not silently fall back to the editable source tree.
repo_src = Path(os.environ["WW_REPO_SRC"]).resolve()
expect_venv = Path(os.environ["WW_EXPECT_VENV"]).resolve()

import wattwise_core
from wattwise_core.config import get_settings

mod_path = Path(wattwise_core.__file__).resolve()
if repo_src in mod_path.parents:
    raise SystemExit(
        f"import resolved to the repo source tree ({mod_path}), not the installed wheel — "
        "the boot-check is not testing the packaged artifact."
    )
if expect_venv not in mod_path.parents:
    raise SystemExit(
        f"import resolved outside the throwaway venv ({mod_path}); expected under {expect_venv}."
    )

# Boot the settings entrypoint offline (defaults; DSN-only on SQLite, no network, no live services).
settings = get_settings()
if settings is None:
    raise SystemExit("get_settings() returned None — settings did not boot.")

print(f"[smoke] installed wattwise_core OK: {mod_path}")
PY

ww_log "install-boot-check PASSED — packaged wheel boots offline from a clean install."
