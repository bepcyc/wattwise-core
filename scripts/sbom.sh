#!/usr/bin/env bash
# sbom.sh — generate a Software Bill of Materials for the wattwise-core image (or filesystem).
#
# Invoked by the Justfile `sbom` recipe. Implements CONT-R1 / CI-R6 / CI-R10 step 4: every release
# produces an SBOM in a standard format (CycloneDX or SPDX), retained as a CI artifact and attached
# to the forge release.
#
# Behaviour:
#   * Uses syft. If syft is absent the gate FAILS LOUDLY (a missing SBOM fails the gate — CONT-R1:
#     "Scan-cannot-run / missing SBOM ... fails gate"). It does NOT silently produce an empty file.
#   * Default format CycloneDX JSON; override with WW_SBOM_FORMAT (e.g. spdx-json).
#   * Target defaults to the image ${WW_IMAGE}; pass `fs` as arg 1 to SBOM the source tree instead.
#
# Usage:  WW_IMAGE=wattwise-core:vX.Y.Z scripts/sbom.sh
#         scripts/sbom.sh fs                       # SBOM the project filesystem (wheel inputs)
#         WW_SBOM_FORMAT=spdx-json scripts/sbom.sh

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

ww_ensure_out

ww_require_tool syft "Install: 'brew install syft' or see https://github.com/anchore/syft. A missing SBOM fails the release gate (CONT-R1)."

sbom_format="${WW_SBOM_FORMAT:-cyclonedx-json}"
target_kind="${1:-image}"

case "${target_kind}" in
  image) target="${WW_IMAGE}";              out="${WW_OUT_DIR}/sbom.${sbom_format}.json" ;;
  fs)    target="dir:${WW_REPO_ROOT}";      out="${WW_OUT_DIR}/sbom-fs.${sbom_format}.json" ;;
  *)     ww_die "unknown SBOM target '${target_kind}' (expected 'image' or 'fs')." ;;
esac

ww_log "generating ${sbom_format} SBOM for ${target} -> ${out}"
syft scan "${target}" -o "${sbom_format}=${out}"

# A zero-byte / missing SBOM is a gate failure, not a pass.
[ -s "${out}" ] || ww_die "SBOM generation produced no output (${out}) — failing the gate (CONT-R1)."

ww_log "SBOM written: ${out}"
printf '%s\n' "${out}"
