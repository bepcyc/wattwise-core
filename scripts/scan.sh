#!/usr/bin/env bash
# scan.sh — vulnerability scan of the wattwise-core image AND the project filesystem.
#
# Invoked by the Justfile `scan` recipe. Implements SEC-R13.2 (SCA: fail on a dependency vuln at/above
# a severity threshold) and CONT-R1 (image scan, FAIL on any Critical; no Critical OS/package admitted).
#
# Behaviour:
#   * Prefers Trivy; falls back to Grype. If NEITHER is installed the gate FAILS LOUDLY (a scan that
#     cannot run must not pass silently — SEC-R13).
#   * Scans (a) the built image `${WW_IMAGE}` and (b) the repo filesystem (lockfile-driven SCA).
#   * Fails on findings at or above ${WW_FAIL_SEVERITY} (default HIGH,CRITICAL).
#
# Usage:  WW_IMAGE=wattwise-core:vX.Y.Z scripts/scan.sh
#         WW_FAIL_SEVERITY=CRITICAL scripts/scan.sh   # CONT-R1 image gate (Critical-only)
#         WW_SCAN_TARGETS=fs scripts/scan.sh          # lockfile SCA only (no image built yet)
#
# WW_SCAN_TARGETS selects which targets run: "image,fs" (default), "image", or "fs".
# The fs-only mode exists for the PR-stage SCA gate, which runs BEFORE any image is
# built; the image gate (CONT-R1) runs in the sbom/release path against a built tag.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

ww_ensure_out

scanner="$(ww_first_tool trivy grype)"
[ -n "${scanner}" ] || ww_die "no image/SCA scanner found (need 'trivy' or 'grype'). Install one: \
'brew install trivy' / 'apt-get install trivy' or 'brew install grype'. Gate fails closed."

WW_SCAN_TARGETS="${WW_SCAN_TARGETS:-image,fs}"
case "${WW_SCAN_TARGETS}" in
  image,fs|fs,image|image|fs) : ;;
  *) ww_die "WW_SCAN_TARGETS must be 'image,fs', 'image' or 'fs' (got '${WW_SCAN_TARGETS}')." ;;
esac
scan_image=0; scan_fs=0
case ",${WW_SCAN_TARGETS}," in *,image,*) scan_image=1 ;; esac
case ",${WW_SCAN_TARGETS}," in *,fs,*) scan_fs=1 ;; esac

image_target="${WW_IMAGE}"
fs_target="${WW_REPO_ROOT}"
rc=0

ww_log "vulnerability scan via '${scanner}' (targets: ${WW_SCAN_TARGETS}), failing on severities: ${WW_FAIL_SEVERITY}"

case "${scanner}" in
  trivy)
    # Image scan — fail on the configured severities (CONT-R1). `--exit-code 1` makes a finding a
    # non-zero exit = gate failure. Report is retained for CI artifacts (CI-R6).
    if [ "${scan_image}" -eq 1 ]; then
      ww_log "scanning image: ${image_target}"
      if ! trivy image \
            --severity "${WW_FAIL_SEVERITY}" \
            --exit-code 1 \
            --ignorefile .trivyignore \
            --format table \
            --output "${WW_OUT_DIR}/trivy-image.txt" \
            "${image_target}"; then
        ww_warn "image scan found vulnerabilities at/above ${WW_FAIL_SEVERITY} (see ${WW_OUT_DIR}/trivy-image.txt)"
        rc=1
      fi
    fi
    # Filesystem / lockfile SCA (SEC-R13.2) — catches a vulnerable pinned dependency.
    if [ "${scan_fs}" -eq 1 ]; then
      ww_log "scanning filesystem (SCA): ${fs_target}"
      if ! trivy fs \
            --severity "${WW_FAIL_SEVERITY}" \
            --exit-code 1 \
            --scanners vuln \
            --format table \
            --output "${WW_OUT_DIR}/trivy-fs.txt" \
            "${fs_target}"; then
        ww_warn "filesystem SCA found vulnerabilities at/above ${WW_FAIL_SEVERITY} (see ${WW_OUT_DIR}/trivy-fs.txt)"
        rc=1
      fi
    fi
    ;;
  grype)
    # Grype expresses the threshold via --fail-on (lowest severity that fails). We pass the lowest of
    # the configured set; HIGH,CRITICAL → fail-on high.
    fail_on="$(printf '%s' "${WW_FAIL_SEVERITY}" | tr 'A-Z,' 'a-z\n' | grep -vx '' | sort | head -n1)"
    : "${fail_on:=high}"
    if [ "${scan_image}" -eq 1 ]; then
      ww_log "scanning image: ${image_target} (fail-on=${fail_on})"
      if ! grype "${image_target}" --fail-on "${fail_on}" -c .grype.yaml -o table > "${WW_OUT_DIR}/grype-image.txt"; then
        ww_warn "image scan found vulnerabilities at/above ${fail_on} (see ${WW_OUT_DIR}/grype-image.txt)"
        rc=1
      fi
    fi
    if [ "${scan_fs}" -eq 1 ]; then
      ww_log "scanning filesystem (SCA): ${fs_target} (fail-on=${fail_on})"
      if ! grype "dir:${fs_target}" --fail-on "${fail_on}" -o table > "${WW_OUT_DIR}/grype-fs.txt"; then
        ww_warn "filesystem SCA found vulnerabilities at/above ${fail_on} (see ${WW_OUT_DIR}/grype-fs.txt)"
        rc=1
      fi
    fi
    ;;
esac

if [ "${rc}" -ne 0 ]; then
  ww_die "vulnerability gate FAILED — fix or pin out the flagged packages before release (reports in ${WW_OUT_DIR})."
fi
ww_log "vulnerability gate PASSED — no findings at/above ${WW_FAIL_SEVERITY}."
