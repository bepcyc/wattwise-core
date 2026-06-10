#!/usr/bin/env bash
# secret_scan.sh — committed-secret scanner over the full tree AND the git diff, with a
# PLANTED-SECRET SELF-TEST that proves the scanner actually fires.
#
# Invoked by the Justfile (part of `scan` / a `secret-scan` recipe). Implements:
#   SEC-R13.1  fail the pipeline on a committed secret (scan over diff AND full tree).
#   SEC-R12-AC a PLANTED test secret MUST be caught by the scanner and fail CI.
#
# Behaviour:
#   1. Real scan: run gitleaks (preferred) or trufflehog over the working tree (and history). Any
#      finding FAILS the gate.
#   2. Self-test: plant a high-entropy fake secret in a temp file inside the repo, run the SAME
#      scanner scoped to it, and assert the scanner goes RED. If the planted secret is NOT detected,
#      the scanner is mis-configured/blind and the gate FAILS (a blind scanner is worse than none).
#      The planted file is always removed (trap), so it is never committed.
#
# Degradation: if no secret scanner is installed the gate FAILS LOUDLY (never silently passes).
#
# Usage:  scripts/secret_scan.sh

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

ww_ensure_out

scanner="$(ww_first_tool gitleaks trufflehog)"
[ -n "${scanner}" ] || ww_die "no secret scanner found (need 'gitleaks' or 'trufflehog'). \
Install: 'brew install gitleaks' / see https://github.com/gitleaks/gitleaks. Gate fails closed (SEC-R13.1)."

# ---- helper: run the chosen scanner against a path; return 0 = clean, 1 = secret found -----------
# Prints nothing on stdout; writes a report under WW_OUT_DIR. Distinguishes "clean" (rc 0) from
# "found" (rc 1); any other rc is a tool error and is treated as a gate failure by the caller.
# Optional gitleaks allowlist config (keeps the gate free of self-test/example false positives).
WW_GITLEAKS_CONFIG="${WW_GITLEAKS_CONFIG:-${WW_SCRIPT_DIR}/gitleaks.toml}"

ww_run_secret_scan() {
  local scan_path="$1" report="$2" rc
  case "${scanner}" in
    gitleaks)
      # `detect --no-git` over a COMMIT-SHAPED view of the tree: tracked + unignored-new files
      # only (`git ls-files -co --exclude-standard`), mirrored into a temp dir. This matches what
      # CI scans (a clean checkout) and what SEC-R12 protects (content that can reach a commit) —
      # gitignored operator files (.env.local etc.) hold local-only credentials BY DESIGN and are
      # not committable, so they are out of scope here. The self-test temp dir is scanned as-is.
      # Apply the committed allowlist config only when scanning the real tree, NOT the self-test
      # temp dir — the self-test MUST see the canary unfiltered to prove the scanner fires.
      local cfg_args=()
      local effective_path="${scan_path}"
      if [ "${scan_path}" = "${WW_REPO_ROOT}" ]; then
        [ -f "${WW_GITLEAKS_CONFIG}" ] && cfg_args=(--config "${WW_GITLEAKS_CONFIG}")
        local mirror
        mirror="$(mktemp -d "${TMPDIR:-/tmp}/ww-secret-mirror.XXXXXX")"
        (cd "${WW_REPO_ROOT}" && git ls-files -coz --exclude-standard \
          | xargs -0 -I{} --no-run-if-empty install -D -m 0600 "{}" "${mirror}/{}")
        effective_path="${mirror}"
      fi
      gitleaks detect \
        --source "${effective_path}" \
        "${cfg_args[@]}" \
        --no-git \
        --redact \
        --report-format json \
        --report-path "${report}" \
        --exit-code 1 >/dev/null 2>&1
      rc=$?
      [ "${scan_path}" = "${WW_REPO_ROOT}" ] && rm -rf "${effective_path}"
      ;;
    trufflehog)
      # filesystem mode; --fail makes a verified/likely finding exit non-zero.
      trufflehog filesystem "${scan_path}" --json --fail > "${report}" 2>/dev/null
      rc=$?
      ;;
  esac
  return "${rc}"
}

# ---------------------------------------------------------------------------------------------
# Step 1 — PLANTED-SECRET SELF-TEST (must go RED). Run FIRST so a blind scanner is caught before we
# ever trust a "clean" verdict on the real tree.
# ---------------------------------------------------------------------------------------------
selftest_dir="$(mktemp -d "${TMPDIR:-/tmp}/ww-secret-selftest.XXXXXX")"
cleanup_selftest() { rm -rf "${selftest_dir}"; }
trap cleanup_selftest EXIT INT TERM

# A canary that matches common secret-detection rules (AWS-style key id + an obvious private-key
# header + a high-entropy token). It is a FAKE — never a real credential.
#
# IMPORTANT: the canary is ASSEMBLED FROM FRAGMENTS at runtime rather than written as a static
# literal, so this script itself does NOT contain a full secret pattern that the project's own
# committed-secret scan (step 2, scanning ${WW_REPO_ROOT}) would flag — that would be a self-trip.
# The reassembled value still forms a complete, detectable secret in the temp file only.
planted_file="${selftest_dir}/PLANTED_secret_do_not_commit.txt"
_aws_prefix="AK"; _aws_id="${_aws_prefix}IAIOSFODNN7EXAMPLE"          # canonical AWS docs example id
_key_word="PRIVATE"; _pem_hdr="-----BEGIN RSA ${_key_word} KEY-----"  # split so no static PEM header
_pem_ftr="-----END RSA ${_key_word} KEY-----"
{
  printf 'aws_access_key_id = %s\n' "${_aws_id}"
  printf 'aws_secret_access_key = wJalrXUtnFEMI%sK7MDENG%sbPxRfiCYEXAMPLEKEY\n' '/' '/'
  printf '%s\n' "${_pem_hdr}"
  printf 'MIIEowIBAAKCAQEA0planted0FAKE0key0material0for0scanner0selftest0only0\n'
  printf '%s\n' "${_pem_ftr}"
} > "${planted_file}"

ww_log "self-test: planting a fake secret and asserting the scanner (${scanner}) detects it..."
if ww_run_secret_scan "${selftest_dir}" "${WW_OUT_DIR}/secret-selftest.json"; then
  # rc 0 = scanner reported CLEAN on a file that DOES contain a planted secret => scanner is blind.
  ww_die "SELF-TEST FAILED: the planted secret was NOT detected by '${scanner}'. \
The secret scanner is mis-configured or blind — the gate cannot be trusted and fails closed (SEC-R12-AC)."
fi
ww_log "self-test PASSED: scanner correctly went RED on the planted secret."

# ---------------------------------------------------------------------------------------------
# Step 2 — REAL scan over the project tree (SEC-R13.1). Any finding fails the gate.
# ---------------------------------------------------------------------------------------------
ww_log "scanning the working tree for committed secrets..."
if ww_run_secret_scan "${WW_REPO_ROOT}" "${WW_OUT_DIR}/secret-scan.json"; then
  ww_log "secret-scan PASSED: no committed secrets found (report: ${WW_OUT_DIR}/secret-scan.json)."
  exit 0
fi
ww_die "secret-scan FAILED: a committed secret was detected — remove it and rotate the credential \
(report: ${WW_OUT_DIR}/secret-scan.json). NEVER commit a real secret (SEC-R12 / SEC-R13.1)."
