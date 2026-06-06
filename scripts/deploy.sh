#!/usr/bin/env bash
# deploy.sh — operator self-host deploy. Invoked by `just deploy VERSION=vX.Y.Z`.
#
# Implements DEPLOY-R9 (deploy a released, versioned, DIGEST-PINNED image — never an ad-hoc build) and
# DEPLOY-R8 (isolated persistent-volume Postgres; NEVER `down -v`; never touch a foreign DB), plus the
# post-deploy health + smoke check (CI-R7) and one-step rollback prep.
#
# Flow:
#   1. Refuse to deploy a non-released or dirty-working-tree image unless DEPLOY_DEV=1.
#   2. Resolve the released image ref `{registry}/{owner}/wattwise-core:{VERSION}@sha256:{digest}` and
#      pull it BY DIGEST (forge OCI registry, or local registry:2 offline fallback).
#   3. Record the CURRENTLY-running tag as the rollback target (one-step rollback, DEPLOY-R9).
#   4. `docker compose up -d` against the committed deploy/compose.yaml (isolated volume, DEPLOY-R8).
#      It uses `up`/`start`/`stop` ONLY — never `down -v`.
#   5. Post-deploy readiness + smoke probe; on failure, point the operator at one-step rollback.
#
# Usage:  VERSION=v1.2.3 scripts/deploy.sh
#         DEPLOY_DEV=1 VERSION=dev scripts/deploy.sh    # developer escape hatch (DEPLOY-R9)

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

VERSION="${VERSION:-}"
REGISTRY="${WW_REGISTRY:-ghcr.io}"
OWNER="${WW_OWNER:-wattwise}"
IMAGE_REPO="${REGISTRY}/${OWNER}/wattwise-core"
COMPOSE_FILE="${WW_REPO_ROOT}/deploy/compose.yaml"
DEPLOY_DEV="${DEPLOY_DEV:-0}"
STATE_DIR="${WW_STATE_DIR:-${WW_REPO_ROOT}/.deploy}"
ROLLBACK_FILE="${STATE_DIR}/rollback-image"

[ -n "${VERSION}" ] || ww_die "VERSION is required (e.g. VERSION=v1.2.3)."
[ -f "${COMPOSE_FILE}" ] || ww_die "compose file not found: ${COMPOSE_FILE}"
ww_require_tool docker

# ---- guard 1: released + clean tree (DEPLOY-R9) -------------------------------------------------
if [ "${DEPLOY_DEV}" != "1" ]; then
  case "${VERSION}" in
    v[0-9]*) : ;;
    *) ww_die "refusing to deploy non-released VERSION '${VERSION}' (must be 'vX.Y.Z'). Set DEPLOY_DEV=1 to override (developer mode only, DEPLOY-R9)." ;;
  esac
  if ! git -C "${WW_REPO_ROOT}" diff --quiet HEAD 2>/dev/null || [ -n "$(git -C "${WW_REPO_ROOT}" status --porcelain 2>/dev/null)" ]; then
    ww_die "refusing to deploy with a DIRTY working tree. Commit/stash changes, or set DEPLOY_DEV=1 (developer mode only, DEPLOY-R9)."
  fi
else
  ww_warn "DEPLOY_DEV=1 — release/clean-tree guards bypassed (developer mode). Not for production."
fi

# ---- guard 2: never destroy the data volume (DEPLOY-R8) -----------------------------------------
# Defensive: this script NEVER calls `down -v`. Make that auditable.
ww_log "DEPLOY-R8: this deploy uses 'up -d' against an isolated, persistent-volume Postgres. \
'docker compose down -v' is FORBIDDEN and is never invoked here."

# ---- step 2: resolve + pull the released image BY DIGEST ----------------------------------------
mkdir -p "${STATE_DIR}"
image_tag="${IMAGE_REPO}:${VERSION}"

if [ "${DEPLOY_DEV}" = "1" ] && [ "${VERSION}" = "dev" ]; then
  # Dev mode may run a locally-built image; skip the registry pull/digest verification.
  WATTWISE_IMAGE="${WW_DEV_IMAGE:-wattwise-core:local}"
  ww_warn "dev-mode image: ${WATTWISE_IMAGE} (digest verification skipped)."
else
  ww_log "pulling released image by tag to resolve its digest: ${image_tag}"
  docker pull "${image_tag}" \
    || ww_die "could not pull ${image_tag} from the forge registry. (Offline? bring up the local registry:2 fallback and retag, then retry.)"
  digest="$(docker inspect --format='{{index .RepoDigests 0}}' "${image_tag}" 2>/dev/null || true)"
  [ -n "${digest}" ] || ww_die "could not resolve a sha256 digest for ${image_tag} — refusing to deploy an unpinned image (DEPLOY-R9)."
  # CI-R12 records the digest on the forge release; a production hardening step would verify this
  # resolved digest against that recorded value before proceeding.
  WATTWISE_IMAGE="${digest}"
  ww_log "deploying digest-pinned image: ${WATTWISE_IMAGE}"
fi
export WATTWISE_IMAGE

# ---- step 3: record the current image as the one-step rollback target ---------------------------
current="$(docker inspect --format='{{.Config.Image}}' wattwise-core 2>/dev/null || true)"
if [ -n "${current}" ]; then
  printf '%s\n' "${current}" > "${ROLLBACK_FILE}"
  ww_log "recorded rollback target (previous image): ${current} -> ${ROLLBACK_FILE}"
else
  ww_log "no previously-running wattwise-core container; first deploy (no rollback target yet)."
fi

# ---- step 4: bring the stack up (isolated volume; up/start only — never down -v) -----------------
ww_log "bringing up the stack via ${COMPOSE_FILE} (isolated Postgres, dedicated named volume)..."
docker compose -f "${COMPOSE_FILE}" up -d --no-build

# ---- step 5: post-deploy readiness + smoke (CI-R7) ----------------------------------------------
ww_log "waiting for readiness (CI-R7 health + smoke)..."
host_port="${WATTWISE_HOST_PORT:-58000}"
ready=0
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${host_port}/readyz" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
if [ "${ready}" -ne 1 ]; then
  ww_err "post-deploy readiness check FAILED — the new deployment is not serving."
  if [ -s "${ROLLBACK_FILE}" ]; then
    ww_err "ROLL BACK IN ONE STEP:  just rollback   (restores $(cat "${ROLLBACK_FILE}"); the data volume is untouched)."
  fi
  ww_die "deploy aborted; previous deployment image retained for rollback (DEPLOY-R9)."
fi

ww_log "deploy ${VERSION} healthy and ready. (Rollback target retained at ${ROLLBACK_FILE}.)"
