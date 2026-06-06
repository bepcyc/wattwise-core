#!/usr/bin/env bash
# rollback.sh — one-step rollback to the previously-deployed image tag. Invoked by `just rollback`.
#
# Implements the DEPLOY-R9 one-step rollback: on a failed/regressed deploy, restore the prior image
# tag (recorded by deploy.sh) WITHOUT touching the data. The isolated persistent volume
# `wattwise_pgdata` (DEPLOY-R8) is NEVER destroyed here — only the app container's image is swapped
# back. `down -v` is never invoked.
#
# Usage:  scripts/rollback.sh
#         WATTWISE_IMAGE='<registry>/<owner>/wattwise-core:<prev>@sha256:<digest>' scripts/rollback.sh

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

COMPOSE_FILE="${WW_REPO_ROOT}/deploy/compose.yaml"
STATE_DIR="${WW_STATE_DIR:-${WW_REPO_ROOT}/.deploy}"
ROLLBACK_FILE="${STATE_DIR}/rollback-image"

[ -f "${COMPOSE_FILE}" ] || ww_die "compose file not found: ${COMPOSE_FILE}"
ww_require_tool docker

# Resolve the rollback image: explicit override wins; otherwise the tag deploy.sh recorded.
if [ -n "${WATTWISE_IMAGE:-}" ]; then
  target="${WATTWISE_IMAGE}"
elif [ -s "${ROLLBACK_FILE}" ]; then
  target="$(cat "${ROLLBACK_FILE}")"
else
  ww_die "no rollback target found (${ROLLBACK_FILE} absent). Pass WATTWISE_IMAGE=<prev-ref> explicitly."
fi

ww_log "DEPLOY-R8: rollback swaps ONLY the app image. The data volume 'wattwise_pgdata' is untouched; \
'docker compose down -v' is never invoked."
ww_log "rolling back wattwise-core to: ${target}"

export WATTWISE_IMAGE="${target}"
# Recreate ONLY the app service against the prior image; Postgres keeps running on its durable volume.
docker compose -f "${COMPOSE_FILE}" up -d --no-deps --force-recreate wattwise-core

# Verify the restored deployment serves (CI-R7).
host_port="${WATTWISE_HOST_PORT:-58000}"
ok=0
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${host_port}/readyz" >/dev/null 2>&1; then ok=1; break; fi
  sleep 2
done
[ "${ok}" -eq 1 ] || ww_die "rollback target ${target} did not become ready — investigate manually (data volume is intact)."

ww_log "rollback complete — ${target} is healthy and serving. Data volume preserved."
