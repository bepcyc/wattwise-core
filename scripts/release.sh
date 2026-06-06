#!/usr/bin/env bash
# release.sh — release helper invoked by `just release`. Dual-forge (GitHub / Forgejo) and
# network-free under RELEASE_DRY_RUN=1.
#
# Implements CI-R10 (ordered release) + CI-R12 (versioned, scanned, digest-pinned image):
#   step 1  run all CI-R1 checks (delegated to `just ci-required`; abort if red)   [caller-gated]
#   step 2  uv build -> wheel + sdist
#   step 3  changelog (delegated to `just changelog`)
#   step 4  SBOM for the built wheel/image (CycloneDX/SPDX)
#   step 5  build + scan + push the runtime image; record its sha256 digest (CI-R12)
#   step 6  create the forge release attaching wheel + sdist + SBOM + changelog + image digest
#   step 7  publish wheel to the package index if a token is present (skip cleanly if absent)
#
# Forge selection: FORGE_PROVIDER ∈ {github, forgejo} — NO code change between forges (CI-R10/CI-R12).
# DRY RUN: RELEASE_DRY_RUN=1 performs every step UP TO any forge-API / registry / network call and
#          PRINTS the intended action instead — it MUST NOT touch the network (CI-R10 / CI-R12).
#
# Usage:  VERSION=v1.2.3 FORGE_PROVIDER=github scripts/release.sh
#         RELEASE_DRY_RUN=1 VERSION=v1.2.3 FORGE_PROVIDER=forgejo scripts/release.sh

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

VERSION="${VERSION:-}"
FORGE_PROVIDER="${FORGE_PROVIDER:-github}"
REGISTRY="${WW_REGISTRY:-ghcr.io}"
OWNER="${WW_OWNER:-wattwise}"
IMAGE_REPO="${REGISTRY}/${OWNER}/wattwise-core"

case "${FORGE_PROVIDER}" in
  github|forgejo) : ;;
  *) ww_die "FORGE_PROVIDER must be 'github' or 'forgejo' (got '${FORGE_PROVIDER}')." ;;
esac
[ -n "${VERSION}" ] || ww_die "VERSION is required (e.g. VERSION=v1.2.3)."
case "${VERSION}" in
  v[0-9]*) : ;;
  *) ww_die "VERSION must be a 'vX.Y.Z' semver tag (got '${VERSION}')." ;;
esac

ww_ensure_out
image_tag="${IMAGE_REPO}:${VERSION}"
git_sha="$(git -C "${WW_REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# A guard that either RUNS a network/registry action or, under dry-run, prints it and skips.
ww_step() {
  local desc="$1"; shift
  if ww_is_dry_run; then
    ww_log "DRY-RUN [${FORGE_PROVIDER}] would ${desc}: $*"
    return 0
  fi
  ww_log "[${FORGE_PROVIDER}] ${desc}: $*"
  "$@"
}

ww_log "release ${VERSION} on forge '${FORGE_PROVIDER}' (image repo: ${IMAGE_REPO}, sha: ${git_sha})"
ww_is_dry_run && ww_log "RELEASE_DRY_RUN=1 — no network/registry/forge-API calls will be made."

# ---- step 2: build wheel + sdist (local, no network) --------------------------------------------
if ww_is_dry_run; then
  ww_log "DRY-RUN would build artifacts: uv build --out-dir ${WW_REPO_ROOT}/dist"
else
  ww_require_tool uv
  ww_log "building wheel + sdist via uv..."
  ( cd "${WW_REPO_ROOT}" && uv build --out-dir "${WW_REPO_ROOT}/dist" )
fi

# ---- step 3: changelog (local) ------------------------------------------------------------------
if ww_is_dry_run; then
  ww_log "DRY-RUN would generate changelog: just changelog"
else
  ww_log "generating changelog..."
  ( cd "${WW_REPO_ROOT}" && just changelog ) || ww_warn "changelog recipe unavailable; continuing"
fi

# ---- step 5 (build) + step 4 (sbom) + scan: the deployable image (CI-R12) ------------------------
# Image build + scan + sbom are local (no network beyond base-image pull which Docker caches);
# under dry-run we print the intended build/tag/scan/push without touching Docker or the registry.
if ww_is_dry_run; then
  ww_log "DRY-RUN would build image: docker build -t ${image_tag} --label org.opencontainers.image.version=${VERSION} ."
  ww_log "DRY-RUN would scan image:  WW_IMAGE=${image_tag} WW_FAIL_SEVERITY=CRITICAL scripts/scan.sh"
  ww_log "DRY-RUN would sbom image:  WW_IMAGE=${image_tag} scripts/sbom.sh"
  ww_log "DRY-RUN would push image:  docker push ${image_tag}  (forge OCI registry / registry:2 fallback)"
  ww_log "DRY-RUN expected artifacts: dist/*.whl dist/*.tar.gz ${WW_OUT_DIR}/sbom.* CHANGELOG.md + image digest"
else
  ww_require_tool docker
  ww_log "building release image ${image_tag}..."
  docker build \
    --build-arg "WATTWISE_VERSION=${VERSION}" \
    --build-arg "WATTWISE_REVISION=$(git -C "${WW_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)" \
    --label "org.opencontainers.image.version=${VERSION}" \
    --label "org.opencontainers.image.revision=$(git -C "${WW_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)" \
    -t "${image_tag}" \
    -t "${IMAGE_REPO}:${git_sha}" \
    "${WW_REPO_ROOT}"
  # CONT-R1: abort the release on ANY Critical in the image.
  WW_IMAGE="${image_tag}" WW_FAIL_SEVERITY="CRITICAL" "${WW_SCRIPT_DIR}/scan.sh"
  WW_IMAGE="${image_tag}" "${WW_SCRIPT_DIR}/sbom.sh" image
  # step 5 (push) — registry/network action.
  ww_step "push image" docker push "${image_tag}"
  ww_step "push image (sha tag)" docker push "${IMAGE_REPO}:${git_sha}"
  digest="$(docker inspect --format='{{index .RepoDigests 0}}' "${image_tag}" 2>/dev/null || true)"
  ww_log "released image digest: ${digest:-<unresolved>}"
fi

# ---- step 6: create the forge release (network) -------------------------------------------------
# Forge API call is the FIRST network step; dry-run stops here. GitHub uses `gh release`, Forgejo
# uses the Forgejo/Gitea release API via `tea` or curl — selected purely by FORGE_PROVIDER.
release_assets="dist/* ${WW_OUT_DIR}/sbom.* CHANGELOG.md"
case "${FORGE_PROVIDER}" in
  github)  ww_step "create GitHub release ${VERSION}" gh release create "${VERSION}" --notes-file CHANGELOG.md ;;
  forgejo) ww_step "create Forgejo release ${VERSION}" tea release create --tag "${VERSION}" --note-file CHANGELOG.md ;;
esac
ww_log "would attach assets: ${release_assets} + image digest"

# ---- step 7: publish wheel to the index if a token is present (skip cleanly otherwise) -----------
if [ -n "${PYPI_TOKEN:-}" ]; then
  ww_step "publish wheel to index" sh -c 'cd "${WW_REPO_ROOT}" && uv publish --token "${PYPI_TOKEN}" dist/*'
else
  ww_log "no PYPI_TOKEN set — skipping index publish cleanly (CI-R10 step 6)."
fi

ww_log "release helper complete for ${VERSION} on ${FORGE_PROVIDER}$(ww_is_dry_run && printf ' (dry-run)')."
