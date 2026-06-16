# wattwise-core — hardened, reproducible runtime image.
#
# Implements the runtime/packaging + supply-chain contracts of the spec:
#   RUN-R1  base image pinned BY DIGEST; deps from a uv FROZEN lockfile; deterministic layers.
#   RUN-R2  multi-stage build → minimal runtime: no build toolchain, no dev deps, no tests,
#           no secrets; non-root user with a FIXED UID; read-only root fs where feasible.
#   RUN-R3.1 runtime targets Python 3.13 (pinned).
#   SEC-R12 / RUN-R10  no hardcoded/fallback secret baked into the image.
#   CONT-R1 multi-stage minimal runtime, non-root fixed UID, reproducible (frozen lock, digest base).
#   CI-R12  OCI labels org.opencontainers.image.{source,revision,version}.
#
# Runtime base choice — python:3.13-slim, NOT distroless (justification):
#   The spec (RUN-R2 / CONT-R1) explicitly allows "Python 3.13-slim OR distroless". Google's
#   distroless `python3-debian12` ships a FIXED Python 3.11 (verified at build time of this file),
#   which would VIOLATE the hard RUN-R3.1 "Python 3.13" mandate — distroless has no 3.13 variant.
#   We therefore use python:3.13-slim-bookworm, which is an allowed minimal runtime, satisfies
#   RUN-R3.1, and lets uvicorn run with a real shell-less-friendly entrypoint. We recover most of
#   distroless's hardening posture manually: non-root fixed UID, no build toolchain in the final
#   layer, a venv copied from the builder, read-only-root-fs-compatible layout, and a tini-free
#   exec-form entrypoint. (If a future distroless image ships Python 3.13, switch the runtime stage
#   to it: copy the same /opt/venv and keep the non-root USER.)
#
# Digest-pin format is `image:tag@sha256:<64-hex>`. The two digests below were resolved live against
# the upstream registries at authoring time. CI (CI-R12) re-resolves/records the digest on release;
# Dependabot/renovate bumps these. If you cannot resolve a digest offline, the pin shape to use is:
#   FROM ghcr.io/astral-sh/uv:<tag>@sha256:<TODO-PIN-DIGEST>   # PIN TODO (offline)

# ---------------------------------------------------------------------------
# Stage 1 — builder: resolve + install the FROZEN locked dependency set with uv.
# ---------------------------------------------------------------------------
# hadolint ignore=DL3007
FROM ghcr.io/astral-sh/uv:0.5.13-python3.13-bookworm-slim@sha256:c64168148341106dd8b9bf8ff1f1c5f443b156c40386b7d0c4ddd4dda42174a0 AS builder

# Deterministic, hermetic install:
#   - compile bytecode so the runtime stage does no first-import write (read-only fs friendly);
#   - copy mode (not hardlink) so the venv is self-contained when copied across stages;
#   - never touch the network for an unpinned resolve.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Layer 1: dependency-only install (no project source) → cached across source-only changes.
# `--frozen` (RUN-R1 / SEC-R13.4 / RUN-R3.2): fail if uv.lock is missing or out of date; never
# silently re-resolve a floating range. `--no-dev` drops the dev/test toolchain (RUN-R2).
# `--no-install-project` installs ONLY third-party deps in this layer.
# The PostgreSQL/MariaDB async drivers are OPTIONAL extras (the bare pip install stays
# lean, SQLite-only) — but the IMAGE is the deployment artifact and must serve any of the
# three supported DSNs (deploy/compose.yaml runs PostgreSQL), so both driver extras are
# installed here (issue #19: without them the documented compose boot cannot connect at all).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra postgresql --extra mariadb

# Layer 2: install the project itself (its own wheel into the same venv).
COPY README.md NOTICE LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra postgresql --extra mariadb

# ---------------------------------------------------------------------------
# Stage 2 — runtime: minimal, non-root, no build tools, no dev deps, no tests.
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm@sha256:05b95397cac02b060ff1251afaa78087d92d7034369afbc8eb765631cada8257 AS runtime

# OCI provenance labels (CI-R12). version/revision are injected at build/release time; defaults keep
# a bare `docker build` honest. ARGs are NOT secrets (SEC-R12) — purely build metadata.
ARG WATTWISE_VERSION="0.0.0-dev"
ARG WATTWISE_REVISION="unknown"
LABEL org.opencontainers.image.title="wattwise-core" \
      org.opencontainers.image.description="Endurance-training analytics engine + trustworthy coaching agent (OSS)." \
      org.opencontainers.image.source="https://github.com/bepcyc/wattwise-core" \
      org.opencontainers.image.url="https://github.com/bepcyc/wattwise-core" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Viacheslav Rodionov" \
      org.opencontainers.image.revision="${WATTWISE_REVISION}" \
      org.opencontainers.image.version="${WATTWISE_VERSION}"

# Fail-closed runtime hygiene:
#   - PYTHONDONTWRITEBYTECODE: no .pyc writes at runtime (read-only-root-fs friendly; bytecode was
#     already compiled in the builder).
#   - PYTHONUNBUFFERED / faulthandler: structured stdout/stderr logging only (LOG-R1), no buffering.
#   - PATH points at the copied venv; no system-site packages.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}" \
    WATTWISE_VERSION="${WATTWISE_VERSION}" \
    WATTWISE_API__HOST="0.0.0.0" \
    WATTWISE_API__PORT="8000"

# Drop privileges to a FIXED, non-root UID/GID (RUN-R2 / CONT-R1). No login shell, no home writes.
# A static UID (10001) keeps file ownership reproducible and lets the platform map it predictably.
RUN groupadd --system --gid 10001 wattwise \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin wattwise

# Copy ONLY the self-contained venv from the builder — no compiler, no uv, no dev deps, no .git,
# no tests, no lockfile, no source tree beyond what the installed wheel already contains.
COPY --from=builder --chown=root:root /opt/venv /opt/venv

# A scratch dir the app may need for ephemeral, NON-PII working files (the engine never writes PII
# to disk — PRIV-R6; persistent athlete data lives only in the external store). Owned by the runtime
# user so the rest of the root fs can be mounted read-only (`read_only: true` in compose).
# /var/lib/wattwise is the default data root (object store; SQLite DB when so configured) — created
# OWNED BY the runtime UID so a NAMED VOLUME mounted there inherits correct ownership on first use
# (named volumes copy the image's ownership; bind mounts do not — see the README).
RUN install -d -o 10001 -g 10001 -m 0750 /var/run/wattwise \
    && install -d -o 10001 -g 10001 -m 0750 /var/lib/wattwise

WORKDIR /app

# Ship the versioned migrations + a runtime alembic config IN the image so a fresh container can
# bring its own database to head with NO source checkout on the host (issue #19; RUN-R6 stays the
# readiness gate, this makes first boot self-sufficient). Root-owned, read-only to the runtime user,
# like the venv. The entrypoint runs `alembic upgrade head` before serving unless
# WATTWISE_MIGRATE_ON_START is falsy (alembic is already a runtime dependency of the wheel — the
# entrypoint adds NO packages).
COPY --chown=root:root migrations /app/migrations
COPY --chown=root:root docker/alembic.ini /app/alembic.ini
COPY --chown=root:root --chmod=0755 docker/entrypoint.sh /usr/local/bin/wattwise-entrypoint

USER 10001:10001

EXPOSE 8000

# Liveness healthcheck (OBS-R6.1): probes the liveness endpoint only — process up + event loop
# responsive — and MUST NOT depend on external deps (a DB blip must not flap the container). Uses the
# venv python (no curl in the slim base, and we keep the image minimal). `--spider`-style HEAD via
# urllib; exit 1 on any non-200 so Docker/compose marks the container unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request; p=os.environ.get('WATTWISE_API__PORT','8000'); req=urllib.request.Request(f'http://127.0.0.1:{p}/healthz', method='GET'); sys.exit(0 if urllib.request.urlopen(req, timeout=4).status==200 else 1)"]

# Boot via the migrate-then-serve entrypoint (exec form, non-root): apply schema migrations against
# WATTWISE_DATABASE_DSN (WATTWISE_MIGRATE_ON_START, default on; a migration failure aborts the boot,
# fail-closed), then `exec` uvicorn as PID 1 (clean signal handling / fast SIGTERM) with the CMD args
# passed through. The app is exposed as a factory `create_app()`; `--factory` calls it.
# Host/port come from config (WATTWISE_API__*, RUN-R4) with the env defaults set above.
ENTRYPOINT ["wattwise-entrypoint"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
