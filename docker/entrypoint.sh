#!/bin/sh
# wattwise-core container entrypoint — migrate-then-serve.
#
# A fresh container must be able to bring ITS OWN database to the newest schema with no
# source checkout on the host (issue #19). The image ships the versioned migrations +
# a runtime alembic config at /app, and this entrypoint applies them before serving:
#
#   WATTWISE_MIGRATE_ON_START (default ON): run `alembic upgrade head` against
#   WATTWISE_DATABASE_DSN before booting the API. A migration failure ABORTS the boot
#   loudly (fail-closed) — the process never starts serving over a half-migrated store.
#   Set it to 0/false/no/off to manage migrations yourself; readiness (RUN-R6) will then
#   refuse to serve until the database is migrated to head.
#
# The script is exec-form-ENTRYPOINT compatible: it runs as the unprivileged image user
# (UID 10001 — no root at runtime), passes the CMD args straight through to uvicorn, and
# `exec`s so uvicorn is PID 1 (clean SIGTERM handling, no shell left behind).
set -eu

case "${WATTWISE_MIGRATE_ON_START:-1}" in
    0 | false | FALSE | False | no | NO | No | off | OFF | Off)
        echo "[wattwise] WATTWISE_MIGRATE_ON_START is off — skipping schema migration" \
            "(readiness refuses to serve an unmigrated database)" >&2
        ;;
    *)
        echo "[wattwise] applying schema migrations (alembic upgrade head)" >&2
        if ! alembic -c /app/alembic.ini upgrade head; then
            echo "[wattwise] FATAL: schema migration failed — refusing to start." \
                "Fix WATTWISE_DATABASE_DSN / the database and restart, or set" \
                "WATTWISE_MIGRATE_ON_START=0 to manage migrations yourself." >&2
            exit 1
        fi
        ;;
esac

exec python -m uvicorn --factory wattwise_core.api.app:create_app "$@"
