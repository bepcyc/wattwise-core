# Configuration

This guide is for the person running wattwise for themselves — a runner or cyclist
who wants to start the container, point it at their data, pick a coaching model, and
keep it running. You do not need to read the source. Every command and example below
was run against a freshly built container before this page was published.

- [How configuration works](#how-configuration-works)
- [Run it](#run-it)
- [Connect your data](#connect-your-data)
- [Pick your model](#pick-your-model)
- [Tune the coach](#tune-the-coach)
- [Operate it](#operate-it)
- [When boot refuses to start](#when-boot-refuses-to-start)
- [Reference: every setting](#reference-every-setting)

## How configuration works

Settings come from three layers. Each layer overrides the one before it:

1. **Packaged defaults** — sensible values ship inside the image. Most people never
   change these.
2. **An optional operator file** — a TOML file you point at with
   `WATTWISE_CONFIG_FILE`. Good for a long list of overrides you want to keep in one
   place.
3. **Environment variables** — anything prefixed with `WATTWISE_`. These win over the
   first two, and they are how you pass secrets.

**Naming.** Every setting has a section and a key. Join them with a double underscore
and add the `WATTWISE_` prefix. For example, the `port` key in the `[api]` section is
`WATTWISE_API__PORT`. A deeper key like the intervals.icu retry budget is
`WATTWISE_ADAPTERS__INTERVALS_ICU__BUDGET_MAX_ATTEMPTS`. The same `section__key` form
is used in the operator TOML file (as normal nested tables) and as the environment
variable suffix.

**Keep secrets out of files.** Pass the database connection string, the encryption
key, the token signing key, and the LLM key via the environment (or a secret manager).
The packaged defaults carry no secret values and nothing is baked into the image — but
the loader does not strip secret keys from an operator file, so a secret you put there
WILL be honored when the matching environment variable is unset. Treat that as a
footgun, not a feature: environment only.

**Boot fails closed.** If a required secret is missing, or a value is out of range
(say, a port above 65535), the container refuses to start and prints a clear error
instead of running in a broken or insecure state. See
[When boot refuses to start](#when-boot-refuses-to-start).

## Run it

You need three things to boot in the default `production` environment:

- a **database connection string** (`WATTWISE_DATABASE_DSN`),
- an **encryption root key** (`WATTWISE_ENCRYPTION_ROOT_KEY`),
- a **token signing key** (`WATTWISE_TOKEN_SIGNING_KEY`).

For local file-only evaluation, `development` can run without the encryption root
key. In that mode FIT/GPX/TCX upload and local analytics still work, but API-key
connectors are disabled until `WATTWISE_ENCRYPTION_ROOT_KEY` is configured. The
service never stores connector credentials in plaintext or under an ephemeral key.

Generate the two keys once and keep them safe — rotating them invalidates stored
encrypted data and issued tokens. The commands below create fresh values:

```sh
# A 32-byte encryption root key, base64-encoded:
python3 -c 'import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())'

# A 32-byte token signing key, hex-encoded:
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

Start the container (SQLite on a named volume — the simplest setup):

```sh
docker run -d --name wattwise \
  -p 127.0.0.1:8000:8000 \
  -v wattwise_data:/var/lib/wattwise \
  -e WATTWISE_DATABASE_DSN='sqlite+aiosqlite:////var/lib/wattwise/wattwise.sqlite' \
  -e WATTWISE_ENCRYPTION_ROOT_KEY='<the base64 value you generated>' \
  -e WATTWISE_TOKEN_SIGNING_KEY='<the hex value you generated>' \
  wattwise-core:local
```

Prefer a **named volume** as above: Docker initializes named volumes with the ownership
baked into the image, so they are writable out of the box. The container runs as an
unprivileged fixed user (UID 10001), never root, so it cannot fix file permissions for
you — if you bind-mount a host directory instead, set the ownership yourself:
`sudo chown 10001:10001 <dir>` on rootful Docker, or under rootless Docker/Podman
`podman unshare chown 10001:10001 <dir>` (the subuid that maps to 10001).

The container migrates its own database on first boot, then starts serving. Check that
it is alive and ready:

```sh
curl http://127.0.0.1:8000/healthz
# {"status":"alive"}

curl http://127.0.0.1:8000/readyz
# {"status":"ready", ...}
```

`/healthz` says the process is up. `/readyz` says it can actually serve — database
reachable, schema migrated, configuration valid. Wait for `/readyz` to return
`"ready"` before sending real requests.

Now mint an access token. The owner secret you present is the same value you set as
`WATTWISE_TOKEN_SIGNING_KEY`:

```sh
curl -X POST http://127.0.0.1:8000/v1/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"owner_secret":"<your WATTWISE_TOKEN_SIGNING_KEY>"}'
# {"access_token":"...","token_type":"bearer","expires_in":3600, ...}
```

Use the returned `access_token` as a bearer token on every other request:
`-H "Authorization: Bearer <access_token>"`. The token lasts one hour by default
(`expires_in: 3600`); mint a new one or use the refresh token when it expires.

### Use PostgreSQL instead of SQLite

Point the connection string at your database. PostgreSQL is recommended for real use:

```sh
-e WATTWISE_DATABASE_DSN='postgresql+asyncpg://USER:PASSWORD@HOST:5432/wattwise'
```

The connection string shape matters. SQLite uses the `sqlite+aiosqlite://` driver and
PostgreSQL uses `postgresql+asyncpg://`. A connection string with the wrong driver, a
bad password, or an unreachable host stops the first-boot migration and the container
exits with an error — fix the string and restart.

## Connect your data

There are two ways to get activities in.

**Upload files.** Send FIT, FIT.GZ, GPX, or TCX files to the import endpoint (anything
else is refused with `415`). The upload itself already hands the file to the processor;
the follow-up sync run picks up anything still pending and refreshes the canonical
surfaces, so it is a safe "make sure everything landed" step, not a required activation:

```sh
AUTH="Authorization: Bearer <your access token>"
curl -X POST http://127.0.0.1:8000/v1/imports -H "$AUTH" -F file=@ride.fit
curl -X POST http://127.0.0.1:8000/v1/sync/run -H "$AUTH"
```

**Connect intervals.icu.** This is done at runtime through the API, not through
configuration — your intervals.icu API key is stored as a per-athlete credential, not
an environment variable. List the available connectors, then initiate and complete the
connection with your key:

```sh
curl http://127.0.0.1:8000/v1/connections -H "$AUTH"
```

Follow the steps the connector returns to supply your intervals.icu API key. Once
connected, syncs pull your history on demand.

If you started in development without `WATTWISE_ENCRYPTION_ROOT_KEY`, completing an
API-key connector returns `credential-storage-disabled`. Restart with a real
encryption root key to enable encrypted credential storage; file uploads remain
available without it.

You do not normally touch any configuration for this. If intervals.icu rate-limits
your account or you want to pace requests differently, the connector's retry and
request-rate behavior is tunable under
`WATTWISE_ADAPTERS__INTERVALS_ICU__*` (see the
[reference](#data-sources-and-syncing) below) — most people leave these alone.

### Privacy: raw GPS

Raw GPS coordinates are stored by default. To keep coordinates out of storage while
still computing every non-locating metric, set:

```sh
-e WATTWISE_PRIVACY__STORE_RAW_GPS=false
```

## Pick your model

The coach talks to **any OpenAI-compatible chat-completions endpoint**. You set three
things: where to send requests, which model, and the key.

```sh
-e WATTWISE_AGENT__BASE_URL='https://your-provider.example/api/v1' \
-e WATTWISE_AGENT__MODEL='your-model-id' \
-e WATTWISE_LLM_API_KEY='sk-...'      # your provider key — keep it secret
```

The LLM key is **optional at boot**: the container starts and serves your data
without it. It is required only for the coaching agent — without it, asking the coach
a question fails, but uploads, syncs, and analytics all work.

A few model settings worth knowing:

- `WATTWISE_AGENT__MAX_OUTPUT_TOKENS` (default `8192`) is the answer budget. Reasoning
  models spend tokens thinking before they answer, and that thinking is billed against
  this same budget — so a small value can leave no room for the actual answer. Keep it
  generous.
- `WATTWISE_AGENT__CONTEXT_WINDOW_TOKENS` (default `131072`) must match your model's
  real context window. If you swap to a smaller model, lower this so the engine does
  not overrun it.
- `WATTWISE_AGENT__TEMPERATURE` (default `0.0`) keeps answers steady and repeatable.
- `WATTWISE_AGENT__REQUEST_TIMEOUT_SECONDS` (default `60`) caps how long a single
  model call may take.

If you want per-run cost reporting to be accurate, set your provider's real prices in
`WATTWISE_AGENT__COST__INPUT_PER_MILLION_USD` and
`WATTWISE_AGENT__COST__OUTPUT_PER_MILLION_USD` (USD per million tokens). A local or
free model can leave these at `0`.

## Tune the coach

The coach is grounded: it answers only from your canonical training data and refuses
to invent numbers. The defaults are good. A few knobs matter to a self-hoster:

- **Answer language.** The coach answers in English, German, or Russian out of the box,
  and will attempt other languages on request. Language is chosen per request, not by a
  global setting.
- **Reply length and budget.** The agent run is bounded so a single question cannot run
  away. The bounds live under `WATTWISE_ENTITLEMENT__*`:
  - `NODE_VISIT_CEILING` (default `60`) — the maximum reasoning steps per run.
  - `MAX_OUTPUT_TOKENS` (default `8192`) — the per-run output ceiling enforced by the
    run guard. (This is separate from `WATTWISE_AGENT__MAX_OUTPUT_TOKENS`, which is the
    value sent to the model.)
  - `WALL_CLOCK_SECONDS` (default `120`) — the wall-clock limit for one run.
  - `MAX_TOOL_ITERATIONS` (default `16`) — the cap on data-fetch loops.
  You can raise these if runs feel cut short. They appear in the `/readyz` `plan` block,
  so you can confirm an override took effect.
- **Grounding strictness.** `WATTWISE_AGENT__GROUNDING_MIN_COVERAGE` (default `1.0`)
  requires every claim to be backed by your data. Lowering it loosens that guarantee;
  most people should not.
- **Provider redaction.** `WATTWISE_AGENT__REDACT_PROVIDER_PAYLOADS` (default `true`)
  masks personal data before it is sent to your model provider. Leave it on.

The coach's persona, prompts, and metric vocabulary are all configuration rather than
code, so they can be replaced wholesale — but doing so is an advanced customization,
not a normal setup step.

## Operate it

**Migrations.** By default the container brings its own database up to date on boot.
If you manage migrations yourself, set `WATTWISE_MIGRATE_ON_START=0`. The container
will then **not** migrate, and `/readyz` will refuse to serve (returning HTTP 503 with
`"migrations_applied": false`) until the schema is current. This is intentional — it
will not serve over a half-migrated database.

**Logging.** Set `WATTWISE_APP__LOG_LEVEL` to `DEBUG`, `INFO` (default), `WARNING`, or
`ERROR`.

**Network.** The API binds inside the container on `WATTWISE_API__HOST` /
`WATTWISE_API__PORT` (default `0.0.0.0:8000`). You usually leave these alone and map a
host port with Docker's `-p`. Put the service behind your own TLS-terminating reverse
proxy rather than exposing it directly.

**Allowed hosts and CORS.** For a real deployment, narrow `WATTWISE_SECURITY__ALLOWED_HOSTS`
from the permissive default `*` to your actual hostname(s), and set
`WATTWISE_SECURITY__CORS_ALLOW_ORIGINS` to the origins your web client is served from.
A wildcard origin combined with `cors_allow_credentials = true` is rejected at boot —
the two cannot be used together.

**Upload size.** `WATTWISE_API__REQUEST_MAX_BYTES` (default `33554432`, i.e. 32 MiB)
caps upload size. Raise it if your activity files are larger.

**Rate limits.** Per-minute request ceilings are
`WATTWISE_RATELIMIT__READ_PER_MINUTE` (default `120`),
`WATTWISE_RATELIMIT__MUTATING_PER_MINUTE` (default `30`), and the agent ceiling
`WATTWISE_ENTITLEMENT__REQUEST_RATE_PER_MINUTE` (default `20`).

**Retention.** `WATTWISE_RETENTION__RAW_FILE_DAYS` and
`WATTWISE_RETENTION__AGENT_STATE_DAYS` both default to `0`, meaning keep forever. Set a
positive number of days to expire old uploaded files or old agent conversation state.

**Object storage.** Uploaded original files live on local disk by default
(`WATTWISE_OBJECT_STORE__KIND=local`, under
`WATTWISE_OBJECT_STORE__LOCAL_ROOT`). To use S3-compatible storage instead, set
`WATTWISE_OBJECT_STORE__KIND=s3` and provide both
`WATTWISE_OBJECT_STORE__S3_ENDPOINT` and `WATTWISE_OBJECT_STORE__S3_BUCKET` — boot
fails if either is missing.

## When boot refuses to start

The service fails closed: a bad configuration aborts startup with a clear message
rather than running broken. These are the messages you are most likely to see.

**A required secret is missing.** In `production` and `staging`, the encryption and
signing keys are required:

```
ConfigError: fail-closed: required configuration is missing: WATTWISE_TOKEN_SIGNING_KEY
(must be provided via the environment / a secret manager; BOOT-R4)
```

The database connection string is required in **every** environment, including
`development`:

```
ConfigError: fail-closed: required configuration is missing: WATTWISE_DATABASE_DSN ...
```

Fix: set the named variable and restart.

**The signing key is too weak.** A key with under 32 bytes of material, or one built
from a tiny set of repeated characters, is rejected:

```
ConfigError: fail-closed: WATTWISE_TOKEN_SIGNING_KEY carries insufficient entropy
(needs >= 32 bytes / 256 bits, got 5); the service refuses to start with a weak signing key
```

Fix: generate a real key with the command in [Run it](#run-it).

**A value is out of range.** For example, an invalid port:

```
ValidationError: 1 validation error for Settings
api__port
  Input should be less than or equal to 65535 ...
```

Fix: correct the value to be within the range shown in the
[reference table](#reference-every-setting).

**The database is unreachable or the migration failed.** On first boot the container
runs migrations before serving. If the connection string is wrong or the database is
down, it aborts loudly:

```
[wattwise] FATAL: schema migration failed — refusing to start.
Fix WATTWISE_DATABASE_DSN / the database and restart, or set
WATTWISE_MIGRATE_ON_START=0 to manage migrations yourself.
```

Fix: correct `WATTWISE_DATABASE_DSN`, make sure the database is up, and restart.

> **Tip for development only.** Setting `WATTWISE_APP__ENVIRONMENT=development` relaxes
> production secret requirements. The service can boot with only a database connection
> string, but credential-backed connectors stay disabled until
> `WATTWISE_ENCRYPTION_ROOT_KEY` is configured. Never run a real deployment in
> `development`.

## Reference: every setting

Every setting below is read from the layers described in
[How configuration works](#how-configuration-works). The **Variable** column is the
environment-variable name; for the operator file, drop the `WATTWISE_` prefix and write
it as a nested TOML table. **Default** is the packaged value. **Limits** are the
validation bounds — a value outside them fails boot.

### Application

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_APP__ENVIRONMENT` | `production` | `production`, `staging`, or `development` | Which environment the service runs as; `development` relaxes secret requirements. |
| `WATTWISE_APP__LOG_LEVEL` | `INFO` | — | Log verbosity (`DEBUG`/`INFO`/`WARNING`/`ERROR`). |

### Required secrets (environment only)

| Variable | Default | Effect |
| --- | --- | --- |
| `WATTWISE_DATABASE_DSN` | — (required, every environment) | The database connection string. SQLite (`sqlite+aiosqlite://...`) or PostgreSQL (`postgresql+asyncpg://...`). |
| `WATTWISE_ENCRYPTION_ROOT_KEY` | — (required in staging/production) | Root key protecting stored secrets at rest. Generate once; rotating it invalidates encrypted data. |
| `WATTWISE_TOKEN_SIGNING_KEY` | — (required in staging/production) | Signs access tokens, and is the owner secret you present to mint one. Must carry at least 32 bytes of real entropy. |
| `WATTWISE_LLM_API_KEY` | — (optional) | Your model provider's API key. Optional at boot; required only to use the coach. |

These optional secrets let you split database roles or add a service-to-service factor;
single-host setups leave them unset:

| Variable | Default | Effect |
| --- | --- | --- |
| `WATTWISE_DATABASE_MASTER_DATA_DSN` | falls back to `DATABASE_DSN` | Separate credential for athlete-authored master data. |
| `WATTWISE_DATABASE_READ_DSN` | falls back to `DATABASE_DSN` | Separate read-only credential for analytics reads. |
| `WATTWISE_AGENT_STATE_DSN` | falls back to `DATABASE_DSN` | Separate credential for agent conversation state. |
| `WATTWISE_SECURITY__SERVICE_AUTH_SECRET` | unset | Shared secret a first-party service presents in addition to the user token; unset means that check is disabled. |

### Container operation

These are read by the container entrypoint and the web server, not the settings file.

| Variable | Default | Effect |
| --- | --- | --- |
| `WATTWISE_MIGRATE_ON_START` | `1` (on) | Migrate the database to the latest schema before serving. Set `0` to skip; `/readyz` then refuses until you migrate yourself. |
| `WATTWISE_CONFIG_FILE` | unset | Path to an optional operator TOML file (the middle config layer). Boot fails if the path is set but the file is missing. |

### Network and API

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_API__HOST` | `0.0.0.0` | — | Interface the server binds inside the container. |
| `WATTWISE_API__PORT` | `8000` | 1–65535 | Port the server binds inside the container. |
| `WATTWISE_API__RATE_LIMIT_PER_MINUTE` | `60` | ≥ 1 | Baseline per-minute request ceiling. |
| `WATTWISE_API__REQUEST_MAX_BYTES` | `33554432` | ≥ 1 | Maximum upload/request body size in bytes (32 MiB). |

### Authentication and tokens

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_AUTH__ACCESS_TTL_SECONDS` | `3600` | 1–3600 | Access-token lifetime in seconds (at most one hour). |
| `WATTWISE_AUTH__REFRESH_TTL_SECONDS` | `1209600` | ≥ 60 | Refresh-token family lifetime (14 days). |
| `WATTWISE_AUTH__LINK_TTL_SECONDS` | `600` | ≥ 30 | Single-use account-link challenge lifetime. |
| `WATTWISE_EXPORTS__SIGNED_URL_TTL_SECONDS` | `300` | 1–300 | Lifetime of a signed export-download URL (at most five minutes). |

### Security headers, CORS, and allowed hosts

| Variable | Default | Effect |
| --- | --- | --- |
| `WATTWISE_SECURITY__ALLOWED_HOSTS` | `["*"]` | Hostnames the service accepts. Narrow to your real host in production. |
| `WATTWISE_SECURITY__CORS_ALLOW_ORIGINS` | `["http://localhost:5173", "http://127.0.0.1:5173"]` | Browser origins allowed to call the API. |
| `WATTWISE_SECURITY__CORS_ALLOW_CREDENTIALS` | `true` | Whether credentialed cross-origin requests are allowed. Cannot be `true` while origins include `*`. |
| `WATTWISE_SECURITY__CORS_ALLOW_METHODS` | `["GET","POST","PUT","PATCH","DELETE","OPTIONS"]` | HTTP methods allowed cross-origin. |
| `WATTWISE_SECURITY__CORS_ALLOW_HEADERS` | `["Authorization","Content-Type","Last-Event-ID"]` | Request headers allowed cross-origin. |
| `WATTWISE_SECURITY__HSTS_MAX_AGE_SECONDS` | `31536000` | HSTS max-age in seconds (one year). |
| `WATTWISE_SECURITY__REFERRER_POLICY` | `no-referrer` | Value of the `Referrer-Policy` response header. |
| `WATTWISE_SECURITY__CONTENT_SECURITY_POLICY` | `default-src 'none'; ...` | Content-Security-Policy for any HTML surface. |

### Privacy and retention

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_PRIVACY__STORE_RAW_GPS` | `true` | — | Whether to store raw GPS coordinates; `false` keeps coordinates out while keeping derived metrics. |
| `WATTWISE_RETENTION__RAW_FILE_DAYS` | `0` | ≥ 0 | Days to keep uploaded original files; `0` means keep forever. |
| `WATTWISE_RETENTION__AGENT_STATE_DAYS` | `0` | ≥ 0 | Days to keep agent conversation state; `0` means keep forever. |

### Database connection pool

Applies to PostgreSQL/MariaDB; SQLite ignores these.

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_DATABASE__POOL_SIZE` | `5` | ≥ 1 | Number of pooled connections kept open. |
| `WATTWISE_DATABASE__MAX_OVERFLOW` | `10` | ≥ 0 | Extra connections allowed beyond the pool under load. |
| `WATTWISE_DATABASE__POOL_TIMEOUT_S` | `30.0` | > 0 | Seconds to wait for a connection before erroring. |
| `WATTWISE_DATABASE__POOL_RECYCLE_S` | `1800` | ≥ -1 | Recycle connections older than this many seconds; `-1` disables. |

### Object storage

| Variable | Default | Effect |
| --- | --- | --- |
| `WATTWISE_OBJECT_STORE__KIND` | `local` | `local` (filesystem) or `s3` (S3-compatible). |
| `WATTWISE_OBJECT_STORE__LOCAL_ROOT` | `/var/lib/wattwise/objects` | Directory for stored originals when `kind=local`. |
| `WATTWISE_OBJECT_STORE__S3_ENDPOINT` | unset | S3 endpoint URL; required when `kind=s3`. |
| `WATTWISE_OBJECT_STORE__S3_BUCKET` | unset | S3 bucket name; required when `kind=s3`. |

### Rate limiting and agent budget

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_RATELIMIT__READ_PER_MINUTE` | `120` | ≥ 1 | Per-minute ceiling for read requests. |
| `WATTWISE_RATELIMIT__MUTATING_PER_MINUTE` | `30` | ≥ 1 | Per-minute ceiling for write requests. |
| `WATTWISE_ENTITLEMENT__REQUEST_RATE_PER_MINUTE` | `20` | ≥ 1 | Per-minute ceiling for agent requests. |
| `WATTWISE_ENTITLEMENT__NODE_VISIT_CEILING` | `60` | ≥ 1 | Maximum reasoning steps per agent run. |
| `WATTWISE_ENTITLEMENT__MAX_OUTPUT_TOKENS` | `8192` | ≥ 1 | Output-token ceiling enforced on an agent run. |
| `WATTWISE_ENTITLEMENT__WALL_CLOCK_SECONDS` | `120` | > 0 | Wall-clock limit for one agent run. |
| `WATTWISE_ENTITLEMENT__MAX_TOOL_ITERATIONS` | `16` | ≥ 1 | Maximum data-fetch loops per agent run. |

### Ingestion and syncing

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_INGESTION__BATCH_SIZE` | `500` | ≥ 1 | Rows written per database round-trip during import. |
| `WATTWISE_INGESTION__SYNC_CONCURRENCY` | `4` | ≥ 1 | How many sources sync at once (clamped to 1 on SQLite). |
| `WATTWISE_INGESTION__BACKFILL_WINDOW_DAYS` | `90` | ≥ 1 | Size in days of each historical backfill window. |

### Data sources and syncing

The intervals.icu connector's resilience. Most people never change these; adjust them
only if your account is being rate-limited.

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__BUDGET_MAX_ATTEMPTS` | `4` | ≥ 1 | Maximum retry attempts per request. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__BUDGET_MAX_ELAPSED_S` | `30.0` | > 0 | Maximum total wall time spent retrying one request. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__BUDGET_BASE_BACKOFF_S` | `0.5` | ≥ 0 | Initial backoff delay between retries. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__BUDGET_MAX_BACKOFF_S` | `8.0` | ≥ 0 | Cap on the backoff delay. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__BUCKET_RATE_PER_S` | `5.0` | > 0 | Steady request rate per second to the source. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__BUCKET_CAPACITY` | `10.0` | > 0 | Burst capacity of the request limiter. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__BUCKET_REDUCE_FACTOR` | `0.5` | 0 < x < 1 | How sharply the rate drops after a rate-limit response. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__BUCKET_MIN_RATE` | `0.5` | > 0 | Floor the adaptive rate never drops below. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__HTTP_TIMEOUT_S` | `30.0` | > 0 | Per-request connect-plus-read timeout. |
| `WATTWISE_ADAPTERS__INTERVALS_ICU__DISCOVER_PAGE_SIZE` | `200` | ≥ 1 | Page size when discovering activities to sync. |

### Analytics

These shape the training-analytics math. The defaults follow established
performance-management practice; change them only if you know what you are doing.

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_ANALYTICS__CTL_TIME_CONSTANT_DAYS` | `42` | > 0 | Fitness (chronic load) decay time constant in days. |
| `WATTWISE_ANALYTICS__ATL_TIME_CONSTANT_DAYS` | `7` | > 0 | Fatigue (acute load) decay time constant in days. |
| `WATTWISE_ANALYTICS__TRAINING_LOAD_CONFIDENCE_PENALTY` | `0.7` | 0 < x ≤ 1 | Confidence multiplier when a day's load came from a lower-fidelity source. |
| `WATTWISE_ANALYTICS__SIGNATURE_MIN_FIT_R2` | `0.85` | 0–1 | Minimum fit quality for a modeled fitness signature to be used. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_WEIGHT_CTL` | `0.4` | ≥ 0 | Relative weight of the fitness component in the endurance score. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_WEIGHT_DURABILITY` | `0.3` | ≥ 0 | Relative weight of the durability component. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_WEIGHT_DECOUPLING` | `0.3` | ≥ 0 | Relative weight of the decoupling component. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_CTL_FULL_SCALE` | `100.0` | > 0 | Fitness value at which the fitness component saturates. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_DURABILITY_FLOOR` | `0.5` | ≥ 0 | Durability ratio mapped to the bottom of its band. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_DURABILITY_CEILING` | `1.0` | > 0 | Durability ratio mapped to the top of its band. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_DECOUPLING_FULL_PENALTY_PCT` | `10.0` | > 0 | Aerobic-decoupling percent at which that component bottoms out. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_ALLOW_PARTIAL` | `true` | — | Whether to score on available components when some are missing. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_PARTIAL_CONFIDENCE_PENALTY` | `0.7` | 0 < x ≤ 1 | Confidence multiplier for a partial endurance score. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_WINDOW_DAYS` | `90` | ≥ 1 | Lookback window for the endurance-score inputs. |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_LONG_DURATION_S` | `1200` | ≥ 1 | Long duration (seconds) in the durability ratio (20 min). |
| `WATTWISE_ANALYTICS__ENDURANCE_SCORE_SHORT_DURATION_S` | `300` | ≥ 1 | Short duration (seconds) in the durability ratio (5 min). |
| `WATTWISE_ANALYTICS__CP_POWER_SPREAD_EPSILON` | `1e-06` | 0 < x < 1 | Minimum power spread below which a critical-power fit is refused. |

### Coaching model

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_AGENT__BASE_URL` | `https://openrouter.ai/api/v1` | — | OpenAI-compatible endpoint base URL. |
| `WATTWISE_AGENT__MODEL` | `deepseek/deepseek-v4-flash` | — | Model identifier to call. |
| `WATTWISE_AGENT__TIER` | `flash` | — | Label tagged on model spans for observability. |
| `WATTWISE_AGENT__REASONING_EFFORT` | `low` | — | Reasoning-effort label tagged on model spans. |
| `WATTWISE_AGENT__TEMPERATURE` | `0.0` | 0.0–2.0 | Sampling temperature. |
| `WATTWISE_AGENT__MAX_OUTPUT_TOKENS` | `8192` | ≥ 1 | Output-token budget sent to the model. |
| `WATTWISE_AGENT__CONTEXT_WINDOW_TOKENS` | `131072` | ≥ 1024 | Your model's context window; set to match the deployed model. |
| `WATTWISE_AGENT__REQUEST_TIMEOUT_SECONDS` | `60` | > 0 | Timeout for one model request. |
| `WATTWISE_AGENT__GROUNDING_MIN_COVERAGE` | `1.0` | 0.0–1.0 | Required fraction of claims backed by your data. |
| `WATTWISE_AGENT__REDACT_PROVIDER_PAYLOADS` | `true` | — | Mask personal data before it reaches the model provider. |
| `WATTWISE_AGENT__IDEMPOTENCY_DEDUP_WINDOW_SECONDS` | `60` | ≥ 0 | Window in which a resubmitted identical turn reuses the same run; `0` disables. |
| `WATTWISE_AGENT__ALLOWED_HOSTS` | `["wattwise.app","www.wattwise.app","docs.wattwise.app"]` | — | Hosts whose links the coach may keep in answers. |
| `WATTWISE_AGENT__COST__INPUT_PER_MILLION_USD` | `0.14` | ≥ 0 | Input-token price (USD per million) for cost reporting. |
| `WATTWISE_AGENT__COST__OUTPUT_PER_MILLION_USD` | `0.28` | ≥ 0 | Output-token price (USD per million) for cost reporting. |

### Coach behavior and grounding

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_AGENT__COACH__SYSTEM_PROMPT` | (the default coach persona) | — | The coach's system prompt. |
| `WATTWISE_AGENT__COACH__GROUNDING_REL_TOLERANCE` | `0.02` | ≥ 0 | Relative tolerance when matching a stated number to your data. |
| `WATTWISE_AGENT__COACH__GROUNDING_ABS_TOLERANCE` | `0.05` | ≥ 0 | Absolute tolerance when matching a stated number to your data. |
| `WATTWISE_AGENT__COACH__GROUNDING_DISPLAY_DECIMALS` | `1` | 0–6 | Decimal places when a verified value is shown. |
| `WATTWISE_AGENT__COACH__LATEST_LOOKBACK_DAYS` | `42` | ≥ 1 | Lookback window for resolving an undated "latest" claim. |
| `WATTWISE_AGENT__COACH__LANGUAGE_PASSTHROUGH` | `true` | — | Allow answering in a requested language with no built-in pack. |
| `WATTWISE_AGENT__COACH__LANGUAGE_PASSTHROUGH_DIRECTIVE` | (a templated directive) | — | The instruction used for pass-through languages. |
| `WATTWISE_AGENT__COACH__PROMPTS` | (table) | — | Named system-prompt fragments the coach composes. |
| `WATTWISE_AGENT__COACH__GROUNDING_RULES` | (table) | — | Named grounding/abstention policy texts. |
| `WATTWISE_AGENT__COACH__MANIFEST` | (table) | — | The coach bundle's identity and schema version. |
| `WATTWISE_AGENT__COACH__SKILLS` | (list) | — | The named, versioned coach skills. |
| `WATTWISE_AGENT__COACH__LANGUAGES` | (table) | — | Per-language prompt and abstain-copy packs. |
| `WATTWISE_AGENT__METRIC_ALIASES` | (table) | — | Maps natural metric phrasings to canonical metric keys. |

### Claim-binding and fact-checking

Advanced grounding controls. The deterministic binding layer is on by default; the
optional fact-checking model is off and needs extra setup to enable.

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_AGENT__BINDING__MODE` | `enforce` | `off`, `shadow`, or `enforce` | How the deterministic claim-binding guard runs. |
| `WATTWISE_AGENT__BINDING__PRESENT_DEIXIS` | (a list of present-tense words) | — | Words the temporal check treats as "now". |
| `WATTWISE_AGENT__BINDING__FRESHNESS_DAYS` | `1` | ≥ 0 | How many days behind a value may lag and still count as current. |
| `WATTWISE_AGENT__BINDING__REQUIRE_METRIC_LABEL` | `false` | — | Whether to scrub a number whose sentence names no metric (high over-scrub risk; off). |
| `WATTWISE_AGENT__ENTAILMENT__ENABLED` | `false` | — | Enable the optional local fact-checking model. |
| `WATTWISE_AGENT__ENTAILMENT__MODEL_ID` | `lytang/MiniCheck-RoBERTa-Large` | — | Checkpoint for the fact-checking model. |
| `WATTWISE_AGENT__ENTAILMENT__DEVICE` | `cpu` | — | Device the fact-checking model runs on. |
| `WATTWISE_AGENT__ENTAILMENT__THRESHOLD_NUMBER` | `0.5` | 0.0–1.0 | Support threshold for number-bearing sentences. |
| `WATTWISE_AGENT__ENTAILMENT__THRESHOLD_STATEMENT` | `0.5` | 0.0–1.0 | Support threshold for numberless sentences. |
| `WATTWISE_AGENT__ENTAILMENT__ALPHA` | `0.05` | 0 < x < 1 | Risk level when thresholds are derived from a calibration artifact. |
| `WATTWISE_AGENT__ENTAILMENT__CALIBRATION_PATH` | `` (empty) | — | Path to a calibration artifact; empty uses the fixed thresholds. |
| `WATTWISE_AGENT__ENTAILMENT__CALIBRATION_DATASET_VERSION` | `` (empty) | — | Optional exact pin on the calibration artifact's dataset version. |
| `WATTWISE_AGENT__ENTAILMENT__MAX_CHECKS` | `16` | ≥ 1 | Maximum fact-checks per answer. |

### Offline evaluation budgets

Used by the offline evaluation harness, not at request time.

| Variable | Default | Limits | Effect |
| --- | --- | --- | --- |
| `WATTWISE_AGENT__EVAL__MEDIAN_COST_USD` | `0.05` | > 0 | Median per-task cost the eval gate enforces. |
| `WATTWISE_AGENT__EVAL__P95_LATENCY_MS` | `30000.0` | > 0 | 95th-percentile latency the eval gate enforces. |
| `WATTWISE_AGENT__EVAL__COST_PER_1K_TOKENS_USD` | `0.0002` | > 0 | Price per 1,000 tokens used to cost a recorded eval run. |
