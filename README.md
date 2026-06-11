<div align="center">

# wattwise-core

### Self-hosted endurance-training analytics — with an AI coach that refuses to make things up.

Your power meter never lies. Your analytics platform shouldn't either.

[![CI](https://img.shields.io/github/actions/workflow/status/bepcyc/wattwise-core/ci.yml?branch=main&logo=github&label=CI)](https://github.com/bepcyc/wattwise-core/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Typed: mypy --strict](https://img.shields.io/badge/typed-mypy%20--strict-2A6DB2.svg)](https://mypy-lang.org/)

**[Quickstart](#quickstart)** · **[Why](#why-wattwise)** · **[Features](#features)** · **[API](#api)** · **[FAQ](#faq)** · **[Roadmap](#roadmap)**

</div>

---

An illustration of the design goal — the coach answers from *your* data or not at all:

```text
You   ▸ How fried am I? Can I race on Sunday?

Coach ▸ Your CTL climbed 58 → 71 over the last six weeks; TSB is −4 today.
        That ramp is aggressive, but you absorbed it — no monotony spikes,
        HRV trend stable.

        Whether you can *race* Sunday, I won't answer: there is no event,
        taper, or race priority anywhere in your canonical record, and I
        don't guess. Tell me about the race and I'll do the math.
```

That refusal is the point. The agent runs behind a **fail-closed grounding gate**: every
number it wants to say is verified against your canonical training record by deterministic
code, and anything unverifiable is scrubbed before you see it. Refusing beats inventing.

It also checks whether your data is **fresh enough** to answer, not just whether a number is
correct. If a connector quietly stops syncing, your recent training simply stops arriving — and
a naive coach reads that gap as rest and tells you you're fresh. wattwise won't: when your latest
data is stale and a source looks disconnected, it says so and holds back the call instead of
cheering you into a hard day on tired legs. A real taper (your sync is fine, you just rested) is
trusted as usual.

## Why wattwise?

- **Your data lives in someone else's silo.** Years of FIT files behind a login you don't
  control, an API whose terms can change any season, a subscription that quietly creeps up.
  wattwise-core unifies your sources into **one canonical record of truth** on **your**
  server, under an Apache-2.0 license.
- **Most "AI coaches" hallucinate.** A language model will happily invent a CTL value with
  total confidence. Here the model never self-certifies: it proposes claims, and
  deterministic code verifies each one against canonical data before it reaches you.
  Unverifiable numbers are scrubbed — when in doubt, the agent abstains.
  And checking goes beyond "does this number exist somewhere in your data": each sentence
  is verified to mean what your record says. The words you read pick the metric and the
  date the number is checked against — so "your fatigue is 71" can never be quietly
  "verified" against your fitness, and a stale value can never be passed off as today's.
  When the wires get crossed, the agent doesn't just delete the number — it puts the true
  one in its place.
- **Sports-science metrics done carefully.** PMC, NP/IF/TSS, CP/W′, W′bal, aerobic
  decoupling, HRV, TRIMP — computed from the canonical record, with an offline evaluation
  suite guarding the agent's grounding and abstention behavior.

## Features

- **Source-agnostic canonical model** — every source flows through pluggable adapters into
  one unified record; cross-source duplicates and field conflicts are resolved by an
  explicit, configurable trust policy.
- **Performance analytics** — PMC (CTL/ATL/TSB), Normalized Power, Intensity Factor, TSS,
  Critical Power and W′, W′bal, aerobic decoupling, HRV (time- and frequency-domain), TRIMP.
- **Trustworthy coaching agent** — LangGraph-based, with **fail-closed grounding**: claims
  are verified against canonical data in deterministic code; no fabrication, explicit
  abstention.
- **Data-freshness aware** — the readiness call checks whether your record is recent enough to
  trust: a stale record behind a broken/stalled sync never earns a "go hard" verdict; it abstains
  and tells you to check your connection. A genuine taper still reads as fresh.
- **Multi-format upload** — FIT, FIT.GZ, GPX, TCX, PWX (including Strava bulk exports).
- **intervals.icu connector** — direct sync over HTTP Basic auth (API key).
- **REST API** — a single versioned `/v1` surface, OpenAPI 3.1, SSE streaming for agent
  answers.
- **Bearer auth** — token authentication with scope-based authorization.
- **Database portability** — **SQLite** (default), **PostgreSQL**, or **MariaDB**, switched
  by DSN only; no code changes, no vendor-specific SQL outside one audited seam.
- **Hardened by design** — prompt-injection isolation, server-derived identity end to end,
  fail-closed boot, and an offline agent eval suite.
- **Binding-faithful answers** — the sentence you read selects what it is checked against:
  metric wording and dates in the prose pick the canonical cell, mis-attributed figures are
  corrected in place, and an optional self-hosted fact-checking model
  ([MiniCheck](https://arxiv.org/abs/2404.10774)) re-reads every claim sentence against your
  data — with statistically calibrated strictness when you provide a calibration set.
- **Self-hosted, single-athlete** — one instance per human; a radically simpler trust and
  data model.

## Quickstart

```sh
git clone https://github.com/bepcyc/wattwise-core.git
cd wattwise-core
just bootstrap          # uv sync + database migration
just lint type test     # fast dev loop (full CI-parity gate: just gate)
```

Then point your LLM provider (any OpenAI-compatible endpoint) via configuration and start
the API. See [Configuration](#configuration). Prefer containers? The image is fully
self-sufficient — see [Run in a container](#run-in-a-container).

## Run in a container

Everything the engine needs ships in the image — including its database migrations. On
**first boot the container migrates its own database** to the latest schema before
serving, so you don't need a source checkout, `just`, or a manual `alembic` step. (Set
`WATTWISE_MIGRATE_ON_START=0` to manage migrations yourself; the readiness probe at
`/readyz` will then refuse to serve until the schema is up to date.)

New to self-hosting? [docs/CONFIGURATION.md](docs/CONFIGURATION.md) walks you through every setting task by task.

```sh
docker build -t wattwise-core:local .    # or use a released image

docker run -d --name wattwise \
  -p 127.0.0.1:8000:8000 \
  -v wattwise_data:/var/lib/wattwise \
  -e WATTWISE_DATABASE_DSN='sqlite+aiosqlite:////var/lib/wattwise/wattwise.sqlite' \
  -e WATTWISE_ENCRYPTION_ROOT_KEY="$(python3 -c 'import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())')" \
  -e WATTWISE_TOKEN_SIGNING_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" \
  -e WATTWISE_LLM_API_KEY='sk-...' \
  wattwise-core:local
```

That's a complete single-container setup: SQLite database and uploaded originals live on
the `wattwise_data` named volume. For PostgreSQL, point `WATTWISE_DATABASE_DSN` at your
database instead (see [`deploy/compose.yaml`](deploy/compose.yaml) for a hardened
two-service composition with an isolated Postgres on a dedicated named volume) — first
boot migrates it the same way.

**Volume ownership.** The container runs as an unprivileged fixed user (UID 10001) and
never as root, so it cannot fix file permissions for you. Prefer a **named volume** (as
above): Docker initializes named volumes with the ownership baked into the image, so
they are writable out of the box. If you bind-mount a host directory instead, you own
the permissions yourself: `sudo chown 10001:10001 <dir>` on rootful Docker, or — under
rootless Docker/Podman, where host ownership must map into the container's user
namespace — `podman unshare chown 10001:10001 <dir>` (or the subuid that maps to 10001).

First requests, end to end — the owner secret for minting a token is the value you set
as `WATTWISE_TOKEN_SIGNING_KEY`:

```sh
BASE=http://127.0.0.1:8000

# 1. Mint an access token
TOKEN=$(curl -fsS -X POST "$BASE/v1/auth/token" \
  -H 'Content-Type: application/json' \
  -d "{\"owner_secret\":\"$WATTWISE_TOKEN_SIGNING_KEY\"}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
AUTH="Authorization: Bearer $TOKEN"

# 2. Upload a workout file (FIT / GPX / TCX / PWX) and sync it in
curl -fsS -X POST "$BASE/v1/imports" -H "$AUTH" -F file=@ride.fit
curl -fsS -X POST "$BASE/v1/sync/run" -H "$AUTH"

# 3. Read your activities and the performance-management chart
curl -fsS "$BASE/v1/activities" -H "$AUTH"
curl -fsS "$BASE/v1/performance/load-fitness?from=2024-01-01&to=2024-01-08" -H "$AUTH"

# 4. Ask the coach (Server-Sent Events stream; needs WATTWISE_LLM_API_KEY)
curl -N -X POST "$BASE/v1/agent/ask" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"question":"How much training load have I done recently?","stream":true}'
```

## Configuration

Configuration is **layered**: packaged defaults (`defaults.toml`) → optional operator file
(`WATTWISE_CONFIG_FILE`) → environment variables (`WATTWISE_*`). Nest per-section values
with the `__` delimiter (e.g. `WATTWISE_API__PORT`).

**Secrets come from the environment only** — there is no `.env` reading at runtime:

- `WATTWISE_DATABASE_DSN`
- `WATTWISE_ENCRYPTION_ROOT_KEY`
- `WATTWISE_TOKEN_SIGNING_KEY`
- `WATTWISE_LLM_API_KEY` *(optional at boot — needed for the coaching agent)*

Boot is **fail-closed**: a missing required secret aborts startup in staging and production.

Tunable without code changes:

- Analytics time constants (`CTL=42d`, `ATL=7d` defaults).
- LLM provider (`base_url`, `model`, `temperature`, `max_output_tokens`,
  `grounding_min_coverage`) — any OpenAI-compatible endpoint, including local ones.
- Rate limiting (defaults: **120 read · 30 mutating · 20 agent req/min**) and upload
  ceiling (default **32 MiB**).
- Object store (local filesystem or S3-compatible) and retention window (days or
  indefinite).

## Import sources

| Source | Format / Auth | Notes |
| --- | --- | --- |
| File upload | FIT / FIT.GZ | Garmin native format (also gzip-compressed) |
| File upload | GPX | GPS exchange format |
| File upload | TCX | Training Center XML |
| File upload | PWX | TrainingPeaks / PeaksWare XML |
| Strava export | FIT / GPX / TCX | Uploaded as exported files |
| intervals.icu | HTTP Basic (API key) | Direct connector with on-demand sync |

## API

A single, versioned **`/v1`** surface — no scattered prefixes.

- **OpenAPI 3.1** at `GET /v1/openapi.json`; human-readable docs at `GET /v1/docs`.
- **Bearer auth** via `POST /v1/auth/token`, with scope-based access control: `read`,
  `write`, `agent`, `sync`, `export`, `admin`.
- **Analytics** — endpoints for PMC, Critical Power, W′bal, aerobic decoupling, HRV, TRIMP.
- **Agent** — `POST /v1/agent/ask` with SSE streaming (`text/event-stream`).
- **Activities** — history and per-activity detail.
- **Imports** — file uploads via `POST /v1/imports`.
- **Connections** — credential management for external sources.
- **Sync** — on-demand data synchronization.
- **Ops** — `GET /v1/system/status` for system status; `GET /healthz` liveness probe
  (process health only, no external dependencies).

## Architecture

![wattwise-core high-level architecture](./assets/img/wattwise-high-level-architecture.png)

Every data source flows through a pluggable adapter into one canonical record; analytics
are computed from that record; the coaching agent answers on top of it through the
fail-closed grounding gate. Storage is a single DSN — SQLite, PostgreSQL, or MariaDB.

## Tech stack

`Python 3.13` · `FastAPI` · `Uvicorn` · `SQLAlchemy 2.0+ (async)` · `Alembic` ·
`Pydantic + Pydantic Settings` · `LangGraph` · `OpenAI-compatible provider interface` ·
`NumPy + SciPy` · `Garmin FIT SDK + fitdecode + gpxpy + lxml` · `httpx` · `structlog` ·
`Docker (multi-stage hardened image)` · `uv`

## Data safety

wattwise-core is deliberately careful with your database:

- Creates and targets **only its own isolated schema**.
- **Never** connects to, reads, or modifies any pre-existing database.
- **Never** destroys or manages data volumes outside its schema.
- **Single-athlete scope** — one canonical store per deployment instance.
- Encryption root key and signing keys live **in the environment only** — never in code or
  images.
- **No PII on disk** except in the configured object store (original file retention).
- **Structured logging to stdout/stderr only** — never to log files — with central log
  redaction enforced.

## FAQ

**Can I keep my data 100% local — including the LLM?**
Yes. The agent talks to any OpenAI-compatible endpoint: point `base_url` at Ollama, vLLM,
llama.cpp-server, whatever you run. Your watts never have to leave your LAN.

**Why single-athlete? I want to host my whole club.**
Scope is a feature. Single-athlete means a radically simpler trust and data model — no
tenant-isolation bugs, no "oops, you saw my FTP". Instances are cheap; run one per human.

**Garmin Connect / Wahoo direct sync?**
Not yet — file uploads and intervals.icu cover most flows today, and direct connectors are
on the [roadmap](#roadmap). Adapters are pluggable; PRs welcome.

**Is this medical or coaching advice?**
No. wattwise computes and explains *your* numbers. Decisions about training, health, and
racing belong to you (and your human coach).

## Roadmap

- [x] Canonical model + FIT / FIT.GZ / GPX / TCX / PWX ingestion
- [x] PMC, NP/IF/TSS, CP/W′, W′bal, aerobic decoupling, HRV, TRIMP
- [x] Fail-closed grounded coaching agent (LangGraph, SSE streaming)
- [x] intervals.icu direct connector
- [x] Database portability: SQLite / PostgreSQL / MariaDB
- [ ] More direct connectors (Garmin Connect, Wahoo)
- [ ] Deeper running & swimming metrics
- [ ] Structured plan review & taper analysis in the agent
- [ ] First-party web UI

Vote with 👍 on issues — popularity genuinely moves things up the list.

## Contributing

The dev loop is `just bootstrap` → `just lint type test` (`just gate` for the full
pre-merge gate CI runs). Sports scientists, Python engineers, and people who are personally offended by
wrong TSS implementations are all equally welcome. Found a bug in the math? That's a
**high-priority issue** — open it. Start at [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

[Apache-2.0](./LICENSE) — self-host it, fork it, build on it. Just keep the notices.

---

<div align="center">
<sub>Built by athletes who read the papers. <b>Your watts. Your server. Your truth.</b></sub>
</div>
