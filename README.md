# wattwise-core

> Self-hostable endurance-training analytics engine with a trustworthy AI coaching agent.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Typed: mypy](https://img.shields.io/badge/typed-mypy-2A6DB2.svg)](https://mypy-lang.org/)

Connect your training-data sources and wattwise-core unifies everything into **one canonical record of truth**, then computes **sports-science-correct analytics** — PMC, NP/IF/TSS, CP/W', W'bal, aerobic decoupling, HRV, TRIMP. On top of that sits a **LangGraph coaching agent** that answers questions and reviews your training load, **grounded fail-closed** against your real data: if it can't back a number with your canonical record, it won't say it. Self-hostable, single-athlete, Apache-2.0.

## Features

- **Source-agnostic canonical model** — every source flows through pluggable adapters into one unified record.
- **Performance analytics** — PMC (Chronic Training Load), NP (Normalized Power), IF (Intensity Factor), TSS (Training Stress Score), CP (Critical Power), W' (W-prime), W'bal (W-prime balance), aerobic decoupling, and HRV (time- and frequency-domain metrics).
- **TRIMP** — Training Impulse for load quantification.
- **Trustworthy coaching agent** — LangGraph-based, with **fail-closed grounding** and no fabrication.
- **Direct typed ingestion** — direct API clients for data sources; **MCP only for agent tool-use**.
- **REST API** — OpenAPI specification plus a generated typed client.
- **Bearer auth** — token authentication with scope-based authorization.
- **Database portability** — **SQLite** (default), **PostgreSQL**, or **MariaDB** by DSN only.
- **Multi-format upload** — FIT, FIT.GZ, GPX, TCX (including Strava exports).
- **Intervals.icu connector** — direct, over HTTP Basic auth.
- **Hardened by design** — prompt-injection isolation and an offline agent eval suite.

## Quickstart

```sh
git clone https://github.com/wattwise/wattwise-core.git
cd wattwise-core
just bootstrap
just lint type test
just test-db-portable
```

## Configuration

Configuration is **layered**: packaged defaults (`defaults.toml`) → optional operator file (`WATTWISE_CONFIG_FILE`) → environment variables (`WATTWISE_*`). Nest per-section values with the `__` delimiter (e.g. `WATTWISE_API__PORT`).

**Secrets come from the environment only** — there is no `.env` reading at runtime:

- `WATTWISE_DATABASE_DSN`
- `WATTWISE_ENCRYPTION_ROOT_KEY`
- `WATTWISE_TOKEN_SIGNING_KEY`
- `WATTWISE_LLM_API_KEY`

Boot is **fail-closed**: a missing required secret aborts startup in staging and production.

Tunable without code changes:

- Analytics time constants (`CTL=42d`, `ATL=7d` defaults).
- LLM provider (`base_url`, `model`, `temperature`, `max_output_tokens`, `grounding_min_coverage`).
- Rate limiting (default **60 req/min**) and upload ceiling (default **32 MiB**).
- Object store (local filesystem or S3-compatible) and retention window (days or indefinite).

## Import sources

| Source | Format / Auth | Notes |
| --- | --- | --- |
| File upload | FIT | Garmin native format |
| File upload | FIT.GZ | Compressed FIT |
| File upload | GPX | GPS exchange format |
| File upload | TCX | Training Center XML |
| Strava export | FIT / GPX / TCX | Uploaded as exported files |
| Intervals.icu | HTTP Basic (API key) | Direct connector |

## API

A single, versioned **`/v1`** surface — no scattered prefixes.

- **OpenAPI 3.1** at `GET /v1/openapi.json`; human-readable docs at `GET /v1/docs`.
- **Bearer auth** via `POST /v1/auth/token`, with scope-based access control: `read`, `write`, `agent`, `sync`, `export`, `admin`.
- **Analytics** — endpoints for PMC, Critical Power, W'balance, aerobic decoupling, HRV, and TRIMP.
- **Agent** — `POST /v1/agent/ask` with SSE streaming (`text/event-stream`).
- **Activities** — history and per-activity detail.
- **Imports** — file uploads via `POST /v1/imports`.
- **Connections** — credential management.
- **Sync** — on-demand data synchronization.
- **Ops** — `GET /v1/system/status` for system status; `GET /healthz` liveness probe (process health only, no external dependencies).

## Tech stack

`Python 3.13` · `FastAPI` · `Uvicorn` · `SQLAlchemy 2.0+ (async)` · `Alembic` · `Pydantic + Pydantic Settings` · `LangGraph` · `OpenAI-compatible provider interface` · `NumPy + SciPy` · `Garmin FIT SDK + fitdecode + gpxpy + lxml` · `httpx` · `structlog` · `Docker (multi-stage hardened image)` · `uv`

## Data safety

wattwise-core is deliberately careful with your database:

- Creates and targets **only its own isolated schema**.
- **Never** connects to, reads, or modifies any pre-existing database.
- **Never** destroys or manages data volumes outside its schema.
- **Single-athlete scope** — one canonical store per deployment instance (no multi-tenant isolation; single-owner model).
- Encryption root key and signing keys live **in the environment only** — never in code or images.
- **No PII on disk** except in the configured object store (original file retention).
- **Structured logging to stdout/stderr only** — never to log files — with central log redaction enforced.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

## License

[Apache-2.0](./LICENSE).
