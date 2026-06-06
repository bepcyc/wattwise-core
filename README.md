# wattwise-core

> The open-source endurance-training **analytics engine** with a **trustworthy AI coaching agent**.
> Apache-2.0 · Python 3.13 · single-package (`import wattwise_core`) · single-athlete, self-host.

`wattwise-core` is the engine of the **`wattwise`** OSS family. An athlete connects the training-data
sources they already use and uploads raw activity files; the engine unifies everything into **one canonical
record of truth**, computes **sports-science-correct analytics** (PMC, NP/IF/TSS, CP/W′, W′bal, aerobic
decoupling, HRV), and runs a **LangGraph coaching agent** that answers questions and reviews training load —
grounded **fail-closed** against the canonical data, never fabricating.

The OSS client surface is the **REST API + OpenAPI specification + a generated typed client** — there is no
bundled GUI. The web app and the Telegram bot are part of the closed commercial product **`athload`**, which
builds multi-tenancy, billing, and managed connectors additively on top of this engine. None of that lives
here.

## Three load-bearing principles

- **A — Source-agnostic canonical model.** Each source is a pluggable **adapter** mapping its source-shaped
  objects into the canonical domain model through an anti-corruption layer. Analytics, the agent, and the API
  read **only** the canonical store — never source-shaped data, never branching on source name.
- **B — Direct typed ingestion; MCP only for agent tool-use.** Ingestion uses direct typed source clients;
  the Model-Context-Protocol tool layer is only the agent's runtime interface to the same internal services.
- **C — A trustworthy agent.** Typed graph state + durable checkpointing, provider-enforced structured
  outputs, deterministic **fail-closed** grounding, prompt-injection isolation, and an offline eval suite.

## Quick start

```sh
just bootstrap         # clone -> running, health-serving instance against a DSN
just lint type test    # the deterministic gates
just test-db-portable  # ORM round-trips identically on SQLite / PostgreSQL / MariaDB
```

Configuration is layered (packaged defaults → optional operator file → `WATTWISE_*` env vars). **Secrets**
(database DSN, LLM key, signing key, encryption root key) come **only** from the environment / a secret
manager at startup — never from a committed file. A missing required secret **fails the boot closed**.

## Supported sources (Phase-1 / OSS)

| Source | Archetype | Library |
|---|---|---|
| FIT / GPX / TCX file upload (incl. Strava's own export) | `file_upload` | `garmin-fit-sdk` · `gpxpy` · `lxml` |
| Intervals.icu | `api_key` (HTTP Basic) | thin `httpx.AsyncClient` |

Garmin and Xert direct connectors are **commercial managed connectors** (`athload`), not shipped here.

## Data safety

The engine creates and targets **only its own** schema in an isolated store. It never connects to, reads, or
modifies any pre-existing database, and never destroys a data volume.

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
