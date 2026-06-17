<div align="center">

# wattwise-core

### A self-hosted training brain that never makes the numbers up.

Your power meter never lies. Your analytics shouldn't either.

[![CI](https://img.shields.io/github/actions/workflow/status/bepcyc/wattwise-core/ci.yml?branch=main&logo=github&label=CI)](https://github.com/bepcyc/wattwise-core/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Typed: mypy --strict](https://img.shields.io/badge/typed-mypy%20--strict-2A6DB2.svg)](https://mypy-lang.org/)

**[What this is](#what-this-is)** · **[Why it is useful](#why-it-is-useful)** · **[Quick start](#quick-start)** · **[How it works](#how-it-works)** · **[Roadmap](#roadmap)** · **[For developers](#for-developers)**

</div>

---

## What this is

wattwise-core is a training brain you run on your own machine. It pulls your rides and your
daily wellness out of the files and platforms you already use, and brings them together into
one honest record of your training. From that record it computes the established
sports-science numbers — the same ones the coaching books and the research papers use.

On top of that record sits a coach you can talk to in any language. Ask it how your form is,
how much you have trained lately, whether you are recovering. It answers in plain words.

The thing that makes it different: the coach never invents a number. Every figure it states
is checked against your actual data first. If a number cannot be verified, the coach removes
it and tells you so, rather than saying something confident and wrong. When your data is too
old or a sync has quietly stopped, it says that too instead of mistaking a broken connection
for rest. A refusal you can trust beats a guess you cannot.

## Why it is useful

- **Your data stays yours.** Everything runs in one container on your box. Your rides, your
  wellness, your training history — none of it has to leave your network. You can even run
  the coach against a local model, so nothing ever goes to an outside service.
- **Answers grounded in your record.** The coach reads only your data and checks every claim
  against it. You get honest refusals instead of confident nonsense, the numbers you read
  are the numbers you actually rode, and it won't tell you to do what you've told it you can't —
  your stated limits (an injury, a doctor's advice) gate the advice.
- **Any language.** Ask in English, German, Russian, whatever you speak. The honesty rules
  hold in every language.
- **Real metrics.** The Performance Management Chart (fitness, fatigue, form), Normalized
  Power, Intensity Factor, TSS, Critical Power and W′, W′bal, aerobic decoupling, HRV,
  TRIMP, session-RPE load, a durability measure, and more — all computed from your record.
  [docs/METRICS.md](docs/METRICS.md) explains what every number means.
- **Works with what you already have.** Upload FIT, FIT.GZ, GPX, or TCX files
  (including Strava bulk exports), or sync directly from intervals.icu. No new gadget to buy.

## Quick start

You need [Docker](https://docs.docker.com/get-docker/), two random secrets (one command
each), and an LLM key for the coach (any OpenAI-compatible endpoint —
[OpenRouter](https://openrouter.ai/), or a local [Ollama](https://ollama.com/) if you want
to keep everything on your LAN). Build the image, then start it:

```sh
docker build -t wattwise-core:local .    # or use a released image

# Two secrets — keep SIGNING_KEY in your shell, it is also your login secret below
ENCRYPTION_KEY=$(python3 -c 'import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())')
SIGNING_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

docker run -d --name wattwise \
  -p 127.0.0.1:8000:8000 \
  -v wattwise_data:/var/lib/wattwise \
  -e WATTWISE_DATABASE_DSN='sqlite+aiosqlite:////var/lib/wattwise/wattwise.sqlite' \
  -e WATTWISE_ENCRYPTION_ROOT_KEY="$ENCRYPTION_KEY" \
  -e WATTWISE_TOKEN_SIGNING_KEY="$SIGNING_KEY" \
  -e WATTWISE_LLM_API_KEY='sk-...' \
  wattwise-core:local

# Wait for it — first boot sets up the database, then this returns {"status":"ready", ...}
curl --retry 15 --retry-delay 2 --retry-all-errors -fsS http://127.0.0.1:8000/readyz
```

> **Port conflict?** If port 8000 is already in use, pick a different host port:
> `-p 127.0.0.1:8001:8000`. If re-running after a previous test, remove the old
> container first: `docker rm -f wattwise`.

That is a complete setup. Your database and your uploaded files live on the `wattwise_data`
volume. On first boot the container builds its own database, so there is no extra migration
step. Your data is encrypted at rest, every request needs your token, the process runs as an
unprivileged user, and a bad configuration stops startup instead of running on quietly.

Now make your first requests. Your `SIGNING_KEY` is also the owner secret that mints your
first access token:

```sh
BASE=http://127.0.0.1:8000

# 1. Get an access token
TOKEN=$(curl -fsS -X POST "$BASE/v1/auth/token" \
  -H 'Content-Type: application/json' \
  -d "{\"owner_secret\":\"$SIGNING_KEY\"}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
AUTH="Authorization: Bearer $TOKEN"

# 2. Upload a ride (FIT / FIT.GZ / GPX / TCX) and bring it in
curl -fsS -X POST "$BASE/v1/imports" -H "$AUTH" -F file=@ride.fit
curl -fsS -X POST "$BASE/v1/sync/run" -H "$AUTH"

# 3. Read your activities and your fitness chart
curl -fsS "$BASE/v1/activities" -H "$AUTH"
curl -fsS "$BASE/v1/performance/load-fitness?from=2024-01-01&to=2024-01-08" -H "$AUTH"

# 4. Ask the coach (a streamed answer; needs WATTWISE_LLM_API_KEY)
curl -N -X POST "$BASE/v1/agent/ask" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"question":"How much training load have I done recently?","stream":true}'
```

**Configure it.** You can change the coaching model, point at PostgreSQL or MariaDB instead
of SQLite, tune the analytics, and more, all without touching code.
[docs/CONFIGURATION.md](docs/CONFIGURATION.md) covers every setting in plain language, each
one checked against a real running container. For a full PostgreSQL-backed production setup
with isolated networking, see [`deploy/compose.yaml`](deploy/compose.yaml).

## How it works

```
your sources   →  one honest record  →  sports-science numbers  →  grounded coach  →  API
(FIT/GPX/TCX,     (de-duplicated,        (PMC, CP/W', HRV,           (every claim
 intervals.icu)    conflicts resolved)    decoupling, TRIMP...)       checked, or removed)
```

![wattwise-core high-level architecture](./assets/img/wattwise-high-level-architecture.png)

A few rules hold the whole thing together:

- **One record of truth.** Every source flows through its own adapter into a single record.
  When two sources disagree, a clear, configurable trust policy decides which one wins, so
  your data does not silently fork.
- **No data, no number.** A metric that cannot be computed correctly is reported as
  unavailable with a reason, never as a zero or a plausible-looking guess.
- **The coach proves its claims.** The model proposes an answer; separate, deterministic code
  verifies each number against your record and removes anything it cannot stand behind. When
  it catches a number attached to the wrong thing, it puts the right one in its place.
- **You approve the plans.** When the coach moves from explaining your data toward suggesting
  a training plan, it stops at an approval step rather than acting on its own.

Storage is a single connection string: SQLite by default, or PostgreSQL or MariaDB by
changing that one setting, with no code changes.

## Roadmap

Releases are named after the people who changed how endurance sport is measured and
coached. Each name marks what that release is *about*. Every open issue is tagged with the
milestone it belongs to (`v0.0.1-banister`, `v0.0.2-coggan`, `future`, or `backlog`), so it
is always clear whether a piece of work is shipping in a named version, is on the longer-term
frontier, or is parked for triage.

### v0.0.1 — **Banister** · the honest foundation

Named for **Eric W. Banister**, who in the 1970s introduced the impulse-response
(fitness–fatigue) model and TRIMP — the mathematical ancestor of every fitness/fatigue/form
chart wattwise computes. This release is the bedrock: one de-duplicated record of truth, the
established sports-science metrics, and a coach that refuses to invent a number. The work
here makes the core honesty promise something we can actually *prove*.

- **#93** — Remove false-confidence tests (including GDPR-erasure tests that never exercise
  the production erase path) so "you can trust it" is a tested guarantee, not a hope.
- **#98** — VOICE-R2: turn the presentation strip into an allow-list so no internal metric
  code can ever leak into athlete-facing prose.
- **#95** — Surface the gathered activity id into the compose fact sheet so per-ride TSS
  claims are genuinely citable in production, not just in theory.
- **#103** — Scope the slow CI tiers to the change: a docs/text-only PR shouldn't pay the
  database, e2e, and image-build tax, while any DDL or source change still runs the full gate.

### v0.0.2 — **Coggan** · the metrics vocabulary

Named for **Andrew Coggan**, who turned the fitness–fatigue model into the power-meter
language the world now speaks: Normalized Power, Intensity Factor, TSS, and the Performance
Management Chart. This release widens the set of metrics the coach can compute *and cite*,
and makes the conversation layer sturdier.

- **#39** — Wire durability (fatigue resistance) all the way onto the service and agent
  surface, with `work_above_cp_j` persisted on ingest so the number can be retrieved and cited.
- **#87** — Two-layer coach answer: a verifiable evidence layer the grounder reads, plus warm
  visible prose for the athlete, split fail-closed so honesty and tone stop fighting.

### Future — **Seiler** · the training-science frontier

Named for **Stephen Seiler**, whose polarized-training research reframed how endurance
athletes balance easy and hard work. This is the direction wattwise is heading: understanding
not just *how much* you trained but *how*, and reasoning about the athlete as a human rather
than a row of numbers.

- **#76** — Training Intensity Distribution + Polarization Index: the easy/hard mix the
  engine doesn't yet see.
- **#78** — A pluggable feasibility-model registry that *falsifies* a prescription per
  (sport, goal) instead of pretending to predict an outcome.
- **#79** — Model the athlete as a human — life-state, real availability, motivation — and
  close the observe→adapt loop instead of optimizing open-loop.

### Backlog — triaged, not yet scheduled

- **#97** — Make the checkpoint `interrupt_id` stable across pause/resume (a known LangGraph
  re-run quirk; the fix may be a deterministic id or a documented spec carve-out).

## For developers

The engine is Python 3.13, FastAPI, SQLAlchemy (async), and a LangGraph coaching agent,
served over a single versioned `/v1` REST surface with OpenAPI 3.1 at `GET /v1/openapi.json`.

```sh
just bootstrap          # set up dependencies and the database
just gate               # the full pre-merge gate CI runs: lint, types, all test tiers, eval, coverage
```

Tests are tiered by marker (unit, property, golden, contract, fuzz, integration, e2e,
portability, injection, logging) and run in parallel (`uv run pytest -n auto`). The agent's
grounding and abstention behaviour is guarded by an offline evaluation suite.

- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every setting, in plain language.
- [docs/METRICS.md](docs/METRICS.md) — what every number means and how it is computed.
- [CONTRIBUTING.md](./CONTRIBUTING.md) — how to build, test, and send a change.
- [Apache-2.0](./LICENSE) — self-host it, fork it, build on it. Just keep the notices.

---

<div align="center">
<sub>Built by athletes who read the papers. <b>Your watts. Your server. Your truth.</b></sub>
</div>
