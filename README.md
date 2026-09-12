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
- **Works with what you already have.** Upload FIT, FIT.GZ, GPX, or TCX files exported from
  your device or any service — Garmin, Strava ("Export Original"), intervals.icu. No new
  gadget to buy.

## Quick start

You need [Docker](https://docs.docker.com/get-docker/), two random secrets (one command
each), and an LLM key for the coach (any OpenAI-compatible endpoint —
[OpenRouter](https://openrouter.ai/), or a local [Ollama](https://ollama.com/) if you want
to keep everything on your LAN). Build the image from this repo, then start it:

```sh
# Build from source — matches these docs and works on any machine, Intel/AMD or ARM.
# On Apple Silicon, a Raspberry Pi, or any ARM box, build locally (this is the path to use).
docker build -t wattwise-core:local .

# Prefer not to build? A prebuilt image also exists, but it runs on x86/Intel machines only
# (not Apple Silicon, a Raspberry Pi, or other ARM hardware):
#   docker pull ghcr.io/bepcyc/wattwise-core:v0.0.1    # x86/Intel only; if you use it, put
#                                                       # this name in the `docker run` below.

# Two secrets — keep SIGNING_KEY in your shell, it is also your login secret below
ENCRYPTION_KEY=$(python3 -c 'import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())')
SIGNING_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

docker run -d --name wattwise \
  -p 127.0.0.1:8000:8000 \
  -v wattwise_data:/var/lib/wattwise \
  -e WATTWISE_DATABASE_DSN='sqlite+aiosqlite:////var/lib/wattwise/wattwise.sqlite' \
  -e WATTWISE_ENCRYPTION_ROOT_KEY="$ENCRYPTION_KEY" \
  -e WATTWISE_TOKEN_SIGNING_KEY="$SIGNING_KEY" \
  wattwise-core:local
# (Upload + analytics need no LLM key. The coach in step 4 does — add
#  -e WATTWISE_LLM_API_KEY=<your real key> and restart when you want it.)

# Wait for it — first boot sets up the database, then this returns {"status":"ready", ...}
curl --retry 15 --retry-delay 2 --retry-all-errors -fsS http://127.0.0.1:8000/readyz
```

> **Port conflict?** If port 8000 is already in use, pick a different host port:
> `-p 127.0.0.1:8001:8000`. If re-running after a previous test, remove the old
> container first: `docker rm -f wattwise`.

That is a complete setup. Your database and your uploaded files live on the `wattwise_data`
volume. On first boot the container builds its own database, so there is no extra setup
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
```

**Bring your data in.** wattwise reads activity files you already own — no new gadget needed.
Export one (or many) in **FIT / FIT.GZ / GPX / TCX** from wherever you train:

- **Garmin Connect** — open an activity → the **⋯** menu → *Export to FIT* (or *Export to GPX*).
- **Strava** — open an activity → **⋯** → *Export Original* (the file you uploaded). For your
  whole history, request a bulk archive under *Settings → My Account → Download or Delete Your Account*.
- **intervals.icu** — open an activity → *Download original file*.
- **Your watch / head-unit** — copy the `.fit` files straight off the device over USB
  (e.g. the `GARMIN/Activity` folder).

Then upload the file you exported:

```sh
# 2. Upload your exported file (use your own path, not a literal "ride.fit")
#    The upload ingests the activity in place — no extra sync step for files.
curl -fsS -X POST "$BASE/v1/imports" -H "$AUTH" -F file=@/path/to/your-activity.fit
```

File upload is how you get your data in on this self-hosted build. Automatic sync from a
service like **intervals.icu** isn't available in this build yet — for now, bring your data in
by uploading files, as shown above.

Set your FTP so the power numbers light up:

```sh
# 2b. Set your sport and FTP (watts). Training Stress Score (TSS), Intensity Factor,
#     and the whole fitness/fatigue/form chart are computed FROM your FTP — without it
#     they stay null/zero, so set it once before reading the chart. Use your real FTP.
curl -fsS -X PUT "$BASE/v1/athlete" -H "$AUTH" \
  -H 'Content-Type: application/json' -d '{"current_sport":"cycling"}'
curl -fsS -X PUT "$BASE/v1/athlete/signature" -H "$AUTH" \
  -H 'Content-Type: application/json' -d '{"ftp_w":250}'
```

> First chart all zeros? You almost certainly skipped this step. `GET /v1/onboarding/status`
> will say so — its `suggested_next_step` reads `set_ftp` until an FTP for your current sport
> is set, then advances to `all_set`.

Now read your data back and ask the coach:

```sh
# 3. Confirm it landed — the file you just uploaded should now appear here:
curl -fsS "$BASE/v1/activities" -H "$AUTH"

# Your fitness chart. Use a date window that covers the rides you uploaded
# (from/to are required — substitute the year(s) your activities are from):
curl -fsS "$BASE/v1/performance/load-fitness?from=2026-01-01&to=2026-12-31" -H "$AUTH"

# 4. Ask the coach (a streamed answer). The coach needs an LLM key — add
#    -e WATTWISE_LLM_API_KEY=<your real key> to the `docker run` above and restart first:
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

It runs on SQLite out of the box, and you can point it at PostgreSQL or MariaDB when you
outgrow that — no code changes.

## Roadmap

Releases are named after the people who changed how endurance sport is measured and coached —
each name marks what that release is *about*. The detailed, tracked work for every release
lives on its **[GitHub milestone](https://github.com/bepcyc/wattwise-core/milestones)**, which
is where to look if you want to follow the engineering or help with it.

### v0.0.1 — **Banister** · the honest foundation

Named for **Eric W. Banister**, who in the 1970s introduced the impulse-response
(fitness–fatigue) model and TRIMP — the mathematical ancestor of every fitness/fatigue/form
chart wattwise computes. This release is the bedrock: one de-duplicated record of truth, the
established sports-science metrics, and a coach that refuses to invent a number.

For you, that means the honesty promise becomes something you can rely on rather than take on
faith: the coach answers in plain words, and it can point you to the exact ride a number came
from.

### v0.0.2 — **Coggan** · the metrics vocabulary

Named for **Andrew Coggan**, who turned the fitness–fatigue model into the power-meter
language the world now speaks: Normalized Power, Intensity Factor, TSS, and the Performance
Management Chart.

For you, that means more of the metrics you already know — including a fatigue-resistance
(durability) measure — and a coach that can cite each one as it talks you through your
training.

### Future — **Seiler** · the training-science frontier

Named for **Stephen Seiler**, whose polarized-training research reframed how endurance athletes
balance easy and hard work.

Where wattwise is heading: understanding not just *how much* you trained but *how* — the
easy/hard balance polarized training is built on — and a coach that reasons about you as a
person, with your real availability, your goals, and your limits, and that is honest about
whether a goal is realistic for you.

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
