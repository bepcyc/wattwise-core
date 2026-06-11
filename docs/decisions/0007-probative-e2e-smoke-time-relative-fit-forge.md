# ADR 0007 — Probative E2E smoke: time-relative FIT forge + dual-oracle agent assertions

Status: accepted (2026-06-11). Source: issue #29 — a live container probe showed the E2E
smoke (`tools/e2e_smoke.py`) could go green without ever demonstrating the product's
headline ability. Two weaknesses, one root cause:

1. **`degraded` counted as a pass.** The agent-ask step accepted
   `status in {"completed", "degraded"}`, so an agent that honestly refuses *every*
   question goes green. The refusal is correct runtime behavior (GROUND-R6 fail-closed),
   but a smoke that cannot distinguish "answered" from "refused" proves neither.
2. **The fixture could never ground a recency question.** `ride.fit` is dated 2024-01-02;
   any "recent training" ask is — correctly — refused on recency/sufficiency grounds. A
   perfect engine could only ever show the refusal path under this fixture.

## Decision

Synthesize the FIT fixture at smoke time with timestamps computed relative to "now", and
split the single blended agent assertion into two explicit oracles — the refusal AND the
grounded answer — so each product guarantee is pinned separately and a pass can only mean
what its label says.

### Survey: what exists for synthetic activity files (web research, 2026-06)

- **`garmin-fit-sdk` (official, already a dependency)** is decode-only; Garmin has stated
  on its developer forum that no Python encoder is planned.
- **`fit-tool` (PyPI, BSD-3)** is the only maintained Python FIT *encoder* (latest release
  2026-02). It is a community fork of a package whose original author deleted it from
  PyPI — a supply-chain yellow flag for a dependency this repo would pull only for a smoke
  tool.
- **No off-the-shelf realistic synthetic *workout generator* exists** in any surveyed
  ecosystem — only encoders, converters (FIT↔TCX↔GPX), and file-merge utilities. A
  deterministic, time-relative activity forge is therefore something we build, not buy.

Test-data guidance surveyed for the design (flaky-fixture and test-oracle literature):
tests should *create* the data they need at run time rather than depend on static
fixtures whose meaning decays with the calendar; generated data must be deterministic
(seedless randomness breeds flakes); assertions must be specific enough that a test
cannot pass for the wrong reason; and both the happy path and the failure path are
product guarantees worth pinning separately.

### What was built

- **`tools/fit_forge.py`** — a stdlib-only FIT activity encoder (~300 lines: header,
  definition/data messages, the FIT CRC-16). `forge_recent_batch()` emits four
  deterministic 20-minute tempo rides at 1 Hz placed 1/4/7/11 days before "now": the
  newest sits inside the caveat-free freshness zone (`readiness_fresh_staleness_days=2`)
  and the batch spans the two-week window a recency ask reads, clear of the
  `readiness_max_staleness_days=14` hard floor. Only the timestamps move with the clock —
  every sample is a fixed function of its index, so a failing smoke reproduces
  byte-identically for the same start instant. Each ride carries a distinct
  `file_id.serial_number` + `time_created`, i.e. its own MAP-R10 STRONG fingerprint, so
  dedup can never merge the batch. 1 Hz sampling is load-bearing: the analytics resampler
  interpolates gaps only up to `MAX_INTERP_GAP_S = 3 s` (ANL-R8), so a sparser stream
  fails closed out of every power/HR metric.
- **Round-trip contract test** (`tests/unit/test_fit_forge.py`) — every forged file must
  decode through the *primary* `garmin-fit-sdk` decoder (never the corrupt-file recovery
  fallback) via the production `decode_fit` path, with the exact telemetry the smoke
  relies on. Hand-rolled encoding is acceptable *because* this pin exists: the forge can
  never silently drift from what ingest accepts.
- **`tools/e2e_smoke.py` rework** — the journey now pins both guarantees explicitly:
  - *Honest refusal* (before any import): terminal `degraded` with **zero** citations on
    the empty profile — provoked deliberately, labeled `agent honest refusal (empty
    profile)`, never blended into the answer step's pass.
  - *Set FTP signature* (`PUT /v1/athlete/signature`, `effective_date` predating all
    fixtures): without an effective FTP, NP/IF/TSS → CTL are typed-unavailable and no
    power load can ever exist (ANL-R9/GBO-R26) — the live probe's "refusal even with
    data" had this as a second silent cause.
  - *Deterministic recency proof* (model-free): after importing the forged batch, the PMC
    over the last two weeks must show `fitness > 0`. This pins that the fixture *can*
    ground a recency question even on runs without a model key, where the agent steps are
    SKIPPED-and-failing.
  - *Grounded answer*: terminal `completed` with **≥ 1 citation**. `degraded` no longer
    passes here.

### Alternatives rejected

- **Adopt `fit-tool`** — full-profile encoder, less code; rejected for the smoke because
  of the deleted-then-forked PyPI provenance, an extra dependency for a dev tool, and the
  fact that the round-trip contract test would still be required. Revisit if the forge
  grows toward full-profile realism (HRV, GPS courses, multi-sport) — it is the best
  available library for that.
- **Timestamp-shift the static `ride.fit` bytes** — rewriting uint32 timestamps in place
  across record/session/lap/file_id messages plus CRC recompute is strictly more fragile
  than encoding from scratch, and still cannot vary content.
- **Seed the DB directly (as the in-process journeys do)** — the smoke's value is that it
  drives the real HTTP boundary of the built stack; a DB seed would bypass import, decode,
  dedup, and sync, which is exactly the path the forged upload exercises.

### Follow-up (not in this change)

The forge is deliberately reusable beyond the smoke: the integration/contract suites
currently hand-craft candidate payloads or reuse the static fixtures, and a deterministic
generator parameterized by start/duration/power-shape (and later TCX/GPX via the same
sample plan — both are plain XML) would let recency-, dedup-, and load-sensitive tests
state their data in domain terms. Tracked as future work rather than widened into this
change.
