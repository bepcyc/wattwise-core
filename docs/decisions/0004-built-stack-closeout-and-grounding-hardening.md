# ADR 0004 — Built-stack close-out (ROAD-R1-EXIT) + grounding hardening

Status: accepted (2026-06-08). Source: the ROAD-R1-EXIT close-out — wiring the API-level
E2E journeys (E2E-R1 a–d) against the REAL assembled app surfaced several gaps where the
*built stack* could not actually run, plus a convergence-review panel + a low-reasoning
cross-model (Codex/GPT-5.5) pass over the change set. Findings and resolutions below.

## 1. Owner identity — the built stack could not serve any athlete data — FIXED

The token route minted `sub = "owner"` (a non-UUID string), but every canonical read
coerces `uuid.UUID(athlete_id)`, so the deployed app returned `500` on every data/agent
endpoint. Fixed: a deterministic single-owner anchor `OWNER_ATHLETE_ID =
uuid5(NAMESPACE_DNS, "owner.oss.wattwise.invalid")` (`wattwise_core/identity.py`), seeded as
the one `athlete` row by migration `0001`, and minted as the token subject (`AUTH-R3/R18`).
The id is a referential anchor, not a secret: minting still requires the HMAC-verified
owner secret and using a token still requires the signing key (GBO-R13, single-owner OSS).

## 2. Production agent runtime — never actually invoked — FIXED

The deployable `GraphAgentEngine` was assembled but never exercised end-to-end (no model
key in prior runs). Wiring the E2E exposed two breaks:

- **Compiled-graph invocation.** `deliverables.answer_question` drives the graph through the
  `CoachGraph.run(state)` seam, but `build_graph` returns a langgraph `CompiledStateGraph`
  exposing only `ainvoke` (which requires a `{thread_id, recursion_limit}` config). Added
  `_CompiledCoachGraph` adapter supplying the durable-thread config (CKPT-R3) with the
  superstep bound kept *above* the graph's own node-visit ceiling so a pathological run
  degrades gracefully rather than raising (`GRAPH-R5/OUTCOME-R3`). Thread id fails closed if
  absent (never aliases onto a shared constant key).
- **Numeric grounding never grounded.** The production `ClaimGrounder` passed `CanonicalEvidence`
  (async `metric_value` only) to the synchronous grounder, which reads a sync
  `metric_snapshot` — so every NUMBER claim scrubbed → ABSTAIN, and the agent could never
  confirm a number. Fixed: `ClaimGrounder` pre-resolves each claimed `(metric, as_of)` via the
  async canonical read into a `_SnapshotEvidence` wrapper exposing `metric_snapshot`, and
  `_ExtractedClaim` gained `as_of` → `Claim.ref`. Numbers now ground VERBATIM (`GROUND-R7`).
- **Grounded-number citation dropped.** `_metric_citation` carried no `record_id`, so the
  deliverables projection (`_project_citations` filters on it) silently dropped a grounded
  number's citation — a number would ship uncited (`GROUND-R5`). Added a stable
  `record_id = "{metric}@{as_of}"`.

## 3. Ingestion wiring — connect→sync→data could not run on the built stack — FIXED

`create_app` never wired the import processor / sync orchestrator / credential sink, so
`POST /v1/imports` and `POST /v1/sync/run` returned `500`. Added a composition root
(`api/wiring.py`) that wires them to the real OSS services. To keep the routers source-blind
(`ARCH-R22`), the file-import composition selects the adapter from the registry by the
built-in `file_import` key and drives a new neutral `FileImportAdapter.decode_upload` seam
(`ingestion/base.py`) — no router/composition imports a named adapter. The credential store
is built only when an envelope key is configured (`BOOT-R4`); without one the api_key connect
path stays fail-closed at the probe (`422`), never `500`. `.fit.gz` uploads now decompress in
`decode` (previously advertised-accepted but always rejected).

## 4. Grounding hardening (Codex/convergence findings) — FIXED

The extract-then-verify grounder trusted the model's claim-kind label. Closed two
demonstrated holes deterministically:

- A non-prescriptive `STATEMENT` carrying a numeric literal or URL is no longer publishable
  as `complementary` — it is treated as `ungrounded` and scrubbed (`GROUND-R9`: "a statement
  carries no checkable token"). A number can never ship by being mislabeled non-factual.
- A second, extraction-independent URL sweep scrubs every URL in the body not on the
  allow-list / a matched record, regardless of whether the model extracted it
  (`GROUND-R4`: invented URLs scrubbed unconditionally).

## 5. Accepted residuals (recorded, not yet closed)

- **Multi-turn follow-up / durable resume.** `GraphAgentEngine` uses a fresh `InMemorySaver`
  per call and `answer_question` does not thread the API `thread_id`/`follow_up` into the
  run, so a follow-up turn does not resume a prior durable thread. This matches the OSS
  Phase-1 scope (grounded Q&A + weekly load review; the durable SQLAlchemy checkpointer + the
  follow-up trigger are wired in a later phase). Single-turn grounded Q&A is correct and
  fail-closed today; durable multi-turn is deferred, not silently broken.
- **Number-extraction completeness.** Deterministic verification still covers the claims the
  model extracts; a numeric span the model neither extracts nor labels is bounded by the
  voice number-cap (`VOICE-R7`, enforced in `deliverables`), the leads-with-state projection,
  and the statement/URL deterministic nets above — but a full number-span sweep (scrub every
  numeric token not backed by a grounded value) is deferred to avoid over-scrubbing benign
  non-factual numerals; tracked for the grounding-robustness pass.
