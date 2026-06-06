# 0002 — `source_candidate` carries a `payload` JSON column for re-resolution

- Status: Accepted
- Date: 2026-06-06
- Requirements: UPS-R4, CONF-R6, LIN-R2, MAP-R2 (doc 20); SUB-R3, RAW-T-R1 (doc 80); PRIV-R8 (doc 70)

## Context

`source_candidate` is the per-source lineage envelope (tier-2 store, LIN-R2): one row per
`(athlete_id, source_descriptor_id, source_native_id, gbo_type)` — the only key in which
source identity appears (UPS-R1). The conflict resolver (`resolve_field`, CONF-R2) selects
the winning value for each canonical field across the candidates that overlap on a real
session, using trust ranking and recency.

The original schema stored only the *resolution inputs* on the candidate (content hash, trust
profile, confidence, observed/fetched clocks, resolved-* back-pointers) but **not the mapped
canonical values themselves**. That made several required behaviours impossible without going
back to the source:

- **UPS-R4 / CONF-R6 re-resolution** — when a higher-trust source is withdrawn (SUB-R1) or
  re-added (SUB-R3), each affected canonical field must be re-resolved from the **retained**
  candidates **with no network re-fetch**. Without the mapped values on the candidate, the
  resolver has nothing to re-resolve from.
- **SUB-R3** explicitly forbids a network fetch during upward re-resolution; the value must
  come back from retained state alone.
- **RAW-T-R1** idempotent re-derivation reads tier-1 originals and/or tier-2 candidates, never
  a prior canonical value — so the candidate must hold the per-source mapped fields.

## Decision

Add a non-null portable `payload` JSON column to `source_candidate` holding the **adapter's
mapped canonical payload** — the source's contribution expressed in canonical, source-neutral
terms (canonical field names only; no source-named keys, MAP-R2). The conflict resolver reads
the competing candidates' `payload`s to (re-)select each field's winner, so a withdrawal /
re-add / improved-mapping re-derivation runs entirely over retained tier-2 state.

```python
# wattwise_core/persistence/models/source.py — SourceCandidate
payload: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)
```

The column uses the portable `json_column` factory (TEXT-backed on SQLite, JSON on
PostgreSQL/MariaDB) so it round-trips identically on all three backends (BOOT-R3), and it is
added through a versioned ORM migration, never hand-written DDL (RUN-R7.1 / BOOT-R2).

## Consequences

- **Re-resolution without re-fetch** (UPS-R4 / CONF-R6 / SUB-R3) works: the resolver has the
  per-source mapped values retained on the candidate, so removing or re-adding a source
  recomputes the canonical fact from retained state with no network call.
- **Source-neutrality preserved** — `payload` holds only canonical field names; the source
  cannot leak into the canonical record (MAP-R2 / CON-R2 stay green).
- **Tier-2 stays non-consumer** — `payload` is read only by the resolver/re-derivation path,
  never by analytics/agent/API (LIN-R4 / PRIV-R11.4); consumers still read the resolved
  canonical tier only.
- **Erasure scope unchanged** — `payload` lives on the athlete-scoped `source_candidate` row,
  so per-athlete erasure (PRIV-R8) removes it with the rest of the candidate; no new store or
  retention category is introduced.
- It is a backward-compatible additive column with a `default=dict`, so existing rows and the
  idempotent re-ingest path (UPS-R3) are unaffected.
