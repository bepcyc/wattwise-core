# ADR 0003 — Cross-model (Codex) gate: MMP definition + candidate versioning

Status: accepted (2026-06-06). Source: the per-epic cross-model convergence gate
(`HOW_THIS_WAS_CREATED.md` §9.3) — a low-reasoning Codex/GPT-5.5 review of the converged
persistence/analytics/ingestion epics. Three findings; resolutions below.

## 1. MMP-R1 exactly-`d` vs MMP-R3 monotonicity — SPEC FIXED

**Tension.** Doc 40 `MMP-R1` defined `MMP(d)` as the max mean over *exactly*-`d`-second
windows, while `MMP-R3` mandates the curve be non-increasing in `d` **by construction,
never by clamping**. These contradict: the exact-`d` maximum is not monotone on arbitrary
finite power (e.g. power `[1, 0, 1]` → 2 s peak `0.5` < 3 s peak `0.667`).

**Decision (Codex verdict + adopted).** Amend the spec, not the code. `MMP(d)` is the
field-standard **power-duration envelope**: the best mean over any valid window of length
`≥ d` (the suffix-max of the per-exact-length peaks). This is non-increasing by
construction (the feasible window set shrinks as `d` grows) and matches the implementation
(`analytics/mmp_cp.py`, `MMPWindow.window_len_s ≥ duration_s`). `spec/40` `MMP-R1` and
`MMP-R3` were edited accordingly. `BEST-R1` (best efforts derive from `MMP`) is unchanged.

## 2. ANL-R12 — power MMP not sport-gated — CODE FIXED

`mmp()` accepted a `sport` argument but returned `Computed` for any sport with a power
channel. Power MMP is cycling-power-specific (`ANL-R11/R12`); requested for a non-power
sport it now returns `Unavailable(NOT_APPLICABLE_FOR_SPORT)` rather than a plausible-but-
incommensurable number. (`analytics/mmp_cp.py`.)

## 3. `source_candidate` candidate-key vs versioning — DECISION + residual

**Finding.** The `source_candidate` UNIQUE was widened to
`(athlete_id, source_descriptor_id, source_native_id, gbo_type, content_hash)` so a changed
restatement can land as a new retained version (`UPS-R5`/`PRV-R2`) while byte-identical
re-ingest stays an idempotent no-op (`UPS-R3`). Codex noted the documented candidate key
(`UPS-R1`) is the bare 4-tuple, so the DB no longer enforces "one row per 4-tuple".

**Decision.** The 4-tuple is the *object* identity; the **candidate-version** key is the
5-tuple (object + `content_hash`), which IS UNIQUE-backed (`IDX-R2`). Multiple versions of
one object coexist by design (versioning); the current version is the non-`is_superseded`
row with the latest `observed_at`. This is a faithful realization of `UPS-R4/R5/PRV-R2`
(retain + version + supersede) on a portable schema (a partial unique index "one current
per 4-tuple" is not portable across SQLite/PostgreSQL/MariaDB).

**Residuals (tracked, low-risk for single-athlete serial on-demand OSS sync):**
- `UPS-R2` atomic seam: `ingest._persist_candidate` uses select-then-write rather than the
  dialect-aware upsert seam. The race only matters under concurrent syncs, which OSS does
  not do (sync is owner-triggered and serial). A future round should route the idempotent
  candidate-version insert through the upsert seam (`ON CONFLICT (5-tuple) DO NOTHING`).
- A spec clarification to `doc 20 UPS-R1` naming the candidate-version key explicitly would
  remove the apparent 4-tuple/5-tuple discrepancy.
