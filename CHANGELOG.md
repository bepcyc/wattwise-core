# Changelog

All notable changes to `wattwise-core` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses automated
semver derived from Conventional Commits.

## [Unreleased]

### Changed (BREAKING — endurance-score "durability" renamed to "curve_shape", issue #26)
- The endurance-score component historically named `durability` is the FRESH-state
  power-curve shape ratio `MMP(long)/MMP(short)`, not fatigue resistance. With the true
  work-conditioned durability metric landing (below), the component is renamed
  `curve_shape` so "durability" means exactly one thing. Two operator/consumer-facing
  consequences, both fail-closed rather than silent:
  - **Configuration keys renamed** — `endurance_score_weight_durability` →
    `endurance_score_weight_curve_shape`, `endurance_score_durability_floor` →
    `endurance_score_curve_shape_floor`, `endurance_score_durability_ceiling` →
    `endurance_score_curve_shape_ceiling`. An operator override using an old key (e.g.
    `WATTWISE_ANALYTICS__ENDURANCE_SCORE_WEIGHT_DURABILITY` or the operator config
    file) now fails settings validation at boot; rename the key. Defaults are unchanged.
  - **`QualityReport` component strings renamed** — endurance-score quality reports now
    record `curve_shape` (not `durability`) in `components_present` /
    `components_missing`. Records persisted before this release carry the old string;
    consumers keying on these strings (dashboards, alerts, stored deliverables) should
    treat `durability` in pre-rename endurance-score records as `curve_shape`. Stored
    reports are immutable lineage and are not rewritten.

### Added
- Durability / fatigue resistance (issue #26): the work-conditioned power decrement —
  best target-duration power fresh vs. after a per-athlete amount of accumulated work
  above Critical Power (the intensity-weighted W′-expenditure axis), with sufficiency
  gating as the default path (`Unavailable(INSUFFICIENT_DATA)` when the record cannot
  support the number), a non-blocking `fresh_effort_below_cp` quality flag, and new
  `[analytics]` keys `durability_target_duration_s` / `durability_wprime_multiple`.
- Binding-faithful grounding (issue #10): the deterministic claim-binding layer re-derives
  each NUMBER claim's `(metric, as_of)` verification target from the claim's own sentence
  (metric mis-attributions corrected in place, stale-as-current dates dropped, dated
  sentences pinned to their stated date, metric-shaped sentences barred from the
  user-request echo pass), an optional decorrelated sentence-entailment gate over a
  code-rendered canonical fact sheet (MiniCheck-class local verifier, fail-closed when
  unavailable), and split-conformal calibration of the gate's publication thresholds.
  New `[agent.binding]` / `[agent.entailment]` configuration and binding/entailment
  observability counters. The conformal calibration artifact is provenance-pinned:
  it stamps the verifier model, the claim-extraction prompt hash, and a dataset
  version, all checked at load — a stale or unstamped artifact fails the boot.
- Repository scaffold, PEP-621 packaging, layered fail-closed configuration.
