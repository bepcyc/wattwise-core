# Changelog

All notable changes to `wattwise-core` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses automated
semver derived from Conventional Commits.

## [Unreleased]

### Added
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
