# ADR 0006 — T-MUT split: diff-scoped, time-boxed, cache-warmed advisory PR leg; nightly enforcing campaign

Status: accepted (2026-06-11). Source: issue #35 — the PR-stage T-MUT job (advisory
mutation testing) ran 43+ minutes on a pull request. An advisory job — one that by
design can never block the merge — must not be the long pole of PR feedback. Mutation
testing is inherently expensive (each surviving mutant is a full test session), so the
cost is moved to where it belongs: the PR loop pays only for the **diff**, the nightly
leg pays for the **campaign**.

Research basis: mutmut 3.6 released source (verified, not docs-folklore); the
incremental-mutation pattern proven across ecosystems — StrykerJS incremental mode,
cargo-mutants `--in-diff`, PIT `scmMutationCoverage` (diff scoping alone is the 10–50×
lever); and the standard CI-cache pattern of PRs warm-starting from the latest
main-branch mutation cache.

## 1. The decision

One implementation, `scripts/mutation_gate.py`, drives two legs (the workflow YAMLs
stay thin schedulers, CI-R0, with the IDENTICAL recipe set on both forges, CI-R9):

- **`just test-mut-pr` (PR, advisory)** — mutate only the merge-base diff, under a hard
  wall-clock budget, warm-started from the persisted `mutants/` cache. Survivors and
  budget exhaustion exit 0 (advisory means advisory; the workflow job additionally
  carries `continue-on-error`); only infrastructure failures (clean tests broken,
  mutmut crash before any verdict) are red — a gate that cannot run must fail, never
  silently pass.
- **`just test-mut` (nightly, enforcing)** — the full campaign, incremental on the same
  cache, enforcing the `WW_MUT_FLOOR` mutation-score floor over the correctness-critical
  packages (`wattwise_core.analytics` + `wattwise_core.ingestion.adapters` — the exact
  scope of the 95% coverage floor, DOD-R1). The floor is a ratchet: 0 (report-only)
  until the first nightly baseline lands, then only ever raised.

Tool: **mutmut 3** (`mutmut>=3.6`, dev group). Decisive over alternatives because the
expensive part of issue #35 item 3 is native: mutmut 3 maps tests to mutated functions
during stats collection and runs ONLY those tests per mutant, fastest-first, `-x`,
under per-mutant CPU/wall limits, forking workers across all cores — never
whole-suite-per-mutant.

## 2. How each optimization direction of issue #35 lands

1. **Mutate only the diff.** Changed files vs the `origin/main` merge base (plus local
   tracked edits, so the local run sees what the contributor is editing) map to mutant
   name patterns (`src/wattwise_core/analytics/forms.py` →
   `wattwise_core.analytics.forms.x*`; mutant keys always start with `x`), passed to
   `mutmut run <patterns>`. Generation still covers the tree (cached, see 4); but
   **execution** — the expensive part — touches only the PR's mutants.
2. **Time-box with honest reporting.** `WW_MUT_BUDGET_SECONDS` (default 480 s PR /
   10 800 s nightly). On expiry mutmut receives SIGINT: its KeyboardInterrupt path
   stops workers, and because verdicts are persisted per mutant as they complete
   (`mutants/<path>.meta`), everything measured stays measured. The report states
   "measured N% on the M of K scoped mutants completed within budget" — partial
   information is acceptable for an advisory leg; an unbounded advisory job is not.
3. **Cheap kill signal.** `[tool.mutmut]` pins test selection to the fast offline tiers
   (`tests/unit|property|golden|contract`, TIER-R1 — service-backed tiers never run in
   the mutant loop) and adds `--no-cov -p no:cacheprovider -p no:xdist`: no coverage
   instrumentation inside the loop, no pytest cache churn in the CI-cached dir, no
   xdist under mutmut's own forking.
4. **Cache across runs.** The whole `mutants/` dir (generated mutants + verdicts +
   test-to-function stats with timings) persists via the forge cache, keyed
   `tmut-v1-<hash(uv.lock)>-<run-id>` with prefix restore keys; the nightly run on
   main saves the cache every PR restores. **The trap this ADR exists to record:**
   mutmut's cache validity is mtime-based, and a fresh CI checkout stamps every file
   with clone time — silently invalidating 100% of the restored cache (and re-running
   stats collection, the single biggest fixed cost). The gate therefore restores each
   tracked file's mtime to its last-commit time first (one `git log -m --first-parent`
   walk — merge commits attribute files to merge time, so anything merged after the
   cache snapshot reads as newer), then force-touches the PR-changed files so exactly
   those are re-mutated; changed non-Python files under `src/` and deleted sources get
   their stale copies evicted from `mutants/` explicitly (mutmut only re-copies
   non-Python files when absent).
5. **Skip when out of scope — verifiably.** The path filter is code inside the gate,
   identical on both forges, not per-forge YAML that can silently rot (the suspected
   cause of the 43-minute run on an unrelated PR): a PR with no mutable source change
   writes an explicit "skipped (out of scope)" report and exits 0.

## 3. Scoring and report

Score = detected / (detected + undetected), where detected = killed + timeout +
segfault (+ caught-by-type-check), undetected = survived + **suspicious** (unexpected
exit codes count against the score so noise can never inflate it). "No tests" mutants
are reported separately as a test-signal gap, not blended into the score. Both legs
write `reports/mutation-<leg>.{md,json}` (retained artifacts, CI-R6) and append the
markdown — including the surviving mutant names, the actionable output — to the CI
step summary. Inspect any survivor locally with `uv run mutmut show <name>`.

## 4. Expected effect (acceptance: PR-stage T-MUT P95 ≤ 10 min)

- Typical PR (warm cache): scoped mutants × only-mapped-fast-tests ≈ **2–6 min**.
- P95 ≤ 10 min **by construction**: 8-minute budget + ~2 min env setup; the workflow
  job carries `timeout-minutes: 20` as defense in depth.
- Out-of-scope PR: env setup + explicit in-script skip, no mutation run.
- Cold cache (first run / lockfile bump): stats collection + tree-wide generation eat
  into the budget → a truncated-but-honest partial report, never an unbounded run; the
  next nightly re-warms the cache.

## 5. Rejected alternatives

- **Whole-package PR mutation with a bigger box / more parallelism**: spends the
  budget re-proving mutants the PR cannot have affected; the diff is the only scope a
  PR verdict is *about*.
- **Function-level diff scoping (changed functions only)**: strictly tighter than
  file-level, but needs diff-hunk→AST mapping the file-level pattern already
  approximates within budget; revisit only if file-scoped runs blow the budget.
- **`mutate_only_covered_lines`**: adds a coverage collection run to the loop the
  config just removed coverage from; the per-function test mapping already skips
  signal-less mutants (they surface as "no tests" instead — more honest).
- **Required-check mutation floor on PRs**: a floor on partial, diff-local data would
  either block merges on noise or be permanently lax; the floor belongs to the nightly
  full campaign (the ratchet), the PR leg is feedback.
