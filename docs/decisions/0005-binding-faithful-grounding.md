# ADR 0005 — Binding-faithful grounding (issue #10): the sentence selects the cell

Status: accepted (2026-06-10). Source: issue #10 — the grounding gate verified each
NUMBER's **value** against the canonical `(metric, as_of)` cell named by the claim, but
that binding was extracted by the SAME model that wrote the draft. The defendant routed
its own cross-examination: a mis-bound claim ("your fatigue is 71" checked against `ctl`;
"your CTL is 71" checked at a cherry-picked past date; a metric-shaped sentence excused as
a user-request echo) verified cleanly and shipped with a citation. ADR 0004 §4 closed the
same trust family one level up (the claim-KIND label); the metric/as-of labels remained
model-certified.

## 1. Layer 1 — deterministic re-binding (proposed GROUND-R10), `grounding_binding.py`

Not another referee — an **authority inversion**. `BindingGuard.rebind` re-derives each
NUMBER claim's binding from its OWN sentence (the only text the athlete reads), through
the SAME config-loaded `MetricEquivalence` vocabulary the value verifier resolves with:

- a sentence naming exactly ONE canonical metric binds the claim to it (a wrong/missing
  extracted metric is overridden, never trusted);
- a sentence stating exactly ONE explicit ISO date pins `as_of` to it (closing the
  dated-sentence-with-absent-ref half of H2);
- a present-tense, undated sentence drops a stale extracted date, so the claim verifies
  against the LATEST value — the cherry-picked figure becomes CONTRADICTED and the
  ordinary GROUND-R7 machinery substitutes the TRUE value in place. A mis-extraction is
  thereby a **correction**, not a scrub: usefulness preserved, hallucination gone.

Residual fail-closed floor (`check_number`, enforced inside `ground`): an ambiguous
multi-metric sentence whose claim matches none of its labels is contradicted-class
(never publishable; no swap-by-guess), the strict no-label rule is config-gated
(`require_metric_label`, off by default — high over-scrub risk on free prose), and a
metric-shaped sentence can never ground as a request echo (`echo_blocked`). Rollout modes
`off | shadow | enforce` (`[agent.binding]`, CFG-R1a); SHADOW records would-be
rebinds/violations on the new `wattwise_agent_grounding_binding_events_total` counter
without applying them. Deterministic guards are precise, so the OSS default ENFORCES.

## 2. Layer 2 — decorrelated entailment gate (proposed GROUND-R11), `grounding_entailment.py`

What rules cannot enumerate (paraphrase, trend/direction words, multi-fact composition —
including the `COMPLEMENTARY` numberless free pass) is checked by a SECOND verifier that
shares no weights with the drafting model: a MiniCheck-class local grounded fact-checking
classifier (arXiv:2404.10774) scoring *sentence ⊨ fact sheet*, operationalizing the AIS
criterion (arXiv:2112.12870). The fact sheet is rendered by CODE
(`grounding_factsheet.py`) from the very snapshots the value gate verified plus the
turn's retrieved records and the athlete's own request (so a request echo is entailed,
not vetoed). The gate can only VETO: a failing sentence is removed and the decision is
forced off `proceed`; an unavailable verifier (the ML stack is an operator opt-in, never
a base dependency) degrades the run to the deterministic layers and is RECORDED
(`...entailment_unavailable_total`) — never silently open, never a crash. Disabled by
default (`[agent.entailment]`).

## 3. Layer 3 — split-conformal thresholds (proposed GROUND-R12), `grounding_conformal.py`

The gate's publication threshold carries a guarantee, not a vibe: per-example
max-exceedance over a labelled calibration artifact yields, per claim class, the standard
split-conformal upper quantile — under exchangeability, P(a new deliverable publishes an
unsupported sentence) ≤ α (Quach et al. arXiv:2306.10193; Cherian/Gibbs/Candès
arXiv:2406.09714; group-conditional per arXiv:2602.01285). Too little calibration data
fails closed to τ = 1.0 (nothing certifies); a malformed artifact fails the BOOT. Honest
caveat recorded: the guarantee assumes calibration/deployment exchangeability —
recalibration belongs in the release checklist.

Provenance pinning (landing-review condition, the QA-EVAL-R12 cassette-pin rule applied
to calibration): the artifact is a JSON object stamping `provenance.model_id`,
`provenance.claim_prompt_sha256`, and `provenance.dataset_version` alongside its
`records`. At load the stamp is checked against the configured verifier checkpoint and
the SHA-256 of the configured claim-extraction prompt (plus an optional exact
`calibration_dataset_version` config pin); a stale or unstamped artifact fails the boot
exactly like a malformed one — the conformal guarantee does not transfer across a
model/prompt change, so it must never silently appear to.

## 4. Extraction prompt (Claimify-informed)

`claim_system` now requires decontextualized, sentence-faithful claims (copy the metric
label as the sentence states it; attach a date ONLY when the sentence states one) —
extraction quality criteria per arXiv:2502.10855. Config content only (CFG-R1a).

## 5. Accepted residuals

- Month-name dates ("back on May 1") are not pinned deterministically (ambiguous without
  a year; "may" doubles as a modal) — the entailment layer covers them.
- The strict no-label rule ships off; numberless trend claims pass the deterministic
  layers and are caught only when the entailment gate is enabled.
- Offline eval (GROUND-R8) goldens for mis-binding live in the unit suite
  (`test_grounding_binding.py`); promotion into the recorded eval corpus is a follow-up.
- Complementary to ADR/PR #13 (record sufficiency): #13 gates on whether the RECORD is
  current/complete enough to advise; this gates on whether the SENTENCE means what the
  record says. No shared code paths.
