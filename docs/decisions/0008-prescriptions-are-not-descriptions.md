# ADR 0008 — Prescriptions are not descriptions (issue #25): a target is verified, not restated

Status: accepted (2026-06-15), safety slice landed; the simulation-grounding epic is #78.
Source: issue #25 — the grounding gate had exactly one theory of what a number is: a
**restatement of a past/present canonical fact**. But the product's headline deliverable,
the goal-aware PLAN (GBO-R38), is made of numbers of a different *type*: **prescriptions** —
future targets DERIVED from the record, not contained in it. Verifying the second as the
first is a type error with a dangerous failure mode: a build/taper week out of tolerance of
the only prescriptive attractor (`weekly_load_target = 7xCTL` maintenance) was `CONTRADICTED`
and rewritten **in place** toward maintenance — a recovery week of 320 silently loaded UP to
420 (+31%), the aggressive direction, then shipped through the approval surface as the
"grounded plan". Long-term the same wall made every progressive plan structurally
ungroundable: "keep doing what you're doing" was the only fully groundable plan.

This ADR records the **claim-type split** + the **decision-aware gate** that shipped now, and
fixes the forward **falsification contract** the #78 epic builds against.

## 1. The claim-type split (GROUND-R13), `grounding_numeric.py`

A `NUMBER` claim is now verified by its *type*, decided by the `Claim.prescriptive` flag the
extractor already carries (previously set only on `STATEMENT`s):

- **descriptive** (`prescriptive=false`, the default) — today's behaviour, which is correct
  *for that type*: verify against the canonical `(metric, as_of)` cell within tolerance,
  publish the canonical value in place (GROUND-R7). Unchanged.
- **prescriptive** (`prescriptive=true`, a future target) — dispatched to
  `_verify_prescription` **before any canonical lookup**. A prescription is **never** verified
  against, nor rewritten toward, a descriptive metric: "correcting" an instruction toward
  maintenance authors advice the gate never validated, in the aggressive direction for a
  taper. Until typed feasibility verification exists (#78) the only sayable prescription is an
  **echo of a number the athlete supplied in their own request** (the request's own
  constraint, cited `user_request`); everything else **fails closed** (scrubbed, GROUND-R3),
  and the non-`proceed` decision states the limitation at finalize. The binding guard's R10d
  veto (`echo_blocked`, ADR 0005 §1) still forbids a metric-shaped sentence from grounding as
  an echo.

The claim-extraction prompt (`claim_system`, CFG-R1a) now instructs the model to mark a
number `prescriptive=true` when it is a future target ("week 2: 450 TSS", "aim for 250 W")
and `false` when it restates the record ("your CTL is 71"). The model still only POINTS;
code decides the verdict (GROUND-R1).

Routing consequence: a scrubbed prescription is **not** a re-gatherable metric gap
(`_has_regatherable_metric_gap`) — re-gathering canonical evidence can never make a future
target sayable — so a plan of only prescriptions degrades honestly (`abstain`) instead of
burning the reflection budget re-planning against a wall it cannot clear.

## 2. The approval gate is decision-aware (GROUND-R9 / CKPT-R5), `graph.py`

`interrupt_gate` previously paused on `plan_requires_approval(state)` alone — the deliverable
TYPE — and never consulted the grounding decision. But the `ground` node writes
`grounded_text` on EVERY pass, so a non-`proceed` body (an `abstain` the grounder ruled
unpublishable, or a plan whose prescriptions were scrubbed) would be shipped as
`AWAITING_APPROVAL` and put to a human decision. The gate now requires `PROCEED` before
soliciting approval: a non-`proceed` plan run falls through to `finalize` and degrades like
every other deliverable (a grounder abstain is NOT approval). This is the decision-*ignoring*
pause residual that ADR 0004's decision-*driven*-status work did not cover.

## 3. Forward contract for #78 — falsify, don't predict (proposed GROUND-R14)

The epic verifies a prescription by **refutation against a typed feasibility envelope**, not
by prediction of outcome. The science forbids the predictive framing: the fitness-fatigue /
impulse-response model is statistically ill-conditioned (poor parameter identifiability,
overfitting fatigue terms; *Sci Rep* 2025, s41598-025-88153-7), so a point-prediction
simulator would be a second unvalidated author of prescriptions — the recursion #25 flagged.
A *sound critic* need not be as capable as the *planner* (LLM-Modulo, arXiv:2402.01817); that
asymmetry is what makes the verifier achievable.

Normative contract the epic must honour, so the citation kind is defined before code:

- a prescription publishes when no sound verifier can REFUTE it (not when one proves it will
  work), with stated residual uncertainty;
- the verifier runs the existing deterministic, seedable PMC forward model across a
  **plausible parameter band** and `CONTRADICTED`s only what is infeasible across the whole
  band; in-band ambiguity DEGRADES with the uncertainty stated, never a silent rewrite;
- a passing prescription earns a new citation kind **`feasibility_envelope`** (alongside
  `metric` / `user_request` / `name` / `url`), recording `model_id`, `parameter_band`, and the
  `binding_critique`, reproducible from `(canonical_state, prescription, model_id, band)` —
  preserving the invariant that every published number is certified by deterministic code;
- verifiers live in a **registry keyed by `(sport, goal_type)`** with a declared default and a
  rule-based floor; an unsupported pair abstains honestly rather than flattening to 7xCTL.

The verification target is **adherence to coaching principles** (progressive overload,
recovery/supercompensation, specificity, timeline), which is what survives the model's
parameter instability — not a predicted outcome.

## 4. Accepted residuals / scope boundary

- The decision-aware gate (§2) is **live, not a forward guard**: the multi-day PLAN deliverable
  ships (the `/v1` planning endpoint -> `engine.plan_deliverable(requires_approval=True)`), so
  the gate change actively governs approval. Concretely, since a target-bearing plan's
  prescriptive numbers now scrub: a plan that still grounds *something* (workout NAMEs,
  request-echoes) reaches the gate as `PROCEED` and pauses for approval with its numeric targets
  stripped; a plan whose only content was prescriptive numbers becomes `ABSTAIN` and **degrades
  at finalize instead of pausing** — correctly, since the grounder will not stand behind a gutted
  body. This is the agreed #25 interim, but it is a *visible* change to the approval feature, not
  a dormant one. (An earlier draft of this ADR called the gate "inert in Phase 1"; that was wrong
  — the stale `plan_requires_approval` docstring claimed Phase-1 ships no plan, which the live
  planning endpoint contradicts.)
- The recorded eval cassettes still capture descriptive claims, so the offline suite does not
  yet exercise the prescriptive path end-to-end; the plan goldens that assert "progressive
  targets survive grounding" and "a taper is never rewritten upward via simulation" land with
  #78 (the unit goldens in `test_grounding.py` cover the safety behaviour now). The prompt
  change re-stamped the cassette `prompt_sha256` pins (QA-EVAL-R12, metadata only).
- **ACWR stays orphaned, by design.** The safety envelope (#78) must not be built on the
  acute:chronic workload ratio — it is discredited as a causal/predictive injury metric
  (Impellizzeri et al. 2020, PMID 32502973); the persisted `daily_wellness.acwr` column stays
  unread. Ramp-rate bounds and Foster monotony/strain (PMID 9662690) enter as review flags,
  not injury predictions.
- **Strength / hypertrophy is a different verifier, not the same simulator** — it is not an
  impulse-response phenomenon; it is governed by volume landmarks (MEV/MAV/MRV) and
  progressive overload (ACSM progression-model position stand 2009). The registry (§3) exists
  precisely so endurance and strength can carry different models behind one interface.
- Coordinate with #10 (ADR 0005): its entailment verifier makes prescriptions WORSE if applied
  naively (nothing future is entailed by a record of the past), so the `PRESCRIPTION` type is
  the shared seam — a stronger *descriptive* verifier must not scrub progressive plans harder.
