# Issue draft — could not be filed via API (GitHub App lacks issues:write); please open manually

**Title:** Grounding gate fails open: number rewrite targets the wrong occurrence, shipping non-canonical values with decision=proceed

---

While reading through the grounding path I hit something that I believe breaks the core GROUND-R7 guarantee, and it reproduces deterministically against `ground()` as it is on `main`.

## What happens

When two NUMBER claims in one draft happen to share the same numeric token (e.g. both say "100"), the second claim's canonical rewrite lands on the **first** claim's span. The athlete ends up seeing both numbers wrong — effectively swapped across metrics — and the gate still returns `proceed`, so the run finalizes COMPLETED with citations attached.

Minimal repro, no fixtures needed:

```python
from wattwise_core.agent.grounding import ground
from wattwise_core.agent.contracts import Claim, ClaimKind

class Ev:
    def __init__(self, vals): self.vals = vals
    def metric_snapshot(self, metric, ref): return self.vals.get(metric)
    def url_allowed(self, url): return False
    async def metric_value(self, metric, as_of=None): return self.vals.get(metric)

draft = "Your CTL is 100 and your weekly TSS was 100."
claims = [
    Claim(kind=ClaimKind.NUMBER, text="CTL is 100", metric="ctl", value=100.0),
    Claim(kind=ClaimKind.NUMBER, text="TSS was 100", metric="tss", value=100.0),
]
res = ground(draft, claims, Ev({"ctl": 100.0, "tss": 98.0}), [])
print(res.decision)       # GroundDecision.PROCEED
print(res.scrubbed_text)  # "Your CTL is 98 and your weekly TSS was 100."
```

Canonical values are CTL=100, TSS=98. The published text says CTL=98 and TSS=100 — both wrong, and both claims carry `grounded` verdicts. The TSS claim grounds because 100 vs 98 is inside the 2% tolerance band, then GROUND-R7 wants to publish the canonical "98" — but the rewrite hits the CTL's "100" instead of the TSS's.

It's not limited to within-tolerance cases. With a contradicted claim, the contradicted figure survives in the published body:

```python
draft2 = "Your CTL is 100 and your ATL is 100."   # canonical ctl=100, atl=80
# → scrubbed_text == "Your CTL is 80 and your ATL is 100."
```

The model's contradicted "100" for ATL is still there (GROUND-R9 says contradicted is never published), and the correct CTL got corrupted to 80. The decision is `regenerate` in that case, but `grounded_text` is what the finalize path falls back to when the redraft budget runs out — and a model that re-emits the same draft will loop on the same corruption.

There's also a substring variant: the token search isn't word-bounded, so a claim token `102` rewrites inside an unrelated `1029` (`"Over 1029 TSS…"` → `"Over TSS…"`). That one at least fails closed, but it mangles the text.

## Why

Two pieces interact:

1. `_apply_number_scrub` (`src/wattwise_core/agent/grounding_match.py`, ~line 137) extracts the claim's numeric token and then does `idx = text.find(token)` — first occurrence in the **whole draft**, not anchored to the claim's span, with no word boundary.
2. The numeric coverage sweep that's meant to be the extraction-independent backstop checks coverage by string membership — `if token in grounded_numbers` (`src/wattwise_core/agent/grounding_sweep.py`, ~line 135), where `grounded_numbers` is a flat set of published display strings built in `grounding.py` (~lines 119, 132–134). So the leftover wrong token (the "100" where TSS's canonical 98 should be) is "covered" by the *other* claim's published value, survives the sweep, and the decision stays `proceed`.

## Suggested fix

- **Anchor the rewrite to the claim's own span.** Locate `claim.text` in the draft first, and replace the numeric token *within* that span only. The existing comment about wording drift still applies, so fall back to a `\b`-bounded regex search for the token when `claim.text` isn't found verbatim — but never a bare `str.find` over the whole draft.
- **Make coverage positional, not string-based.** Track the character ranges actually rewritten/verified, and have both subsequent rewrites and `scrub_uncovered_numbers` respect them instead of the `token in grounded_numbers` set membership. With positional coverage, the leftover "100" in the example is no longer "covered", the sweep removes it, and `_downgrade_for_sweep` kicks in — which restores fail-closed behavior even when the anchoring heuristic misses.
- **Word-bound the token match** (`\b` / digit lookarounds) so a token can never match inside a longer number (the 102-in-1029 case).

Regression tests that would pin this: a draft with two claims sharing a numeric token (the CTL/TSS case above must either publish both canonical values correctly or not `proceed`), and a claim token that is a substring of a larger number in the same draft. `tests/unit/test_grounding.py` currently has no case where two claims share a token, which is presumably how this slipped through — single-claim drafts behave correctly.

Worth stressing why I'd rank this critical: for this product the worst outcome isn't refusing to answer, it's confidently presenting a wrong number as canonical with a citation attached — and this is the one code path whose entire job is to make that impossible.
