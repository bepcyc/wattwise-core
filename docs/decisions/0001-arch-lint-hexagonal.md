# 0001 — Arch-lint models ARCH-R1 layering as hexagonal import budgets

- Status: Accepted
- Date: 2026-06-06
- Requirements: ARCH-R1, ARCH-R3, ARCH-R21, ARCH-R22 (doc 10); CON-R2 (doc 80)

## Context

The custom architecture linter (`tools/lint/import_direction.py`, gated by CI-R1 item 14)
enforces two static invariants over every `import` in the engine:

1. **Inward-only layer imports (ARCH-R21 / ARCH-R1)** — dependencies must point toward the
   stable core; an outer layer may not import an inner one.
2. **No source-name branching (ARCH-R22 / ARCH-R2)** — no consumer may import a concrete,
   source-specific adapter module; adapters are selected through the registry/seam so
   "consumers never branch on source" (Principle A).

The initial implementation modeled the layers as a single linear rank taken straight from
doc 10's presentation labels (L2 adapters … L6 edge) and forbade any import where the
target's label was numerically higher than the importer's. That rule was wrong for one
required edge: **ARCH-R3 makes ingestion the ONLY writer to the canonical store**, so the
L3 ingestion service legitimately imports the L4 persistence layer. A naive
`target_label > importer_label → violation` rule flags that mandatory `L3 → L4` write edge
as an architecture breach, which would make the gate un-satisfiable for a spec-compliant
ingestion path.

## Decision

Model ARCH-R1 as a **hexagonal architecture** rather than a linear stack. The L4 canonical
store and L5 analytics are the **stable core**; both the ingestion side and the edge side
point *inward* toward that core. The linter therefore ranks each subpackage by an
**import budget** (the set of ranks a module may depend on), not by its doc-10 presentation
label:

| subpackage   | budget rank | role                                                        |
|--------------|-------------|-------------------------------------------------------------|
| `persistence`| 1           | L4 canonical store — the core; written by ingestion (ARCH-R3) |
| `analytics`  | 2           | L5 domain analytics — reads the store only                   |
| `ingestion`  | 3           | L3 ingestion/sync — the ONLY writer to the store (ARCH-R3)    |
| `api`, `agent`| 4          | L6 edge — reads analytics/store, triggers ingestion          |
| `ingestion/adapters` | 0   | L2 leaf producers — import only the rankless `domain` package |

A module may import only equal-or-lower-rank modules. The core has the smallest budget; the
edge the largest. This **permits the required `ingestion → persistence` write edge** while
still forbidding, for example, `analytics → api`, `persistence → analytics`, or an adapter
importing the store/analytics.

`domain` (the GBO canonical value types — closed enums GBO-R12, coverage descriptors) and
the cross-cutting packages (`config`, `security`, `observability`, `eval`, `testing`) carry
no rank and are importable from any layer, so an L4 ORM model importing `domain.enums` is not
a violation. The adapter-contract seam (`ingestion.base`) is likewise rankless, so an adapter
may implement it without importing the ranked store/analytics.

The source-name rule (ARCH-R22) is unchanged: a non-adapter module importing
`ingestion.adapters.<concrete>` is still a violation; only sibling adapters may import each
other.

## Consequences

- The mandatory `L3 ingestion → L4 store` write edge passes the gate; the ingestion service
  can write the canonical store as ARCH-R3 requires.
- The gate still catches real inversions (edge importing nothing inward-only, store importing
  analytics, adapter importing the store) and every source-name leak (ARCH-R22 / CON-R2).
- The rank table is the single place that encodes the hexagon; adding a layer is a one-line
  change there, not a rewrite of the rule.
- Per-import escapes remain possible with a justified `# noqa: import-direction` token, kept
  rare and reviewed (DELIV-R3).
