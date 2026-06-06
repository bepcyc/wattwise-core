# Contributing to wattwise-core

`wattwise-core` is the open-source endurance-analytics + coaching-agent engine of
the `wattwise` family, licensed under **Apache-2.0** and authored/maintained by
**Viacheslav Rodionov** <viacheslav.rodionov@gmail.com>. The canonical home is
**GitHub**; the project is also self-hostable on **Forgejo** with no functional
degradation (the same gates, the same release).

This document is the contributor contract (DELIV-R3). The binding gate is always
**CI** (the required checks of CI-R1) — everything below exists to make a green
CI the path of least resistance.

---

## 1. One-command bootstrap (BOOT-R1)

Clone, then bring up a working instance and a green offline suite with a single
documented command. The only thing you supply by hand is secrets, via the
environment (BOOT-R4) — never a committed file.

```bash
git clone https://github.com/wattwise/wattwise-core.git
cd wattwise-core
just bootstrap
```

`just bootstrap` installs the pinned toolchain (`uv sync --frozen`), applies the
versioned ORM migrations from empty (BOOT-R2), and starts the engine via uvicorn
against `WATTWISE_DATABASE_DSN`. With no DSN set it defaults to a local SQLite
dev database (`sqlite+aiosqlite:///./.wattwise-dev.sqlite`) so you need **zero**
external services to get a health-serving instance. Point the env var at
PostgreSQL or MariaDB to run on those backends — only the DSN changes (BOOT-R3).

Secrets come from the environment / a secret manager only (BOOT-R4, SEC-R12).
See `.env.example` for the variable **names** (copy the shapes into your secret
manager; do not commit values). In `development` the engine boots without the
production secrets; in `staging`/`production` it fails closed if any are absent.

### The single interface: `just`

The **Justfile is the single source of truth** for every task — build, test,
lint, type-check, migrate, bootstrap, scan, release, deploy (RUN-R3.3, CI-R0).
CI (GitHub Actions and Forgejo Actions) is a thin scheduler that calls the exact
same `just` recipes, so **the command you run locally is the command CI runs**.
There is no CI-only logic. List the catalog with:

```bash
just            # or: just --list
```

> The Justfile is the developer/ops task runner, **not** a product CLI: there is
> no `coach` console or end-user runtime here (DELIV-R4).

---

## 2. Test-first is mandatory (QUAL-R1, QUAL-R7)

Every behavioral change is developed **test-first**, and this applies to external
contributors identically. The per-unit cycle is strict **red → green → refactor**:

1. **RED** — write an executable test that captures the requirement and **run
   it**. It MUST be observed to **fail for the right reason** (a missing/wrong
   behavior, not an import or syntax error) **before** any implementing code
   exists.
2. **GREEN** — write the minimum code that makes it pass.
3. **REFACTOR** — improve structure while keeping it green.

A test written after the fact and never observed to fail does **not** satisfy
this rule. A change that introduces behavior with no covering test is a gate
failure.

Conventions that the lint gate enforces mechanically:

- **Every test function carries a short docstring** stating the behavioral
  contract in plain English (not a restatement of the test name) — QUAL-R10b.
- **One tier marker per test** (`unit`, `property`, `golden`, `contract`,
  `integration`, `e2e`, ...) so a tier is selectable — TIER-R3.
- **English-only source**; `mypy --strict` clean; module ≤ 400 lines / function
  ≤ 60 lines (decompose, don't blanket-suppress) — QUAL-R8/R9/R11.

---

## 3. Run the gates locally before you open a PR

Run the deterministic, offline required checks with one command:

```bash
just gate
```

`gate` runs every deterministic offline gate: lint (+ AST/content/no-vendor-SQL/
arch lints), format-check, `mypy --strict`, commit-message lint, the fast test
tiers (unit/property/golden/contract/fuzz/logging), the recorded-mode agent eval,
the injection corpus, and the coverage gate. The service-backed tiers run via
their own recipes (they need a database / a built image):

| Recipe | What it gates | CI-R1 item |
|---|---|---|
| `just lint` | ruff + the code-craft AST / content / no-vendor-SQL / arch lints | 1, 14, 21 |
| `just type` | `mypy --strict`, zero errors | 2 |
| `just test-unit` / `-property` / `-golden` / `-contract` | offline tiers | 3 |
| `just test-fuzz` | bounded parser/decoder fuzzing | 16 |
| `just test-integration` | T-INT against an ephemeral master store | 4 |
| `just cov` | combined coverage ≥ 80% (analytics/adapters ≥ 95%) | 5 |
| `just eval` | agent eval thresholds (recorded mode) | 6 |
| `just test-inject` | prompt-injection corpus, no regression | 7 |
| `just scan` | secret scan + dependency/SCA scan | 8, 9 |
| `just sbom` | container image scan (no Critical) + SBOM | 10 |
| `just test-logging` | logging-contract (no log files, redaction) | 11 |
| `just test-e2e` | API-level E2E smoke | 12 |
| `just test-db-portable` | SQLite + PostgreSQL + MariaDB portability | 13 |
| `just lint-commits` | Conventional Commits | 15 |
| `just test-forge-portable` | GitHub/Forgejo recipe-set equality + dual dry-run | 18 |
| `just install-boot-check` | wheel builds, installs into a fresh env, boots | 20 |

### Pre-commit hooks (a fast local mirror, CI-R11)

Install the hooks once; they run a **fast subset** of CI on each commit (ruff
lint + format-check, `mypy`, a secret-pattern scan over the diff, hygiene, and a
commit-message check):

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

The hooks are a convenience, **not** a substitute for CI. They never run the slow
tiers or the image/SBOM/SCA scans. **CI remains the authoritative, blocking gate.**

---

## 4. Trunk-based development + Conventional Commits (QUAL-R12)

- **Trunk-based.** Integration targets a single long-lived `main`. Feature
  branches are short-lived (≤ 1 working day, never > 3), small, and merged via
  PR through the full CI-R1 gate. There are **no** gitflow `develop` or
  long-lived release branches. Incomplete work lands behind config feature-flags
  so `main` stays always-releasable.

- **Conventional Commits v1.0.** Every commit merged to `main` MUST follow the
  format: a type from
  `feat | fix | docs | style | refactor | perf | test | chore | ci | build`, an
  optional `(scope)`, a `:`, and a description; a breaking change is marked with
  `!` or a `BREAKING CHANGE:` footer. Examples:

  ```text
  feat(analytics): add W'balance differential model (Skiba 2012)
  fix(adapter-intervals): map RR samples to canonical hrv_method
  test(api): cover token issuance for delegated bot-link tokens
  ```

  `just lint-commits` is a fast required check; a non-conforming message fails
  the build. The semver tag and `CHANGELOG.md` are derived automatically from the
  conforming commit log by the release pipeline (`just changelog`).

---

## 5. The no-bypass rule (QUAL-R3, CI-R2)

A red required check **blocks merge structurally** (branch protection). There is
**no author self-bypass** of a failed required check. An exception, if ever
granted, is an auditable, reviewed maintainer action recorded on the PR — never a
quiet override. Capability breadth is never bought by lowering a gate threshold;
a threshold may only ratchet upward (ROAD-R5).

---

## 6. Scope boundaries (what belongs in this repo)

`wattwise-core` ships the **bare OSS engine** + schemas + a minimal example/
default config bundle (DELIV-R2). Proprietary IP — coach persona/voice, system/
agent prompts, named skills/playbooks, model-routing policy, metric-equivalence
thresholds, grounding-rules text — is **externalized to runtime config** and does
**not** live here; the engine embeds none of it inline. Out of OSS scope (and not
to be added to this repo): multi-user subscription/billing, a coach marketplace,
the web client, and the Telegram bot — those are the commercial `athload` layer
built additively on this engine. Keep the extension **seams** clean and green;
don't re-architect to add commercial behavior.

---

## 7. Forge portability (DELIV-R8)

GitHub is the primary forge; the project is equally self-hostable on Forgejo. The
two workflow files (`.github/workflows/ci.yml`, `.forgejo/workflows/ci.yml`) are
thin schedulers that reference an **identical set of `just` recipes** — verified
by `just test-forge-portable`. If you add a gate to one forge's workflow, add it
to the other; the portability check fails otherwise.

Thank you for contributing.
