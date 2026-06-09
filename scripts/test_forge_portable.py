#!/usr/bin/env python3
"""Forge-portability gate (CI-R1 item 18 / CI-R9 / DELIV-R8).

Cites: CI-R9 / DELIV-R8 (the engine ships on BOTH GitHub Actions and a self-hosted
Forgejo forge; the two workflow files MUST stay in lock-step) and CI-R0 (the Justfile
is the SINGLE source of truth — the workflow YAMLs are thin schedulers that invoke ONLY
`just <recipe>`, with ZERO gate logic in the YAML). This gate proves that portability
holds, mechanically, so a gate cannot be added to one forge and silently forgotten on
the other.

It enforces two things:

  (1) Recipe-set parity. Parse ``.github/workflows`` and ``.forgejo/workflows``, collect
      the SET of ``just <recipe>`` invocations referenced by every step's ``run:`` block,
      and assert the two sets are EQUAL. Any recipe present on one forge but not the other
      fails the gate with a symmetric-difference report. (Frequency/ordering are NOT
      compared — only the set of recipes each forge exposes, which is what "both forges run
      the same gates" means; a recipe used twice on one forge and once on the other is fine.)

  (2) Release dry-run parity. Run ``just release`` once per forge mode
      (``FORGE_PROVIDER`` ∈ {github, forgejo}) under ``RELEASE_DRY_RUN=1`` and assert BOTH
      succeed touching NO network (CI-R10/CI-R12). This proves the SAME release recipe
      drives both forges with only the provider env differing — no per-forge code path.

Runnable directly and via ``just test-forge-portable``::

    uv run python scripts/test_forge_portable.py
    uv run python scripts/test_forge_portable.py --github .github/workflows \
        --forgejo .forgejo/workflows

Exit code is 0 only when the recipe sets match AND both dry-runs succeed; non-zero
otherwise (fail-closed — a portability gate that cannot prove parity MUST fail, never
silently pass). This is the SAME command CI runs (CI-R0: local command == CI command).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Default workflow directories (each forge keeps its scheduler under its own dot-dir).
_GITHUB_WORKFLOWS = ".github/workflows"
_FORGEJO_WORKFLOWS = ".forgejo/workflows"

# A `just <recipe>` invocation inside a step's `run:` block. The recipe name is the
# Justfile identifier grammar (lowercase + digits + hyphen). The `\bjust\s+` anchor skips
# bare `uv run …`, `pre-commit …`, `curl …`, etc., and the word boundary avoids matching
# substrings like `adjust`. Recipes that take args (none of ours do in CI) would still be
# captured by their leading name token.
_JUST_RECIPE = re.compile(r"(?<![\w-])just\s+([a-z][a-z0-9-]*)")

# A throwaway semver tag for the release dry-runs. `scripts/release.sh` requires a
# `vX.Y.Z`-shaped VERSION even under dry-run (it validates inputs before the network
# boundary); the Justfile `release` recipe forwards VERSION from the environment, so we
# inject a synthetic one here. Under RELEASE_DRY_RUN=1 it is only ever printed, never
# tagged or pushed (no network — CI-R10/CI-R12).
_DRY_RUN_VERSION = "v0.0.0-forge-portable-dryrun"


def _iter_workflow_files(workflows_dir: Path) -> list[Path]:
    """Return every ``*.yml`` / ``*.yaml`` workflow file in a forge's workflow dir, sorted."""
    files = sorted(
        p for p in workflows_dir.iterdir() if p.is_file() and p.suffix in {".yml", ".yaml"}
    )
    if not files:
        raise SystemExit(
            f"forge-portable: no workflow files found under {workflows_dir} — "
            "cannot prove parity, failing closed."
        )
    return files


def recipes_in_run_block(run_block: str) -> set[str]:
    """Extract the set of ``just <recipe>`` names referenced in one step's ``run:`` text.

    A single ``run:`` block may chain several recipes (e.g. ``just a && just b``) or span
    multiple lines; every ``just <recipe>`` token in it is collected.
    """
    return set(_JUST_RECIPE.findall(run_block))


def recipes_in_workflow_dir(workflows_dir: Path) -> set[str]:
    """Collect the set of ``just`` recipes invoked by every step in a forge's workflow dir.

    Parses each workflow YAML, walks ``jobs.<job>.steps[].run``, and unions the recipe
    names found. Non-``run`` steps (``uses:`` actions) contribute nothing — they are forge
    plumbing (checkout, uv setup, artifact upload), not gates.
    """
    found: set[str] = set()
    for path in _iter_workflow_files(workflows_dir):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise SystemExit(f"forge-portable: {path} is not a YAML mapping — cannot parse.")
        for job in (doc.get("jobs") or {}).values():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if isinstance(step, dict) and isinstance(step.get("run"), str):
                    found |= recipes_in_run_block(step["run"])
    return found


def _format_recipe_set(label: str, recipes: Iterable[str]) -> str:
    """Render a sorted recipe set for the diff report."""
    items = sorted(recipes)
    return f"{label} ({len(items)}): {', '.join(items) if items else '<none>'}"


def assert_recipe_parity(github_dir: Path, forgejo_dir: Path) -> set[str]:
    """Assert both forges expose the IDENTICAL set of ``just`` recipes (CI-R9/DELIV-R8).

    Returns the shared recipe set on success; raises ``SystemExit`` with a symmetric-
    difference report on any divergence (fail-closed).
    """
    github = recipes_in_workflow_dir(github_dir)
    forgejo = recipes_in_workflow_dir(forgejo_dir)
    if github == forgejo:
        print(
            "forge-portable: recipe-set parity OK — both forges expose "
            f"{len(github)} identical `just` recipes."
        )
        return github

    only_github = github - forgejo
    only_forgejo = forgejo - github
    lines = [
        "forge-portable: FAIL — the GitHub and Forgejo workflows expose DIFFERENT "
        "`just` recipe sets (CI-R9/DELIV-R8).",
        "  A gate must be present on BOTH forges or neither.",
        f"  only in GitHub  ({github_dir}): {sorted(only_github) or '<none>'}",
        f"  only in Forgejo ({forgejo_dir}): {sorted(only_forgejo) or '<none>'}",
        "  " + _format_recipe_set("github", github),
        "  " + _format_recipe_set("forgejo", forgejo),
    ]
    raise SystemExit("\n".join(lines))


def _run_release_dry_run(forge_provider: str) -> None:
    """Run ``just release`` once for one forge under RELEASE_DRY_RUN=1; raise on failure.

    Asserts the release recipe drives the forge with NO network (CI-R10/CI-R12) and that
    only ``FORGE_PROVIDER`` changes between forges — no per-forge code path. A synthetic
    VERSION is injected because ``release.sh`` validates it before the network boundary.
    """
    env = {
        **os.environ,
        "RELEASE_DRY_RUN": "1",
        "FORGE_PROVIDER": forge_provider,
        "VERSION": _DRY_RUN_VERSION,
    }
    print(f"forge-portable: dry-run release on {forge_provider} (RELEASE_DRY_RUN=1)")
    # Fixed argv; the provider is from a closed set and `just` resolves from PATH (as in CI).
    result = subprocess.run(
        ["just", "release"],  # noqa: S607
        cwd=_REPO_ROOT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"forge-portable: FAIL — `just release` dry-run for FORGE_PROVIDER="
            f"{forge_provider} exited {result.returncode} (expected 0, CI-R10/CI-R12)."
        )


def assert_release_dry_runs() -> None:
    """Run the github + forgejo release dry-runs and assert BOTH succeed (CI-R10/CI-R12)."""
    for forge_provider in ("github", "forgejo"):
        _run_release_dry_run(forge_provider)
    print("forge-portable: both release dry-runs (github + forgejo) succeeded — no network.")


def _resolve_dir(raw: str) -> Path:
    """Resolve a workflow-dir argument against the repo root and assert it exists."""
    path = Path(raw)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.is_dir():
        raise SystemExit(f"forge-portable: workflow directory not found: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    """Assert recipe-set parity, then run both release dry-runs; exit non-zero on mismatch."""
    parser = argparse.ArgumentParser(
        description="Forge-portability gate: equal `just` recipe sets + dual release dry-runs."
    )
    parser.add_argument(
        "--github",
        default=_GITHUB_WORKFLOWS,
        help="GitHub Actions workflow directory (default: .github/workflows).",
    )
    parser.add_argument(
        "--forgejo",
        default=_FORGEJO_WORKFLOWS,
        help="Forgejo Actions workflow directory (default: .forgejo/workflows).",
    )
    parser.add_argument(
        "--skip-release-dry-run",
        action="store_true",
        help="Only check recipe-set parity; skip the two `just release` dry-runs.",
    )
    args = parser.parse_args(argv)

    assert_recipe_parity(_resolve_dir(args.github), _resolve_dir(args.forgejo))
    if not args.skip_release_dry_run:
        assert_release_dry_runs()
    print("forge-portable: PASS — recipe sets match and the release path is forge-portable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
