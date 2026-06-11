#!/usr/bin/env python3
"""T-MUT mutation-testing gate (issue #35, ADR 0006) — diff-scoped, time-boxed, cached.

One implementation drives both legs (the workflow YAMLs stay thin schedulers, CI-R0):

``--leg pr`` — the ADVISORY pull-request leg. Mutation testing is inherently
expensive (every surviving mutant is a test session), so the PR loop gets a
bounded, scoped slice and the nightly leg keeps the full campaign:

  1. DIFF SCOPING (issue #35 item 1): only mutants of files actually changed
     against the merge base run, selected via mutmut's mutant-name patterns
     (``wattwise_core.analytics.forms.x*``). A PR touching one module mutates
     that module, not the whole package.
  2. TIME BOX (item 2): a hard wall-clock budget (``WW_MUT_BUDGET_SECONDS``,
     default 480s). On expiry mutmut gets SIGINT — it stops workers and keeps
     every verdict already recorded — and the report says HONESTLY how many of
     the scoped mutants completed within budget. Partial information is fine
     for an advisory leg; an unbounded advisory job is not.
  3. CHEAP KILL SIGNAL (item 3): mutmut 3 maps tests to mutated functions and
     runs ONLY those tests per mutant (fastest-first, ``-x``); the pyproject
     ``[tool.mutmut]`` selection restricts the signal to the fast offline
     tiers and disables coverage instrumentation inside the mutant loop.
  4. WARM CACHE (item 4): mutmut persists generated mutants, verdicts and the
     test-to-function stats under ``mutants/``; CI restores that directory.
     mutmut's cache validity is MTIME-based and a fresh checkout resets every
     mtime, so this script first restores each tracked file's mtime to its
     last-commit time, then force-touches the PR-changed files — unchanged
     files keep their cached artifacts, changed files are re-mutated.
  5. IN-SCRIPT PATH FILTER (item 5): a PR with no mutable source change skips
     with an explicit report — the condition is code on BOTH forges (CI-R9),
     not per-forge YAML that can silently rot.

Survivors NEVER fail the PR leg (advisory = cannot block the merge); only an
infrastructure failure (clean tests broken, mutmut crash) exits non-zero.

``--leg full`` — the nightly ENFORCING leg: the whole campaign (incremental on
the nightly cache), a larger budget, and a mutation-score floor over the
correctness-critical packages (analytics + ingestion adapters, same scope as
the 95% coverage floor). ``WW_MUT_FLOOR`` is the ratchet (percent, default 0
until the first nightly baseline lands — raise it, never lower it).

Both legs write ``reports/mutation-<leg>.{md,json}`` (retained artifacts,
CI-R6) and append the markdown to ``$GITHUB_STEP_SUMMARY`` when present.
Runnable locally exactly as CI runs it (CI-R0): ``just test-mut-pr``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections import Counter, deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Verdict classification — mirrors mutmut 3.6's status_by_exit_code. A timeout
# or segfault under mutation means the mutant CHANGED observable behaviour, so
# both count as detected; "suspicious" (unexpected exit code) is counted as NOT
# detected so noise can never inflate the score. None = not checked (budget cut).
_DETECTED = {
    1: "killed",
    3: "killed",
    -24: "timeout",
    24: "timeout",
    152: "timeout",
    36: "timeout",
    255: "timeout",
    -11: "segfault",
    -9: "segfault",
    37: "caught by type check",
}
_UNDETECTED = {0: "survived"}
_NO_SIGNAL = {5: "no tests", 33: "no tests", 34: "skipped", 2: "interrupted"}

_MUTANTS_DIR = Path("mutants")
_REPORT_DIR = Path("reports")

# Committed ratchet baseline (TIER-R6): per-package mutation-score FLOORS live in DATA
# (mutation-floors.toml at the repo root), never hardcoded here (CFG-R1a). Keys are dotted
# package prefixes (e.g. "wattwise_core.analytics"); values are fractions in [0, 1]. The
# floors are the enforced scope of the full leg — the same correctness-critical packages
# that carry the coverage floor (DOD-R1). They only ever RATCHET UP; the nightly enforcing
# leg fails if any package drops below its committed floor (tests/unit/test_mutation_floors.py
# guards the ratchet-up-only invariant).
_FLOORS_PATH = _REPO_ROOT / "mutation-floors.toml"


def load_floors(path: Path = _FLOORS_PATH) -> dict[str, float]:
    """Read the committed per-package ratchet floors (fractions in [0, 1]) from data."""
    with path.open("rb") as f:
        floors = tomllib.load(f).get("floors", {})
    return {pkg: float(score) for pkg, score in floors.items()}


# "nothing matches" marker mutmut prints when patterns select zero mutants
# (e.g. the PR only touched constants — files with no mutatable functions).
_NO_MATCH_MARKER = "Filtered for specific mutants, but nothing matches"


def _mutmut_config() -> tuple[list[str], list[str]]:
    """Read source_paths + do_not_mutate from pyproject [tool.mutmut] (single source of truth)."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f).get("tool", {}).get("mutmut", {})
    return list(config.get("source_paths", ["src"])), list(config.get("do_not_mutate", []))


def _git(*args: str) -> str:
    # Fixed binary resolved from PATH (as in CI), repo-internal args only.
    return subprocess.run(  # noqa: S603
        ["git", "-c", "core.quotePath=false", *args],  # noqa: S607
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _tracked_files() -> set[str]:
    return {line for line in _git("ls-files").splitlines() if line}


def _resolve_base(base_ref: str) -> str:
    """Resolve the diff base, fetching it once if the checkout is too shallow."""
    for attempt in (1, 2):
        try:
            return _git("merge-base", base_ref, "HEAD").strip()
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            remote, _, branch = base_ref.partition("/")
            subprocess.run(  # noqa: S603
                ["git", "fetch", "--quiet", remote or "origin", branch or "main"],  # noqa: S607
                cwd=_REPO_ROOT,
                check=False,
            )
    raise AssertionError("unreachable")


def _changed_files(base_ref: str) -> set[str]:
    """Files changed against the merge base, plus local tracked modifications (CI-R0: the
    local run sees the same scope a contributor is editing)."""
    base = _resolve_base(base_ref)
    changed = {line for line in _git("diff", "--name-only", base, "HEAD").splitlines() if line}
    changed |= {line for line in _git("diff", "--name-only", "HEAD").splitlines() if line}
    return changed


def _restore_commit_mtimes(tracked: set[str]) -> None:
    """Set every tracked file's mtime to its last-commit time (one history walk).

    mutmut decides "is this cached mutant current?" by comparing the source
    file's mtime against the cached mutant file's mtime — and a fresh CI
    checkout stamps every file with clone time, which would silently invalidate
    the whole restored cache. ``-m --first-parent`` attributes a merged PR's
    files to the merge time, so anything merged AFTER the cache snapshot is
    correctly seen as newer than it.
    """
    remaining = set(tracked)
    proc = subprocess.Popen(
        [  # noqa: S607
            "git",
            "-c",
            "core.quotePath=false",
            "log",
            "-m",
            "--first-parent",
            "--pretty=format:#%ct",
            "--name-only",
        ],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("git log produced no stdout pipe")
    timestamp = None
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                timestamp = int(line[1:])
                continue
            if timestamp is not None and line in remaining:
                remaining.discard(line)
                os.utime(_REPO_ROOT / line, (timestamp, timestamp))
                if not remaining:
                    break
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def _prepare_cache_for_diff(changed: set[str], tracked: set[str]) -> None:
    """Make the restored ``mutants/`` cache mtime-consistent with THIS pull request."""
    _restore_commit_mtimes(tracked)
    now = time.time()
    for name in changed:
        path = _REPO_ROOT / name
        stale_copy = _REPO_ROOT / _MUTANTS_DIR / name
        if path.exists():
            # Force "source newer than cached mutant" so mutmut re-mutates it and
            # resets its verdicts even if commit times race the cache snapshot.
            os.utime(path, (now, now))
            if not name.endswith(".py") and stale_copy.is_file():
                # mutmut only re-copies non-Python files when the copy is absent;
                # drop the stale copy so e.g. a changed defaults.toml is refreshed.
                stale_copy.unlink()
        elif stale_copy.is_file():
            # Deleted source must not linger importable inside the mutated tree.
            stale_copy.unlink(missing_ok=True)
            Path(str(stale_copy) + ".meta").unlink(missing_ok=True)


def _mutable(files: set[str], source_paths: list[str], do_not_mutate: list[str]) -> list[str]:
    """The subset of *files* mutmut would mutate (mirrors its should_mutate filter)."""
    roots = tuple(p.rstrip("/") + "/" for p in source_paths)
    return sorted(
        f
        for f in files
        if f.endswith(".py")
        and f.startswith(roots)
        and not any(fnmatch.fnmatch(f, pattern) for pattern in do_not_mutate)
    )


def _pattern_for(path: str) -> str:
    """Changed file -> mutmut mutant-name pattern (mutant keys always start with 'x')."""
    dotted = path.removesuffix(".py").replace(os.sep, ".").removeprefix("src.")
    dotted = dotted.removesuffix(".__init__")
    return f"{dotted}.x*"


def _run_mutmut(patterns: list[str], budget_s: int) -> tuple[int, bool, str]:
    """Run ``mutmut run`` under a wall-clock budget; return (exit_code, timed_out, output_tail).

    On budget expiry mutmut receives SIGINT: its KeyboardInterrupt path stops
    the worker children and every verdict recorded so far stays on disk — that
    is what makes the partial report honest rather than lossy.
    """
    cmd = [sys.executable, "-m", "mutmut", "run"]
    max_children = os.environ.get("WW_MUT_MAX_CHILDREN")
    if max_children:
        cmd += ["--max-children", max_children]
    cmd += patterns
    print(f"mutation-gate: {' '.join(cmd)}  (budget {budget_s}s)", flush=True)

    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if proc.stdout is None:
        raise RuntimeError("mutmut produced no stdout pipe")
    stdout = proc.stdout
    tail: deque[bytes] = deque(maxlen=400)

    def _pump() -> None:
        while chunk := stdout.read(4096):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            tail.append(chunk)

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    timed_out = False
    try:
        proc.wait(timeout=budget_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(
            f"\nmutation-gate: budget of {budget_s}s exhausted — stopping mutmut "
            "(verdicts recorded so far are kept).",
            flush=True,
        )
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    pump.join(timeout=10)
    return proc.returncode or 0, timed_out, b"".join(tail).decode(errors="replace")


def _collect_verdicts(scope_paths: list[str]) -> tuple[Counter[str], list[str], int]:
    """Aggregate per-mutant verdicts for *scope_paths* from mutants/<path>.meta.

    Returns (status counts, surviving mutant names, total mutants in scope).
    """
    counts: Counter[str] = Counter()
    survivors: list[str] = []
    total = 0
    for path in scope_paths:
        meta_path = _REPO_ROOT / _MUTANTS_DIR / (path + ".meta")
        if not meta_path.exists():
            continue
        exit_code_by_key = json.loads(meta_path.read_text())["exit_code_by_key"]
        for mutant_name, exit_code in exit_code_by_key.items():
            total += 1
            if exit_code is None:
                counts["not checked"] += 1
            elif exit_code in _DETECTED:
                counts[_DETECTED[exit_code]] += 1
            elif exit_code in _UNDETECTED:
                counts["survived"] += 1
                survivors.append(mutant_name)
            elif exit_code in _NO_SIGNAL:
                counts[_NO_SIGNAL[exit_code]] += 1
            else:
                counts["suspicious"] += 1
    return counts, survivors, total


def _score(counts: Counter[str]) -> float | None:
    """Detected / (detected + undetected), in percent; None when nothing measurable ran."""
    detected = sum(counts[s] for s in ("killed", "timeout", "segfault", "caught by type check"))
    undetected = counts["survived"] + counts["suspicious"]
    if detected + undetected == 0:
        return None
    return 100.0 * detected / (detected + undetected)


def _write_report(leg: str, payload: dict[str, object], markdown: str) -> None:
    (_REPO_ROOT / _REPORT_DIR).mkdir(exist_ok=True)
    (_REPO_ROOT / _REPORT_DIR / f"mutation-{leg}.json").write_text(json.dumps(payload, indent=2))
    (_REPO_ROOT / _REPORT_DIR / f"mutation-{leg}.md").write_text(markdown)
    print(markdown, flush=True)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a") as f:
            f.write(markdown + "\n")


def _format_markdown(
    leg: str,
    scope_label: str,
    counts: Counter[str],
    survivors: list[str],
    total: int,
    timed_out: bool,
    budget_s: int,
    score: float | None,
) -> str:
    checked = total - counts["not checked"]
    lines = [f"## T-MUT ({leg} leg, advisory)" if leg == "pr" else f"## T-MUT ({leg} leg)"]
    lines.append(f"Scope: {scope_label}")
    if total == 0:
        lines.append("No mutants in scope.")
        return "\n".join(lines) + "\n"
    if timed_out:
        budget_note = (
            f"budget of {budget_s}s exhausted — measured on the {checked} of "
            f"{total} mutants completed within budget"
        )
    elif checked < total:
        budget_note = f"mutmut stopped early — only {checked} of {total} mutants checked"
    else:
        budget_note = f"all {total} scoped mutants completed"
    lines.append(f"Coverage of scope: {budget_note}.")
    if score is None:
        lines.append("Mutation score: n/a (no mutant in scope had a usable test signal).")
    else:
        detected_statuses = ("killed", "timeout", "segfault", "caught by type check")
        detected = sum(counts[s] for s in detected_statuses)
        lines.append(
            f"**Mutation score: {score:.1f}%** (detected {detected}, "
            f"survived {counts['survived']}, suspicious {counts['suspicious']})"
        )
    if counts["no tests"]:
        lines.append(f"Mutants with NO test signal (coverage gap, not score): {counts['no tests']}")
    if survivors:
        lines.append("")
        lines.append(
            "<details><summary>Surviving mutants "
            f"({len(survivors)}; inspect with `uv run mutmut show <name>`)</summary>"
        )
        lines.append("")
        lines.extend(f"- `{name}`" for name in survivors[:50])
        if len(survivors) > 50:
            lines.append(f"- … and {len(survivors) - 50} more (see reports/mutation-{leg}.json)")
        lines.append("")
        lines.append("</details>")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T-MUT mutation gate (issue #35, ADR 0006).")
    parser.add_argument(
        "--leg",
        choices=("pr", "full"),
        required=True,
        help="pr = advisory diff-scoped time-boxed; full = nightly enforcing.",
    )
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("WW_MUT_BASE_REF", "origin/main"),
        help="Merge-base ref for the pr leg (default: origin/main).",
    )
    args = parser.parse_args(argv)
    os.chdir(_REPO_ROOT)

    default_budget = 480 if args.leg == "pr" else 10800
    budget_s = int(os.environ.get("WW_MUT_BUDGET_SECONDS", default_budget))
    source_paths, do_not_mutate = _mutmut_config()
    tracked = _tracked_files()

    if args.leg == "pr":
        changed = _changed_files(args.base_ref)
        scope_paths = _mutable(changed, source_paths, do_not_mutate)
        if not scope_paths:
            message = (
                "## T-MUT (pr leg, advisory)\nSkipped: this PR touches no mutable "
                "source file — nothing to mutate (in-script path filter, "
                "issue #35 item 5).\n"
            )
            _write_report("pr", {"leg": "pr", "skipped": True, "changed": sorted(changed)}, message)
            return 0
        _prepare_cache_for_diff(changed, tracked)
        patterns = [_pattern_for(p) for p in scope_paths]
        scope_label = f"{len(scope_paths)} changed file(s) vs merge-base of {args.base_ref}"
    else:
        _restore_commit_mtimes(tracked)
        patterns = []
        scope_paths = _mutable(tracked, source_paths, do_not_mutate)
        scope_label = f"full campaign over {source_paths}"

    exit_code, timed_out, output_tail = _run_mutmut(patterns, budget_s)

    if exit_code != 0 and not timed_out and _NO_MATCH_MARKER in output_tail:
        message = (
            "## T-MUT (pr leg, advisory)\nSkipped: the changed files contain no "
            "mutatable functions (constants/declarations only).\n"
        )
        _write_report(args.leg, {"leg": args.leg, "skipped": True, "scope": scope_paths}, message)
        return 0

    counts, survivors, total = _collect_verdicts(scope_paths)
    score = _score(counts)
    markdown = _format_markdown(
        args.leg, scope_label, counts, survivors, total, timed_out, budget_s, score
    )
    payload: dict[str, object] = {
        "leg": args.leg,
        "scope": scope_paths,
        "budget_seconds": budget_s,
        "timed_out": timed_out,
        "mutmut_exit_code": exit_code,
        "counts": dict(counts),
        "total_mutants": total,
        "score_percent": score,
        "survivors": survivors,
    }
    _write_report(args.leg, payload, markdown)

    if exit_code != 0 and not timed_out and total == counts["not checked"]:
        # mutmut died before producing any verdict (clean tests failed, bad config,
        # import mismatch): that is an infrastructure failure, never a silent green.
        print(
            f"mutation-gate: mutmut failed (exit {exit_code}) before any verdict — "
            "failing honestly.",
            file=sys.stderr,
        )
        return 1

    if args.leg == "full":
        return _enforce_floors(scope_paths)

    return 0


def _floor_package(path: str, packages: list[str]) -> str | None:
    """Map a scope file path (src/wattwise_core/analytics/forms.py) to its floor package
    (wattwise_core.analytics), choosing the LONGEST matching prefix so a nested floored
    package (…ingestion.adapters) wins over a broader one if both were ever declared."""
    dotted = path.removesuffix(".py").replace(os.sep, ".").removeprefix("src.")
    dotted = dotted.removesuffix(".__init__")
    matches = [pkg for pkg in packages if dotted == pkg or dotted.startswith(pkg + ".")]
    return max(matches, key=len) if matches else None


def _enforce_floors(scope_paths: list[str]) -> int:
    """Nightly enforcing leg: gate each floored package's measured mutation score against
    its committed ratchet floor (mutation-floors.toml). The ratchet only goes up; a drop
    below floor fails the build. Floors are DATA (CFG-R1a), fractions in [0, 1]; the gate
    reports in percent. A package absent from the floors file is reported, never enforced.

    WW_MUT_FLOOR (percent), if set > 0, is an extra GLOBAL minimum applied to the union of
    all floored packages — a coarse ratchet kept for backward compatibility with the
    pre-per-package contract; the per-package floors are the primary gate.
    """
    floors = load_floors()
    packages = list(floors)
    by_package: dict[str, list[str]] = {pkg: [] for pkg in packages}
    for path in scope_paths:
        pkg = _floor_package(path, packages)
        if pkg is not None:
            by_package[pkg].append(path)

    below_floor = False
    union_paths: list[str] = []
    for pkg in packages:
        floor = floors[pkg]
        union_paths.extend(by_package[pkg])
        counts, _, total = _collect_verdicts(by_package[pkg])
        pct = _score(counts)  # detected/(detected+undetected) in percent, or None
        score_frac = None if pct is None else pct / 100.0
        shown = "n/a" if score_frac is None else f"{score_frac:.3f}"
        verdict = "PASS"
        if score_frac is not None and score_frac < floor:
            verdict = "FAIL"
            below_floor = True
        print(
            f"mutation-gate: [{verdict}] {pkg}: score {shown} "
            f"(floor {floor:.2f}) over {total} mutants."
        )

    # Backward-compatible coarse global floor (percent) over the union of floored packages.
    global_floor = float(os.environ.get("WW_MUT_FLOOR", "0"))
    if global_floor > 0:
        union_counts, _, union_total = _collect_verdicts(union_paths)
        union_score = _score(union_counts)
        print(
            f"mutation-gate: global floored-union score "
            f"{'n/a' if union_score is None else f'{union_score:.1f}%'} "
            f"over {union_total} mutants (WW_MUT_FLOOR {global_floor:.1f}%)."
        )
        if union_score is not None and union_score < global_floor:
            below_floor = True

    if below_floor:
        print(
            "mutation-gate: FAIL — a package dropped below its committed ratchet floor "
            "(mutation-floors.toml); the ratchet only goes up.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
