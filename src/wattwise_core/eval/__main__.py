"""``python -m wattwise_core.eval`` — the CI-gated offline eval entrypoint (EVAL-R1/-R9).

The justfile / CI invoke ``python -m wattwise_core.eval run --mode=recorded
--scorecard=...`` to gate the build: it runs every suite over the versioned checked-in
datasets with the deterministic offline model, writes a machine-readable JSON scorecard
(EVAL-R9), and EXITS NON-ZERO when any suite metric falls below its hard threshold
(EVAL-R1: "CI MUST fail the build when suite metrics fall below thresholds"). ``run`` ALSO
enforces non-regression against the committed baseline scorecard (QA-EVAL-R7): beyond the
absolute thresholds it exits non-zero if any tracked suite metric drops below the stored
baseline — so a safety-suite regression fails the build even when the absolute score still
passes. The ``record`` subcommand is a deterministic confirmation that the recorded-response
fixtures (QA-EVAL-R9) load; in the OSS offline engine the datasets ARE the committed
fixtures. ``update-baseline`` regenerates the baseline artifact from a clean run (the
reviewed way to advance the floor after a deliberate, intended improvement).

Network-free and deterministic (TIER-R1): every suite runs in recorded-response mode only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from wattwise_core.eval.baseline import (
    baseline_tag_failures,
    compare_to_baseline,
    live_run_blocks_baseline,
    write_baseline,
)
from wattwise_core.eval.live import LiveRunReport, LiveStatus, LiveSuiteResult, classify_infra_text
from wattwise_core.eval.recorded_meta import stamp_recorded_datasets, verify_recorded_datasets
from wattwise_core.eval.runner import EvalMode, Scorecard, list_suites, run_suite


async def _run_all() -> list[Scorecard]:
    """Run every catalogued suite and return its scorecard (EVAL-R9)."""
    return [await run_suite(name, mode=EvalMode.RECORDED) for name in list_suites()]


def _write_scorecard(cards: list[Scorecard], path: Path) -> None:
    """Write the aggregate machine-readable scorecard artifact (EVAL-R9)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": all(c.passed for c in cards),
        "suites": [c.to_jsonable() for c in cards],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stale_cassette_failures() -> int:
    """The QA-EVAL-R12(a) static check: stale/missing cassette metadata fails the gate."""
    stale = verify_recorded_datasets()
    for line in stale:
        print(f"stale cassettes (QA-EVAL-R12(a)): {line}", file=sys.stderr)
    return len(stale)


def _run_live_smoke() -> list[LiveSuiteResult]:
    """Run the env-gated live smoke (the ``llm`` pytest tier) and CLASSIFY its outcomes.

    Each live test maps to one result: PASS, a quality FAIL, or — when the failure text
    carries the infrastructure taxonomy (timeout / connection / rate-limit / 5xx) — the
    distinct INFRA_ERROR status (QA-EVAL-R12(b)). Never silently counted as a pass.
    """
    junit = Path("reports/eval-live-smoke.xml")
    junit.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - fixed argv, repo-local dev/CI tooling
        [sys.executable, "-m", "pytest", "-m", "llm", "-q", f"--junit-xml={junit}"],
        check=False,
    )
    results: list[LiveSuiteResult] = []
    tree = ET.parse(junit)  # noqa: S314 - our own pytest junit artifact, not untrusted XML
    for case in tree.getroot().iter("testcase"):
        name = f"live_smoke::{case.get('name', 'unknown')}"
        problems = [el for el in case if el.tag in {"failure", "error"}]
        if not problems:
            if not [el for el in case if el.tag == "skipped"]:
                results.append(LiveSuiteResult(name, LiveStatus.PASS))
            continue
        text = " ".join((el.get("message") or "") + (el.text or "") for el in problems)
        status = LiveStatus.INFRA_ERROR if classify_infra_text(text) else LiveStatus.FAIL
        results.append(LiveSuiteResult(name, status, detail=text[:500]))
    return results


def _cmd_run_live(args: argparse.Namespace) -> int:
    """LIVE mode (QA-EVAL-R9/CI-R4): recorded quality gates + the real-provider smoke.

    Env-gated on ``WATTWISE_LLM_API_KEY`` (fail-closed when absent; never part of the
    offline gate, TIER-R1). Failures are CLASSIFIED (QA-EVAL-R12(b)): a provider/network
    failure is ``INFRA_ERROR`` under the configured max rate (exceeding it alerts and
    blocks promotion); a genuine quality failure alerts as a regression. The artifact
    ``reports/eval-live-scorecard.json`` records per-case statuses + the ``clean`` flag
    that gates baseline advancement (QA-EVAL-R12(c)).
    """
    if not os.environ.get("WATTWISE_LLM_API_KEY"):
        print(
            "fail-closed: live mode requires WATTWISE_LLM_API_KEY "
            "(the env-gated live tier; the offline gate stays network-free, TIER-R1)",
            file=sys.stderr,
        )
        return 1
    cards = asyncio.run(_run_all())
    quality = [
        LiveSuiteResult(c.suite, LiveStatus.PASS if c.passed else LiveStatus.FAIL, scorecard=c)
        for c in cards
    ]
    results = (*quality, *_run_live_smoke())
    report = LiveRunReport.from_results(results)
    artifact = Path(args.scorecard or "reports/eval-live-scorecard.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "mode": "live",
                "clean": report.clean,
                "infra_error_rate": report.infra_error_rate,
                "max_infra_error_rate": report.max_infra_error_rate,
                "results": [
                    {"suite": r.suite, "status": r.status.value, "detail": r.detail}
                    for r in report.results
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for r in report.results:
        print(f"[{r.status.value.upper()}] {r.suite}")
    for line in report.alert_lines():
        print(line, file=sys.stderr)
    if report.quality_failed or report.infra_blocked:
        return 1
    print(f"live eval PASSED (infra_error_rate={report.infra_error_rate:.2f})")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run all suites, gate on absolute thresholds AND baseline non-regression.

    Two independent gates, both of which must pass (EVAL-R1 + QA-EVAL-R7): every suite must
    clear its absolute per-suite threshold, and no tracked suite metric may drop below the
    committed baseline. A safety-suite regression fails the build even if its absolute score
    would still pass (QA-EVAL-R7). When ``--no-baseline`` is set the regression gate is
    skipped (used only by ``update-baseline``, which is about to overwrite the baseline).
    """
    if _stale_cassette_failures():
        return 1
    if args.mode == "live":
        return _cmd_run_live(args)
    cards = asyncio.run(_run_all())
    if args.scorecard:
        _write_scorecard(cards, Path(args.scorecard))
    failed = [c.suite for c in cards if not c.passed]
    for card in cards:
        status = "PASS" if card.passed else "FAIL"
        print(f"[{status}] {card.suite} ({card.total_cases} cases)")
    if failed:
        print(f"eval gate FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    for tag_failure in baseline_tag_failures():
        print(tag_failure, file=sys.stderr)
        return 1
    if not args.no_baseline:
        report = compare_to_baseline(cards)
        for suite in report.new_suites:
            print(f"[NEW]  {suite} (no baseline yet; run eval-update-baseline)")
        if not report.passed:
            print(
                f"eval NON-REGRESSION gate FAILED (QA-EVAL-R7): {report.summary()}",
                file=sys.stderr,
            )
            return 1
        print(f"non-regression: {report.summary()}")
    print("eval gate PASSED")
    return 0


def _cmd_record(_args: argparse.Namespace) -> int:
    """Refresh + verify the recorded-response fixtures' cassette metadata (QA-EVAL-R12(a)).

    Stamps every dataset with the CURRENTLY pinned model + prompt-content fingerprint
    (a minimal, reviewable text edit — committed as a REVIEWED change with a rationale,
    never a test-run side-effect) and confirms the fixtures still load + verify.
    """
    changed = stamp_recorded_datasets()
    asyncio.run(_run_all())
    if _stale_cassette_failures():
        return 1
    if changed:
        print("cassette metadata refreshed (commit as a reviewed change): " + ", ".join(changed))
    print("recorded-response fixtures verified (datasets are the committed fixtures)")
    return 0


def _cmd_update_baseline(_args: argparse.Namespace) -> int:
    """Regenerate the committed baseline scorecard from a clean run (QA-EVAL-R7).

    The reviewed way to advance the non-regression floor: re-runs every suite and rewrites
    ``baseline-scorecard.json`` with the freshly measured per-suite metrics. REFUSES to
    write a baseline that does not itself clear the absolute thresholds — a baseline must
    record a passing run, never enshrine a failing one as the floor.
    """
    block = live_run_blocks_baseline()
    if block is not None:
        print(block, file=sys.stderr)
        return 1
    cards = asyncio.run(_run_all())
    failed = [c.suite for c in cards if not c.passed]
    if failed:
        print(
            "refusing to write baseline: these suites fail their absolute gate: "
            f"{', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    path = write_baseline(cards)
    print(f"baseline written: {path} ({len(cards)} suites)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m wattwise_core.eval")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run all suites and gate the build (EVAL-R1)")
    run.add_argument("--mode", default="recorded", choices=["recorded", "live"])
    run.add_argument("--scorecard", default=None, help="path to write the JSON scorecard")
    run.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the QA-EVAL-R7 non-regression check (used by update-baseline)",
    )
    run.set_defaults(func=_cmd_run)

    rec = sub.add_parser("record", help="verify recorded-response fixtures")
    rec.set_defaults(func=_cmd_record)

    base = sub.add_parser("update-baseline", help="reconfirm the committed baseline")
    base.set_defaults(func=_cmd_update_baseline)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch the eval subcommand and return its process exit code."""
    args = _build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
