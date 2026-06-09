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
import sys
from pathlib import Path

from wattwise_core.eval.baseline import compare_to_baseline, write_baseline
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


def _cmd_run(args: argparse.Namespace) -> int:
    """Run all suites, gate on absolute thresholds AND baseline non-regression.

    Two independent gates, both of which must pass (EVAL-R1 + QA-EVAL-R7): every suite must
    clear its absolute per-suite threshold, and no tracked suite metric may drop below the
    committed baseline. A safety-suite regression fails the build even if its absolute score
    would still pass (QA-EVAL-R7). When ``--no-baseline`` is set the regression gate is
    skipped (used only by ``update-baseline``, which is about to overwrite the baseline).
    """
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
    """Confirm the recorded-response fixtures load deterministically (QA-EVAL-R9 no-op)."""
    asyncio.run(_run_all())
    print("recorded-response fixtures verified (datasets are the committed fixtures)")
    return 0


def _cmd_update_baseline(_args: argparse.Namespace) -> int:
    """Regenerate the committed baseline scorecard from a clean run (QA-EVAL-R7).

    The reviewed way to advance the non-regression floor: re-runs every suite and rewrites
    ``baseline-scorecard.json`` with the freshly measured per-suite metrics. REFUSES to
    write a baseline that does not itself clear the absolute thresholds — a baseline must
    record a passing run, never enshrine a failing one as the floor.
    """
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
    run.add_argument("--mode", default="recorded", choices=["recorded"])
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
