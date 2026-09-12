"""Fail-closed live-smoke reporting, without launching pytest or a paid provider."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from wattwise_core.eval import __main__ as cli
from wattwise_core.eval.live import LiveStatus, LiveSuiteResult
from wattwise_core.eval.scorecard import EvalMode

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("returncode", "xml"),
    [
        (2, '<testsuite><testcase name="test_ok"/></testsuite>'),
        (-2, '<testsuite><testcase name="test_ok"/></testsuite>'),
        (3, '<testsuite><testcase name="test_ok"/></testsuite>'),
        (0, None),
        (0, "<broken"),
        (0, "<testsuite/>"),
        (0, "<testsuite><testcase/></testsuite>"),
        (0, '<testsuite><testcase name="unknown"/></testsuite>'),
        (0, '<testsuite><testcase name="test_skip"><skipped/></testcase></testsuite>'),
        (1, '<testsuite><testcase name="test_ok"/></testsuite>'),
    ],
)
def test_incomplete_smoke_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int, xml: str | None
) -> None:
    """QA-EVAL-R12(b): invalid or interrupted live execution is a blocking result."""
    monkeypatch.chdir(tmp_path)
    artifact = Path("reports/eval-live-smoke.xml")
    artifact.parent.mkdir()
    artifact.write_text('<testsuite><testcase name="stale_success"/></testsuite>')

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if xml is not None:
            artifact.write_text(xml)
        return subprocess.CompletedProcess([], returncode)

    monkeypatch.setattr(cli.subprocess, "run", run)
    results = cli._run_live_smoke()
    assert any(result.status is LiveStatus.FAIL for result in results)


def test_database_traceback_is_quality_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA-EVAL-R12(b): SQL connection stack frames do not excuse a locked database."""
    monkeypatch.chdir(tmp_path)

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path("reports/eval-live-smoke.xml").write_text(
            '<testsuite><testcase name="test_live"><failure '
            'message="sqlalchemy.exc.OperationalError: database is locked">'
            "self._execute_on_connection()\nconnection._execute_clauseelement()"
            "</failure></testcase></testsuite>"
        )
        return subprocess.CompletedProcess([], 1)

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli._run_live_smoke()[0].status is LiveStatus.FAIL


@pytest.mark.parametrize("recorded_passed", [True, False])
def test_recorded_suites_cannot_dilute_live_infrastructure_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded_passed: bool
) -> None:
    """QA-EVAL-R12(b): live infra uses live attempts while recorded quality still gates."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WATTWISE_LLM_API_KEY", "fixture-not-a-credential")

    async def recorded() -> list[SimpleNamespace]:
        return [
            SimpleNamespace(suite=f"recorded{i}", passed=recorded_passed, mode=EvalMode.RECORDED)
            for i in range(20)
        ]

    monkeypatch.setattr(cli, "_run_all", recorded)
    monkeypatch.setattr(
        cli, "_run_live_smoke", lambda: [LiveSuiteResult("live", LiveStatus.INFRA_ERROR)]
    )
    assert cli._cmd_run_live(argparse.Namespace(scorecard=None)) == 1
    artifact = json.loads(Path("reports/eval-live-scorecard.json").read_text())
    assert artifact["infra_error_rate"] == 1.0
    assert not artifact["clean"]


@pytest.mark.parametrize("recorded_passed", [True, False])
def test_genuine_provider_blip_uses_allowance_but_preserves_recorded_quality_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded_passed: bool
) -> None:
    """QA-EVAL-R12(b): one real provider blip in twenty lives uses the configured allowance."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WATTWISE_LLM_API_KEY", "fixture-not-a-credential")

    async def recorded() -> list[SimpleNamespace]:
        return [SimpleNamespace(suite="recorded", passed=recorded_passed, mode=EvalMode.RECORDED)]

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        cases = "".join(f'<testcase name="test_ok{i}"/>' for i in range(19))
        Path("reports/eval-live-smoke.xml").write_text(
            "<testsuite>" + cases + '<testcase name="test_provider"><failure '
            'message="openai.RateLimitError: Error code: 429 - rate limit exceeded">'
            "provider stack trace</failure></testcase></testsuite>"
        )
        return subprocess.CompletedProcess([], 1)

    monkeypatch.setattr(cli, "_run_all", recorded)
    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli._cmd_run_live(argparse.Namespace(scorecard=None)) == (0 if recorded_passed else 1)
    artifact = json.loads(Path("reports/eval-live-scorecard.json").read_text())
    assert artifact["infra_error_rate"] == 0.05
    assert not artifact["clean"]
