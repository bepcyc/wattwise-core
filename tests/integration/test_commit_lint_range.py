"""CI-R1 / QUAL-R12: fallback checks the tip; explicit and PR ranges retain all commits."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
GIT = shutil.which("git") or "/usr/bin/git"
BASH = shutil.which("bash") or "/bin/bash"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/lint_commits.sh"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(  # noqa: S603 — fixed executable and synthetic test arguments
        [GIT, "-C", str(repo), *args], text=True
    ).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Synthetic Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "core.hooksPath", str(tmp_path / "empty-hooks"))
    return tmp_path


def commit(repo: Path, message: str) -> str:
    git(repo, "commit", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def lint(repo: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for key in ("COMMIT_RANGE", "GITHUB_BASE_REF", "GITEA_BASE_REF", "FORGEJO_BASE_REF"):
        env.pop(key, None)
    return subprocess.run(  # noqa: S603 — repository gate against an isolated synthetic repo
        [BASH, str(SCRIPT)],
        cwd=repo,
        env=env | overrides,
        text=True,
        capture_output=True,
        check=False,
    )


def test_main_tip_does_not_walk_bad_history(repo: Path) -> None:
    commit(repo, "historical nonconventional subject")
    commit(repo, "fix(ci): valid current tip")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    result = lint(repo)
    assert result.returncode == 0, result.stderr
    assert "all 1 commit message(s)" in result.stdout
    assert "historical nonconventional" not in result.stdout


def test_bad_tip_still_fails(repo: Path) -> None:
    commit(repo, "fix(ci): valid ancestor")
    commit(repo, "invalid current tip")
    assert lint(repo).returncode == 1


@pytest.mark.parametrize("selection", ["explicit", "pr", "explicit_head"])
def test_declared_range_checks_bad_earlier_commit(repo: Path, selection: str) -> None:
    base = commit(repo, "chore: initial base")
    git(repo, "branch", "base", base)
    commit(repo, "invalid new change")
    commit(repo, "fix(ci): valid latest change")
    options = (
        {"COMMIT_RANGE": f"{base}..HEAD"}
        if selection == "explicit"
        else {"COMMIT_RANGE": "HEAD"}
        if selection == "explicit_head"
        else {"GITHUB_BASE_REF": "base"}
    )
    result = lint(repo, **options)
    assert result.returncode == 1
    assert "invalid new change" in result.stderr


def test_merge_tip_exemption_does_not_select_ancestor(repo: Path) -> None:
    commit(repo, "historical nonconventional subject")
    git(repo, "checkout", "-b", "side")
    commit(repo, "another historical invalid subject")
    git(repo, "checkout", "main")
    commit(repo, "fix(ci): main change")
    git(repo, "merge", "--no-ff", "side", "-m", "Merge side")
    result = lint(repo)
    assert result.returncode == 0, result.stderr
    assert "no non-merge commits" in result.stdout
