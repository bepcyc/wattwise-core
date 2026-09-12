"""CI-R9a: vendor failures remain bounded and never publish partial tools."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
INSTALLER = """#!/bin/bash
set -eu
mkdir -p "$2"
printf '#!/bin/sh\\nexit 0\\n' > "$2/just"
chmod +x "$2/just"
curl -H "Authorization: Bearer ${GITHUB_TOKEN:-}" https://api.github.com/repos/casey/just/releases/latest
"""


@pytest.mark.parametrize(
    ("mode", "success", "attempts"),
    [("success", True, 1), ("recover", True, 3), ("permanent", False, 3), ("outer_fail", False, 0)],
)
def test_vendor_request_failure_is_bounded_and_atomic(
    tmp_path: Path, mode: str, success: bool, attempts: int
) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    for name in ("bash", "sh", "mkdir", "chmod", "mktemp", "rm", "mv", "head"):
        executable = shutil.which(name)
        assert executable is not None
        (commands / name).symlink_to(executable)
    (commands / "sleep").write_text("#!/bin/sh\nexit 0\n")
    (commands / "sleep").chmod(0o755)
    curl = commands / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"installer = {INSTALLER!r}\n"
        "args = sys.argv[1:]\n"
        "mode = os.environ['WW_TEST_MODE']\n"
        "if 'https://just.systems/install.sh' in args:\n"
        "    if '-o' in args:\n"
        "        pathlib.Path(args[args.index('-o') + 1]).write_text(installer)\n"
        "    else:\n"
        "        print(installer)\n"
        "    sys.exit(22 if mode == 'outer_fail' else 0)\n"
        "assert 'https://api.github.com/repos/casey/just/releases/latest' in args\n"
        "audit = pathlib.Path(os.environ['WW_TEST_AUDIT'])\n"
        "calls = json.loads(audit.read_text()) if audit.exists() else []\n"
        "calls.append(args[args.index('-H') + 1])\n"
        "audit.write_text(json.dumps(calls))\n"
        "sys.exit(22 if mode == 'permanent' or (mode == 'recover' and len(calls) < 3) else 0)\n"
    )
    curl.chmod(0o755)
    audit = tmp_path / "requests.json"
    destination = tmp_path / "bin"
    env = {
        "PATH": str(commands),
        "HOME": str(tmp_path),
        "WW_CI_BIN": str(destination),
        "WW_TEST_MODE": mode,
        "WW_TEST_AUDIT": str(audit),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_ACTIONS": "true",
        "GH_TOKEN": "synthetic-job-token",
    }
    result = subprocess.run(  # noqa: S603 — fixed script and synthetic environment
        [str(commands / "bash"), str(ROOT / "scripts/ci_tools.sh"), "just"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is success, result.stderr
    assert (destination / "just").exists() is success
    calls = json.loads(audit.read_text()) if audit.exists() else []
    assert calls == ["Authorization: Bearer synthetic-job-token"] * attempts
    assert not list(destination.glob(".just-install.*"))
    assert "synthetic-job-token" not in result.stdout + result.stderr


def test_github_job_token_is_scoped_to_tool_install_steps() -> None:
    for filename in ("ci.yml", "nightly.yml"):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("run", "").startswith("bash scripts/ci_tools.sh "):
                    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
        forgejo = yaml.safe_load((ROOT / ".forgejo/workflows" / filename).read_text())
        for job in forgejo["jobs"].values():
            for step in job.get("steps", []):
                if step.get("run", "").startswith("bash scripts/ci_tools.sh "):
                    assert "GH_TOKEN" not in step.get("env", {})
