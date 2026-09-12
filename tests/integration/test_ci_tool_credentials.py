"""CI-R9a: upstream installer subprocesses must not inherit foreign credentials."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("server", "actions", "explicit", "expected"),
    [
        ("https://github.com", "true", "", "synthetic-ambient"),
        ("https://github.com/", "true", "", "synthetic-ambient"),
        ("https://forge.example", "true", "", ""),
        ("https://github.com.evil.example", "true", "", ""),
        ("https://github.com/path", "true", "", ""),
        ("https://github.com@evil.example", "true", "", ""),
        ("https://forge.example", "true", "synthetic-explicit", "synthetic-explicit"),
        ("https://github.com", "true", "synthetic-explicit", "synthetic-explicit"),
        ("", "false", "", "synthetic-ambient"),
        ("", "true", "", ""),
    ],
)
@pytest.mark.parametrize("tool", ["just", "trivy", "syft"])
def test_vendor_installer_inherits_only_github_credential(
    tmp_path: Path, server: str, actions: str, explicit: str, expected: str, tool: str
) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    for command in ("bash", "sh", "cat", "mkdir", "chmod", "head", "mktemp", "rm", "mv"):
        executable = shutil.which(command)
        assert executable is not None
        (commands / command).symlink_to(executable)
    curl = commands / "curl"
    # This fake fetch serves an executable installer, so the test observes the
    # actual child shell's environment rather than merely the parent's variables.
    curl.write_text(
        "#!/bin/sh\nout=/dev/stdout\nwhile [ $# -gt 0 ]; do\n"
        'if [ "$1" = -o ]; then out="$2"; shift; fi; shift\ndone\n'
        "cat > \"$out\" <<'INSTALLER'\n"
        'printf "%s" "${GITHUB_TOKEN:-}" > "$WW_TOKEN_AUDIT"\n'
        'destination="$WW_CI_BIN"\nif [ "${1:-}" = --to ]; then destination="$2"; fi\n'
        'mkdir -p "$destination"\n'
        'printf "#!/bin/sh\\nexit 0\\n" > "$destination/$WW_TEST_TOOL"\n'
        'chmod +x "$destination/$WW_TEST_TOOL"\nINSTALLER\n'
    )
    curl.chmod(0o755)
    audit = tmp_path / "inherited-token"
    env = {
        "PATH": str(commands),
        "HOME": str(tmp_path),
        "GITHUB_SERVER_URL": server,
        "GITHUB_ACTIONS": actions,
        "GITHUB_TOKEN": "synthetic-ambient",
        "GH_TOKEN": explicit,
        "WW_CI_BIN": str(tmp_path / "bin"),
        "WW_TOKEN_AUDIT": str(audit),
        "WW_TEST_TOOL": tool,
    }
    script = Path(__file__).resolve().parents[2] / "scripts" / "ci_tools.sh"
    result = subprocess.run(  # noqa: S603 - fixed script and synthetic test arguments
        [str(commands / "bash"), str(script), tool],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert audit.read_text() == expected
    assert "synthetic-" not in result.stdout + result.stderr
