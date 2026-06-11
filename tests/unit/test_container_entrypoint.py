"""Unit tests for the container entrypoint's migrate-then-serve contract (RUN-R6).

The image entrypoint (``docker/entrypoint.sh``) must make a fresh container's FIRST
boot self-sufficient: apply ``alembic upgrade head`` before serving when
``WATTWISE_MIGRATE_ON_START`` is truthy (the DEFAULT), skip it when falsy (readiness
then gates the unmigrated DB), abort the boot loudly — and never start uvicorn — when
the migration fails (fail-closed), and pass the CMD args through to uvicorn unchanged.

Proven here by running the real script under ``sh`` with stubbed ``alembic`` /
``python`` executables on PATH that record their argv — no Docker daemon needed, so the
contract is covered in the fast offline tier.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "entrypoint.sh"

_SH = shutil.which("sh") or "/bin/sh"
_UVICORN_ARGV = [
    "-m",
    "uvicorn",
    "--factory",
    "wattwise_core.api.app:create_app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
]


def _write_stub(bin_dir: Path, name: str, log: Path, exit_code: int = 0) -> None:
    """A PATH stub that appends its argv to ``log`` and exits with ``exit_code``."""
    stub = bin_dir / name
    stub.write_text(f'#!/bin/sh\necho "{name} $@" >> "{log}"\nexit {exit_code}\n')
    stub.chmod(0o755)


def _run(
    tmp_path: Path, *, migrate_env: str | None, alembic_exit: int = 0
) -> tuple[int, list[str]]:
    """Run the entrypoint with stubbed tools; return (exit code, recorded calls)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    _write_stub(bin_dir, "alembic", log, exit_code=alembic_exit)
    _write_stub(bin_dir, "python", log)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    if migrate_env is None:
        env.pop("WATTWISE_MIGRATE_ON_START", None)
    else:
        env["WATTWISE_MIGRATE_ON_START"] = migrate_env
    proc = subprocess.run(  # noqa: S603 — fixed argv over the repo's own script, no shell
        [_SH, str(_SCRIPT), "--host", "127.0.0.1", "--port", "8000"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return proc.returncode, calls


def test_default_is_migrate_then_serve(tmp_path: Path) -> None:
    """Unset env (the image default) migrates FIRST, then execs uvicorn with CMD args."""
    code, calls = _run(tmp_path, migrate_env=None)
    assert code == 0
    assert calls == [
        "alembic -c /app/alembic.ini upgrade head",
        "python " + " ".join(_UVICORN_ARGV),
    ]


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on"])
def test_truthy_values_migrate(tmp_path: Path, truthy: str) -> None:
    code, calls = _run(tmp_path, migrate_env=truthy)
    assert code == 0
    assert calls[0] == "alembic -c /app/alembic.ini upgrade head"


@pytest.mark.parametrize("falsy", ["0", "false", "no", "off", "OFF"])
def test_falsy_values_skip_migration_but_serve(tmp_path: Path, falsy: str) -> None:
    """Falsy values boot WITHOUT migrating — readiness (RUN-R6) then gates the DB."""
    code, calls = _run(tmp_path, migrate_env=falsy)
    assert code == 0
    assert calls == ["python " + " ".join(_UVICORN_ARGV)]


def test_empty_value_means_default_on(tmp_path: Path) -> None:
    """An EMPTY value behaves like unset (the shell `:-` default): migrate-on-start."""
    code, calls = _run(tmp_path, migrate_env="")
    assert code == 0
    assert calls[0] == "alembic -c /app/alembic.ini upgrade head"


def test_migration_failure_fails_the_boot_closed(tmp_path: Path) -> None:
    """A failing migration aborts the boot loudly; uvicorn is NEVER started."""
    code, calls = _run(tmp_path, migrate_env=None, alembic_exit=3)
    assert code != 0
    assert calls == ["alembic -c /app/alembic.ini upgrade head"]


def test_cmd_args_pass_through(tmp_path: Path) -> None:
    """Arbitrary CMD args reach uvicorn unchanged (exec-form ENTRYPOINT + CMD)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    _write_stub(bin_dir, "alembic", log)
    _write_stub(bin_dir, "python", log)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WATTWISE_MIGRATE_ON_START": "0",
    }
    proc = subprocess.run(  # noqa: S603 — fixed argv over the repo's own script, no shell
        [_SH, str(_SCRIPT), "--port", "9001", "--no-server-header"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    expected = (
        "python -m uvicorn --factory wattwise_core.api.app:create_app"
        " --port 9001 --no-server-header"
    )
    assert log.read_text().splitlines() == [expected]
