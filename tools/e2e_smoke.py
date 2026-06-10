"""End-to-end smoke over the BUILT, RUNNING stack (E2E-R1 a-d / DOD-R5).

Drives the assembled engine over real HTTP — no TestClient, no in-process seams:

1. migrate a CLEAN throwaway SQLite to alembic head and boot uvicorn,
2. mint a first-party access token via ``POST /v1/auth/token`` (API-R23),
3. upload a FIT activity file via ``POST /v1/imports`` + trigger ``POST /v1/sync/run``,
4. read the canonical activity list (``GET /v1/activities``),
5. read the PMC (``GET /v1/performance/load-fitness``),
6. ask the coaching agent over SSE (``POST /v1/agent/ask`` with ``stream:true``) and
   require a terminal ``done`` event carrying a status-discriminated answer.

Step 6 needs a real model: set ``WATTWISE_LLM_API_KEY`` (and optionally
``WATTWISE_AGENT__MODEL``). Without the key the agent step is reported SKIPPED and the
script still exits non-zero, because the smoke is only proof when the agent answered.

Usage::

    WATTWISE_LLM_API_KEY=... uv run python -m tools.e2e_smoke

Set ``WATTWISE_E2E_BASE_URL`` to target an already-running server instead of booting one
(the script then skips migrate/boot and needs ``WATTWISE_E2E_OWNER_SECRET`` to sign in).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

_REPO = Path(__file__).resolve().parent.parent
_FIT = _REPO / "tests" / "contract" / "fixtures" / "file_upload" / "ride.fit"
_QUESTION = "How much training load have I done recently?"


def _free_port() -> int:
    """An OS-assigned free localhost TCP port for the throwaway server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_healthy(base: str, deadline_s: float = 30.0) -> None:
    """Poll ``/healthz`` until 200 or fail the smoke after ``deadline_s``."""
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        try:
            if httpx.get(f"{base}/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {base} did not become healthy in {deadline_s}s")


def _boot(tmp: Path) -> tuple[subprocess.Popen[bytes], str, str]:
    """Migrate a clean SQLite and boot uvicorn; return (proc, base_url, owner_secret)."""
    signing = secrets.token_hex(32)
    env = {
        **os.environ,
        "WATTWISE_DATABASE_DSN": f"sqlite+aiosqlite:///{tmp / 'e2e.sqlite'}",
        "WATTWISE_ENCRYPTION_ROOT_KEY": base64.b64encode(secrets.token_bytes(32)).decode(),
        "WATTWISE_TOKEN_SIGNING_KEY": signing,
        # The default object-store root (/var/lib/wattwise) is not writable in a dev
        # smoke; keep the verbatim-original retention inside the throwaway tmp dir.
        "WATTWISE_OBJECT_STORE__LOCAL_ROOT": str(tmp / "objects"),
    }
    uv = shutil.which("uv") or "uv"
    subprocess.run(  # noqa: S603 — fixed argv, no shell, repo-local smoke tool
        [uv, "run", "alembic", "upgrade", "head"],
        cwd=_REPO,
        env=env,
        check=True,
        capture_output=True,
    )
    port = _free_port()
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell, repo-local smoke tool
        [
            uv,
            "run",
            "uvicorn",
            "--factory",
            "wattwise_core.api.app:create_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=_REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    _wait_healthy(base)
    return proc, base, signing


def _step(name: str, ok: bool, detail: str = "") -> bool:
    """Print one PASS/FAIL line; return ``ok`` so callers can aggregate."""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def _sse_done_status(client: httpx.Client, base: str, auth: dict[str, str]) -> str | None:
    """Stream ``POST /v1/agent/ask`` and return the terminal ``done`` event's status."""
    body = {"question": _QUESTION, "stream": True}
    status: str | None = None
    with client.stream(
        "POST", f"{base}/v1/agent/ask", json=body, headers=auth, timeout=300.0
    ) as resp:
        if resp.status_code != 200:
            print(f"  agent/ask HTTP {resp.status_code}")
            return None
        current_event = ""
        for line in resp.iter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event == "done":
                payload = json.loads(line.split(":", 1)[1].strip())
                status = str(payload.get("status"))
    return status


def main() -> int:
    """Run the journey; exit 0 only when every step (incl. the agent ask) passed."""
    external = os.environ.get("WATTWISE_E2E_BASE_URL")
    proc: subprocess.Popen[bytes] | None = None
    results: list[bool] = []
    with tempfile.TemporaryDirectory(prefix="wattwise-e2e-") as tmp:
        if external:
            base, owner_secret = external, os.environ.get("WATTWISE_E2E_OWNER_SECRET", "")
        else:
            proc, base, owner_secret = _boot(Path(tmp))
            results.append(_step("migrate + boot", True, base))
        try:
            with httpx.Client(timeout=30.0) as client:
                # (d) first-party token issuance (API-R23).
                resp = client.post(f"{base}/v1/auth/token", json={"owner_secret": owner_secret})
                results.append(_step("auth token", resp.status_code == 200, str(resp.status_code)))
                if resp.status_code != 200:
                    return 1
                auth = {"Authorization": f"Bearer {resp.json()['access_token']}"}

                # (a) connect → sync: FIT upload + manual sync run.
                up = client.post(
                    f"{base}/v1/imports",
                    headers=auth,
                    files={"file": ("ride.fit", _FIT.read_bytes(), "application/octet-stream")},
                )
                results.append(_step("FIT import", up.status_code == 202, str(up.status_code)))
                run = client.post(f"{base}/v1/sync/run", headers=auth)
                results.append(_step("sync run", run.status_code == 202, str(run.status_code)))

                # Canonical activity surface holds the uploaded ride.
                acts = client.get(f"{base}/v1/activities", headers=auth)
                n_items = len(acts.json().get("data", [])) if acts.status_code == 200 else 0
                results.append(
                    _step(
                        "activities list",
                        acts.status_code == 200 and n_items >= 1,
                        f"{acts.status_code}, items={n_items}",
                    )
                )

                # (b) headline metric: the PMC over the uploaded ride's window.
                pmc = client.get(
                    f"{base}/v1/performance/load-fitness",
                    params={"from": "2024-01-01", "to": "2024-01-08"},
                    headers=auth,
                )
                results.append(
                    _step("PMC load-fitness", pmc.status_code == 200, str(pmc.status_code))
                )

                # (c) grounded agent ask over SSE — needs a real model.
                if os.environ.get("WATTWISE_LLM_API_KEY"):
                    status = _sse_done_status(client, base, auth)
                    results.append(
                        _step(
                            "agent ask SSE",
                            status in {"completed", "degraded"},
                            f"terminal status={status}",
                        )
                    )
                else:
                    results.append(
                        _step("agent ask SSE", False, "SKIPPED — WATTWISE_LLM_API_KEY unset")
                    )
        finally:
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=10)
    ok = all(results)
    print(f"\nE2E smoke: {'GREEN' if ok else 'RED'} ({sum(results)}/{len(results)} steps)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
