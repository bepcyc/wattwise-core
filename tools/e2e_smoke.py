"""End-to-end smoke over the BUILT, RUNNING stack (E2E-R1 a-d / DOD-R5, issue #29).

Drives the assembled engine over real HTTP — no TestClient, no in-process seams:

1. migrate a CLEAN throwaway SQLite to alembic head and boot uvicorn,
2. mint a first-party access token via ``POST /v1/auth/token`` (API-R23),
3. ask the agent on the EMPTY profile and require the HONEST REFUSAL
   (``degraded`` + zero citations — the GROUND-R6 fail-closed guarantee),
4. upload the static 2024 FIT fixture via ``POST /v1/imports`` + ``POST /v1/sync/run``
   and read the canonical activity list + the PMC over the fixture's window,
5. forge a TIME-RELATIVE batch of FIT activities (timestamps computed against "now",
   spread over the last two weeks — :mod:`tools.fit_forge`), import + sync them, and
   require the PMC over the RECENT window to show non-zero load (a deterministic,
   model-free proof that the forged batch can ground a recency question),
6. ask the agent again and require the GROUNDED ANSWER: terminal status ``completed``
   with at least one citation.

Steps 3 and 6 are the two product guarantees the smoke pins SEPARATELY (issue #29): the
agent must refuse when the record cannot support an answer AND answer when it can. A
``degraded`` outcome no longer passes the grounded-answer step — an honest refusal is
only a pass on the refusal step, where it is asserted explicitly.

The agent steps need a real model: set ``WATTWISE_LLM_API_KEY`` (and optionally
``WATTWISE_AGENT__MODEL``). Without the key both agent steps are reported SKIPPED and
the script still exits non-zero, because the smoke is only proof when the agent answered.

Usage::

    WATTWISE_LLM_API_KEY=... uv run python -m tools.e2e_smoke

Set ``WATTWISE_E2E_BASE_URL`` to target an already-running server instead of booting one
(the script then skips migrate/boot and needs ``WATTWISE_E2E_OWNER_SECRET`` to sign in).
"""

from __future__ import annotations

import base64
import datetime as _dt
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
from typing import Any

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:  # `python tools/e2e_smoke.py` direct-run support
    sys.path.insert(0, str(_REPO))

from tools.fit_forge import forge_recent_batch  # noqa: E402  (after the sys.path bootstrap)

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


def _sse_done_payload(
    client: httpx.Client, base: str, auth: dict[str, str]
) -> dict[str, Any] | None:
    """Stream ``POST /v1/agent/ask`` and return the terminal ``done`` event's payload."""
    body = {"question": _QUESTION, "stream": True}
    payload: dict[str, Any] | None = None
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
    return payload


def _ask_detail(payload: dict[str, Any] | None) -> str:
    """Human-readable detail line for an agent-ask step: status + citation count."""
    if payload is None:
        return "no terminal done event"
    citations = (payload.get("grounding") or {}).get("citations") or []
    return f"terminal status={payload.get('status')}, citations={len(citations)}"


def _citation_count(payload: dict[str, Any]) -> int:
    """The number of grounded citations on a terminal ``done`` payload."""
    return len((payload.get("grounding") or {}).get("citations") or [])


def _agent_refusal_step(client: httpx.Client, base: str, auth: dict[str, str]) -> bool:
    """Guarantee #1 (GROUND-R6, issue #29): the EMPTY profile provokes the honest refusal.

    Terminal ``degraded`` with ZERO citations — the abstain path ships only the
    limitation text. Run BEFORE any import so the refusal is provoked, never accidental.
    Without a model key the step is reported SKIPPED and fails (the smoke is only proof
    when the agent answered).
    """
    label = "agent honest refusal (empty profile)"
    if not os.environ.get("WATTWISE_LLM_API_KEY"):
        return _step(label, False, "SKIPPED — WATTWISE_LLM_API_KEY unset")
    refusal = _sse_done_payload(client, base, auth)
    refused = (
        refusal is not None
        and refusal.get("status") == "degraded"
        and _citation_count(refusal) == 0
    )
    return _step(label, refused, _ask_detail(refusal))


def _agent_grounded_step(client: httpx.Client, base: str, auth: dict[str, str]) -> bool:
    """Guarantee #2 (the headline ability, issue #29): a COMPLETED, citation-bearing answer.

    With a fresh record the agent must complete — ``degraded`` is no longer a pass here;
    an honest refusal only passes the refusal step, where it is asserted explicitly.
    """
    label = "agent grounded answer (fresh batch)"
    if not os.environ.get("WATTWISE_LLM_API_KEY"):
        return _step(label, False, "SKIPPED — WATTWISE_LLM_API_KEY unset")
    answer = _sse_done_payload(client, base, auth)
    grounded = (
        answer is not None and answer.get("status") == "completed" and _citation_count(answer) >= 1
    )
    return _step(label, grounded, _ask_detail(answer))


def _signature_step(client: httpx.Client, base: str, auth: dict[str, str]) -> bool:
    """Set the owner FTP signature (GBO-R26) — the write the power analytics ground on.

    Without an effective FTP every power metric (NP/IF/TSS → CTL) is typed-unavailable,
    so the PMC could never show load and a "training load" ask could never ground. The
    ``effective_date`` predates BOTH the static 2024 fixture and the forged recent batch
    so the signature resolves for every imported ride (ANL-R9 time-effective FTP).
    """
    resp = client.put(
        f"{base}/v1/athlete/signature",
        headers=auth,
        json={"ftp_w": 250.0, "signature_type": "cycling", "effective_date": "2024-01-01"},
    )
    return _step("set FTP signature", resp.status_code == 200, str(resp.status_code))


def _static_fixture_steps(client: httpx.Client, base: str, auth: dict[str, str]) -> list[bool]:
    """(a)+(b): static 2024 FIT upload + sync, canonical list, PMC over its window."""
    results: list[bool] = []
    up = client.post(
        f"{base}/v1/imports",
        headers=auth,
        files={"file": ("ride.fit", _FIT.read_bytes(), "application/octet-stream")},
    )
    results.append(_step("FIT import", up.status_code == 202, str(up.status_code)))
    run = client.post(f"{base}/v1/sync/run", headers=auth)
    results.append(_step("sync run", run.status_code == 202, str(run.status_code)))
    acts = client.get(f"{base}/v1/activities", headers=auth)
    n_items = len(acts.json().get("data", [])) if acts.status_code == 200 else 0
    results.append(
        _step(
            "activities list",
            acts.status_code == 200 and n_items >= 1,
            f"{acts.status_code}, items={n_items}",
        )
    )
    pmc = client.get(
        f"{base}/v1/performance/load-fitness",
        params={"from": "2024-01-01", "to": "2024-01-08"},
        headers=auth,
    )
    results.append(_step("PMC load-fitness", pmc.status_code == 200, str(pmc.status_code)))
    return results


def _fresh_batch_steps(client: httpx.Client, base: str, auth: dict[str, str]) -> list[bool]:
    """Time-relative fixture (issue #29): forge, import + sync, prove recency grounding.

    The static 2024 ride can never ground a "recent training" question — only refuse
    it. The forged batch (timestamps computed against "now", spread over the last two
    weeks) makes the grounded path provable; the PMC read over the RECENT window is the
    deterministic, model-free proof the batch grounds recency even when the agent steps
    are skipped (no model key).
    """
    results: list[bool] = []
    batch = forge_recent_batch()
    batch_ok = True
    for ride in batch:
        fup = client.post(
            f"{base}/v1/imports",
            headers=auth,
            files={"file": (ride.filename, ride.payload, "application/octet-stream")},
        )
        batch_ok = batch_ok and fup.status_code == 202
    results.append(_step("forged recent FIT batch import", batch_ok, f"{len(batch)} files"))
    run = client.post(f"{base}/v1/sync/run", headers=auth)
    results.append(_step("sync run (fresh batch)", run.status_code == 202, str(run.status_code)))
    acts = client.get(f"{base}/v1/activities", headers=auth)
    n_total = len(acts.json().get("data", [])) if acts.status_code == 200 else 0
    results.append(
        _step(
            "activities list (fresh batch)",
            acts.status_code == 200 and n_total >= 1 + len(batch),
            f"{acts.status_code}, items={n_total}",
        )
    )
    today = _dt.datetime.now(_dt.UTC).date()
    pmc = client.get(
        f"{base}/v1/performance/load-fitness",
        params={"from": str(today - _dt.timedelta(days=13)), "to": str(today)},
        headers=auth,
    )
    fitness = (pmc.json().get("summary") or {}).get("fitness") if pmc.status_code == 200 else None
    results.append(
        _step(
            "PMC recent window (forged batch grounds recency)",
            pmc.status_code == 200 and fitness is not None and fitness > 0,
            f"{pmc.status_code}, fitness={fitness}",
        )
    )
    return results


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

                results.append(_agent_refusal_step(client, base, auth))
                results.append(_signature_step(client, base, auth))
                results.extend(_static_fixture_steps(client, base, auth))
                results.extend(_fresh_batch_steps(client, base, auth))
                results.append(_agent_grounded_step(client, base, auth))
        finally:
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=10)
    ok = all(results)
    print(f"\nE2E smoke: {'GREEN' if ok else 'RED'} ({sum(results)}/{len(results)} steps)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
