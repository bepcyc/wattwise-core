"""The documented file-upload path produces historical load from a fresh owner (API-R46d)."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tools.fit_forge import forge_ride

from wattwise_core.api.app import create_app
from wattwise_core.api.routers import athlete, onboarding
from wattwise_core.config import load_settings
from wattwise_core.identity import OWNER_ATHLETE_ID
from wattwise_core.persistence.models import Athlete, Base, SourceDescriptor, Sport
from wattwise_core.security.crypto import EnvelopeCipher

pytestmark = pytest.mark.e2e

_SIGNING_KEY = "onboarding-test-secret-0123456789abcdef"
_RIDE_DATE = dt.date(2026, 6, 1)
_REFERENCE_NOW = dt.datetime(2026, 6, 8, 12, tzinfo=dt.UTC)
_REPO_ROOT = Path(__file__).resolve().parents[2]


async def _bootstrap(app: FastAPI) -> None:
    """Seed only the initial owner and registries; all athlete data arrives through the API."""
    database = app.state.database
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with database.session() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        session.add(Athlete(athlete_id=OWNER_ATHLETE_ID, reference_timezone="UTC"))
        session.add(
            SourceDescriptor(
                source_key="file_import", display_name="Activity files", kind="file_upload"
            )
        )


class _ReferenceDatetime(dt.datetime):
    """Freeze date resolution in the two profile routes, without touching authentication clocks."""

    @classmethod
    def now(cls, tz: dt.tzinfo | None = None) -> _ReferenceDatetime:
        return cls.fromtimestamp(_REFERENCE_NOW.timestamp(), tz)


@pytest.fixture
def first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, dict[str, str]]]:
    """Build the stock app on a disposable file database and mint a real owner token."""
    route_datetime = SimpleNamespace(**(vars(dt) | {"datetime": _ReferenceDatetime}))
    monkeypatch.setattr(athlete, "_dt", route_datetime)
    monkeypatch.setattr(onboarding, "_dt", route_datetime)
    settings = load_settings(
        app__environment="development",
        database_dsn=f"sqlite+aiosqlite:///{tmp_path / 'onboarding.sqlite'}",
        token_signing_key=_SIGNING_KEY,
        encryption_root_key=EnvelopeCipher.generate_root_key(),
        object_store__local_root=str(tmp_path / "objects"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        client.portal.call(_bootstrap, app)  # type: ignore[union-attr]
        token = client.post("/v1/auth/token", json={"owner_secret": _SIGNING_KEY})
        assert token.status_code == 200, token.text
        yield client, {"Authorization": f"Bearer {token.json()['access_token']}"}


def _readme_signature() -> dict[str, object]:
    """Read the quickstart's signature payload to catch its default-date regression."""
    readme = (_REPO_ROOT / "README.md").read_text()
    match = re.search(r'/v1/athlete/signature"[^`]*?-d \'([^\']+)\'', readme)
    assert match is not None, "quickstart must provide an executable FTP signature request"
    payload: dict[str, object] = json.loads(match.group(1))
    return payload


def _historical_load(client: TestClient, auth: dict[str, str]) -> tuple[float | None, float]:
    params = {"from": _RIDE_DATE.isoformat(), "to": "2026-06-08"}
    coggan = client.get("/v1/performance/coggan", headers=auth, params=params)
    assert coggan.status_code == 200, coggan.text
    items = coggan.json()["items"]
    assert len(items) == 1
    pmc = client.get("/v1/performance/load-fitness", headers=auth, params=params)
    assert pmc.status_code == 200, pmc.text
    return items[0]["values"]["tss"], pmc.json()["summary"]["fitness"]


def test_readme_upload_and_dated_ftp_produce_historical_load(
    first_use: tuple[TestClient, dict[str, str]],
) -> None:
    """Historical load needs an applicable FTP, even when today's FTP makes onboarding all_set."""
    client, auth = first_use
    profile = client.get("/v1/athlete", headers=auth).json()
    assert profile["current_sport"] is None and profile["fitness_signature"] is None
    assert client.get("/v1/onboarding/status", headers=auth).json()["first_data_ready"] is False

    fit = forge_ride(start=dt.datetime.combine(_RIDE_DATE, dt.time(10), tzinfo=dt.UTC))
    uploaded = client.post(
        "/v1/imports",
        headers=auth,
        files={"file": ("history.fit", fit, "application/octet-stream")},
    )
    assert uploaded.status_code == 202, uploaded.text
    assert uploaded.json()["status"] == "done"
    job = client.get(f"/v1/imports/{uploaded.json()['import_job_id']}", headers=auth)
    assert job.status_code == 200 and job.json()["status"] == "done"
    activities = client.get("/v1/activities", headers=auth).json()["data"]
    assert len(activities) == 1 and activities[0]["has_power"] is True
    assert _historical_load(client, auth) == (None, 0.0)

    sport = client.put("/v1/athlete", headers=auth, json={"current_sport": "cycling"})
    assert sport.status_code == 200, sport.text
    assert (
        client.get("/v1/onboarding/status", headers=auth).json()["suggested_next_step"] == "set_ftp"
    )
    current = client.put("/v1/athlete/signature", headers=auth, json={"ftp_w": 250})
    assert current.status_code == 200, current.text
    assert (
        current.json()["fitness_signature"]["effective_date"] == _REFERENCE_NOW.date().isoformat()
    )
    status = client.get("/v1/onboarding/status", headers=auth).json()
    assert status["suggested_next_step"] == "all_set" and status["first_data_ready"] is True
    assert status["has_connection"] is False
    assert _historical_load(client, auth) == (None, 0.0)

    dated = client.put("/v1/athlete/signature", headers=auth, json=_readme_signature())
    assert dated.status_code == 200, dated.text
    tss, fitness = _historical_load(client, auth)
    assert tss is not None and tss > 0, "the quickstart FTP must cover the uploaded history"
    assert fitness > 0, "the first historical chart must contain computed load without reimporting"
    assert client.get("/v1/activities", headers=auth).json()["data"] == activities
