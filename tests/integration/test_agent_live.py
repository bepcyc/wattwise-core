"""Live-LLM tier: the coaching agent must produce a GROUNDED, COMPLETED answer end-to-end.

This is the regression guard for the headline-feature bug a green OFFLINE suite hid (doc-50
§10 of the dev playbook / MODEL-R5a / GROUND-R7 / §16): driven by a REAL model the agent
answers in NATURAL/AGGREGATE terms ("your fitness is 6.7", "ctl (chronic training load)") with
no as-of date — and the old grounder, which only matched EXACT canonical metric keys, scrubbed
every claim and ABSTAINED on a CORRECT answer. The fix (a config-loaded coach system prompt +
metric-equivalence/alias layer + dateless latest-day resolution + a coaching-grade numeric
tolerance) makes such a real, correct answer GROUND instead of degrade.

This tier is OFF by default: it is marked ``llm`` and skipped unless ``WATTWISE_LLM_API_KEY`` is
set, so the ``-n auto`` offline gate stays network-free and reproducible (TIER-R1). It hits the
configured OpenAI-compatible model (the OSS default ``deepseek/deepseek-v4-flash``) via the real
engine path — ``build_agent_engine`` loads the §16 coach-config from ``defaults.toml``.

A real model is stochastic, so a single draft MAY occasionally phrase the answer in a way that
does not ground (the engine then correctly DEGRADES rather than fabricating). The test therefore
makes a small bounded number of attempts and asserts that the capability WORKS: at least one
attempt yields a grounded, non-abstaining COMPLETED answer carrying ≥1 citation whose value
matches the canonical analytic (no fabrication). That asserts the headline feature, not luck.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from wattwise_core.agent.capabilities import CanonicalEvidence
from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.engine import build_agent_engine
from wattwise_core.agent.engine_services import CoachBundle
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.config import load_settings
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, SignatureOrigin
from wattwise_core.identity import OWNER_ATHLETE_ID
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.persistence import Database
from wattwise_core.persistence.models import (
    Athlete,
    Base,
    FitnessSignature,
    SourceDescriptor,
    Sport,
)
from wattwise_core.storage import content_hash

# Register agent-memory ORM so the durable saver's schema is complete (mirrors the smoke seed).
import wattwise_core.agent.memory  # noqa: F401  isort:skip

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("WATTWISE_LLM_API_KEY"),
        reason="live-LLM tier: set WATTWISE_LLM_API_KEY (sanctioned OPENROUTER_API_KEY) to run",
    ),
]

UTC = _dt.UTC
_RIDE_DAYS = (_dt.date(2026, 6, 6), _dt.date(2026, 6, 7), _dt.date(2026, 6, 8))
_QUESTION = "How much training load have I done over the last six weeks?"
# A real model is stochastic; assert the capability works within a few attempts, not on one draft.
_MAX_ATTEMPTS = 4


def _ride(native_id: str, day: _dt.date) -> GboCandidate:
    """A constant-250 W, 1 h cycling ride (TSS == 100 at FTP 250) on ``day``."""
    seconds, watts = 3600, 250.0
    payload = {
        "start_time": _dt.datetime(day.year, day.month, day.day, 8, 0, tzinfo=UTC),
        "sport": "cycling",
        "elapsed_time_s": seconds,
        "moving_time_s": seconds,
        "avg_power_w": watts,
        "streams": {
            "power_w": {"values": [watts] * seconds, "sample_basis": "time", "sample_rate_hz": 1.0}
        },
        "laps": [
            {"lap_index": 0, "start_offset_s": 0, "duration_s": seconds, "avg_power_w": watts}
        ],
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(native_id.encode()),
        payload=payload,
        trust_tier=Fidelity.RAW_STREAM,
        fetched_at=_dt.datetime(2026, 6, 9, 9, 0, tzinfo=UTC),
    )


@pytest_asyncio.fixture
async def live_db(tmp_path) -> AsyncIterator[Database]:  # type: ignore[no-untyped-def]
    """A REAL file-sqlite pool (not :memory:) seeded with the owner + FTP + three 100-TSS rides.

    File-sqlite gives the durable checkpointer a real multi-connection pool (§7); :memory:/
    StaticPool would false-green the saver. The canonical + agent-state schemas are created here.
    """
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'live_canon.sqlite'}"
    settings = load_settings(app__environment="development", database_dsn=dsn)
    db = Database(settings)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(AgentStateBase.metadata.create_all)
    async with db.session() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        session.add(Athlete(athlete_id=OWNER_ATHLETE_ID, sex="male", reference_timezone="UTC"))
        descriptor = SourceDescriptor(
            source_key="file_import", display_name="Activity files", kind="file_upload"
        )
        session.add(descriptor)
        session.add(
            FitnessSignature(
                athlete_id=OWNER_ATHLETE_ID,
                signature_type="cycling",
                effective_date=_dt.date(2024, 1, 1),
                ftp_w=250.0,
                origin=SignatureOrigin.MEASURED,
            )
        )
        await session.flush()
        ingest = IngestService(session)
        for i, day in enumerate(_RIDE_DAYS):
            await ingest.ingest(
                str(OWNER_ATHLETE_ID), str(descriptor.source_descriptor_id), [_ride(f"r{i}", day)]
            )
    yield db
    await db.dispose()


async def _canonical_matches(db: Database, metric: str | None, value: float | None) -> bool:
    """True iff a cited (metric, value) re-states the canonical analytic verbatim (GROUND-R7).

    Resolves the citation through the SAME canonical evidence (with the loaded equivalence) the
    grounder used and confirms the cited value matches within the loaded coaching tolerance — so
    the test verifies a REAL grounded number, never accepting a fabricated one.
    """
    if metric is None or value is None:
        return False
    settings = load_settings(
        app__environment="development", database_dsn="sqlite+aiosqlite:///:memory:"
    )
    coach = CoachBundle.from_settings(settings)
    async with db.session() as session:
        evidence = CanonicalEvidence(
            AnalyticsService(session),
            str(OWNER_ATHLETE_ID),
            equivalence=coach.equivalence,
            reference_date=_dt.date(2026, 6, 9),
        )
        canonical = await evidence.metric_value(metric, None)
    if canonical is None:
        return False
    return math.isclose(value, canonical, rel_tol=coach.tolerance.rel, abs_tol=coach.tolerance.abs_)


async def test_live_agent_answers_grounded_and_completed(live_db: Database) -> None:
    """The live agent produces a GROUNDED, COMPLETED answer with ≥1 citation matching canonical.

    Drives the REAL engine (loaded §16 coach-config: system prompt + metric-equivalence +
    tolerance) against the configured model. Asserts within a bounded number of attempts: a
    non-abstaining COMPLETED status, a non-empty answer, no abstain/limitation copy, and at least
    one citation whose value re-states the canonical analytic verbatim (GROUND-R7 — no fabrication).
    """
    settings = load_settings(
        app__environment="development",
        database_dsn=str(live_db.engine.url),
        llm_api_key=os.environ["WATTWISE_LLM_API_KEY"],
    )
    engine = build_agent_engine(live_db, settings)
    assert engine is not None, "live tier requires a configured model (WATTWISE_LLM_API_KEY)"

    last_status: RunStatus | None = None
    for _ in range(_MAX_ATTEMPTS):
        answer = await engine.answer(
            athlete_id=str(OWNER_ATHLETE_ID),
            question=_QUESTION,
            thread_id=None,
            response_length="standard",
            follow_up=None,
            locale="en",
        )
        last_status = answer.status
        if answer.status is not RunStatus.COMPLETED or not answer.citations:
            continue
        # A real grounded, completed answer: non-empty, no abstain/limitation copy, and EVERY
        # surfaced citation re-states a canonical analytic verbatim (GROUND-R7, no fabrication).
        assert answer.answer_text.strip()
        assert "enough confirmed data" not in answer.answer_text.lower()
        matches = [
            await _canonical_matches(live_db, c.metric, c.value) for c in answer.citations
        ]
        assert all(matches), f"a citation did not match canonical analytics: {answer.citations}"
        assert any(matches)
        return

    pytest.fail(
        "live agent did not reach a grounded COMPLETED answer with citations in "
        f"{_MAX_ATTEMPTS} attempts (last status={last_status}); the headline grounding path "
        "regressed — natural-term claims are not grounding against canonical analytics."
    )
