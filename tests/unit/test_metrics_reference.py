"""Completeness gate for the athlete-facing metrics & parameters reference.

``docs/METRICS.md`` is the single systematic reference for every parameter the engine
collects and every metric it computes. This test makes that promise enforceable: it
derives the live key inventory straight from the code surfaces a reader would meet — the
closed metric vocabulary (:class:`~wattwise_core.agent.capabilities_metrics.MetricName`),
the per-activity load-metrics bundle, the athlete-level analytics surface, the canonical
ORM columns of ``activity`` / ``daily_wellness`` / ``fitness_signature``, and the canonical
stream channels — and asserts that each documented key has a reference entry in the doc.

If someone adds a metric to ``MetricName`` or a canonical column to one of the tables
without writing its entry, this test FAILS and names the missing keys. The reference is
therefore part of the definition of done for any new metric or collected parameter.

An entry is recognised by its anchor heading: a Markdown ``###`` line that contains the
exact canonical key wrapped in backticks — e.g. ``### Chronic Training Load (`ctl`)``.

Each excluded key below is justified in a comment: it is identity/plumbing/audit
machinery a reader never reads as a measured value, not a collected parameter or computed
metric, so documenting it would dilute the reference rather than complete it.
"""

from __future__ import annotations

import re
from pathlib import Path

from wattwise_core.agent.capabilities_metrics import MetricName
from wattwise_core.domain.enums import StreamChannelName
from wattwise_core.persistence.models import Activity, FitnessSignature
from wattwise_core.persistence.models.wellness import DailyWellness

_DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "METRICS.md"

# Identity / referential / audit / plumbing columns shared by every canonical table.
# These are surrogate keys, FK anchors, write timestamps, the typed coverage descriptor,
# and the lineage/conflict-resolution bookkeeping (LIN-R3/CONF-R6) — never a value an
# athlete reads off a chart. They are deliberately NOT reference entries.
_COMMON_PLUMBING = frozenset(
    {
        "athlete_id",  # FK anchor to the athlete, not a measured value
        "created_at",  # write-side audit timestamp
        "updated_at",  # write-side audit timestamp
        "coverage",  # typed coverage/fidelity descriptor (documented as a cross-cutting concept)
        "policy_version",  # conflict-resolution policy version (lineage only, CONF-R6)
        "field_resolution",  # per-field winner/candidate pointers (lineage only, LIN-R3/R4)
    }
)

_ACTIVITY_EXCLUDE = _COMMON_PLUMBING | frozenset(
    {
        "activity_id",  # surrogate canonical identity (PK), not a measured value
        "start_time_local",  # derived display projection of start_time (GBO-R13)
        "local_date",  # derived athlete-local day bucket of start_time (GBO-R35), not a reading
    }
)

_WELLNESS_EXCLUDE = _COMMON_PLUMBING | frozenset(
    {
        "daily_wellness_id",  # surrogate PK, not a measured value
        "local_date",  # the athlete-local day key identifying the row, not a reading
    }
)

# fitness_signature has no coverage/policy_version/field_resolution columns, so its
# plumbing set is built from only the identity/interval bookkeeping it actually carries.
_SIGNATURE_EXCLUDE = frozenset(
    {
        "athlete_id",  # FK anchor to the athlete, not a measured value
        "created_at",  # write-side audit timestamp
        "updated_at",  # write-side audit timestamp
        "signature_id",  # surrogate PK, not a measured value
        "fit_quality",  # modeled-fit goodness metadata (R^2/n/residuals), reported in quality
    }
)


def _documented_keys() -> set[str]:
    """Every canonical key that appears as an anchor heading in the reference doc.

    An anchor heading is a ``###`` line carrying the exact key(s) in backticks, e.g.
    ``### Normalized Power (`np`)`` or a shared heading naming several flag keys
    ``### Channel-presence flags (`has_power` / `has_hr` / ...)``. Every backticked token
    on a heading line counts, so a key is only credited when it is named in a heading, not
    merely mentioned in prose.
    """
    text = _DOC_PATH.read_text(encoding="utf-8")
    keys: set[str] = set()
    for line in text.splitlines():
        if line.startswith("###"):
            keys.update(re.findall(r"`([^`]+)`", line))
    return keys


def _activity_columns() -> set[str]:
    return {c.key for c in Activity.__table__.columns} - _ACTIVITY_EXCLUDE


def _wellness_columns() -> set[str]:
    return {c.key for c in DailyWellness.__table__.columns} - _WELLNESS_EXCLUDE


def _signature_columns() -> set[str]:
    return {c.key for c in FitnessSignature.__table__.columns} - _SIGNATURE_EXCLUDE


def _metric_names() -> set[str]:
    return {m.value for m in MetricName}


def _stream_channels() -> set[str]:
    return {s.value for s in StreamChannelName}


def _required_keys() -> set[str]:
    """The full enforced key inventory the reference MUST cover."""
    return (
        _metric_names()
        | _stream_channels()
        | _activity_columns()
        | _wellness_columns()
        | _signature_columns()
    )


def test_reference_doc_exists() -> None:
    """The reference document ships with the repo (its existence is part of the contract)."""
    assert _DOC_PATH.is_file(), f"{_DOC_PATH} is missing — the metrics reference MUST ship"


def test_every_metric_name_is_documented() -> None:
    """Every member of the closed metric vocabulary has a reference entry."""
    missing = sorted(_metric_names() - _documented_keys())
    assert not missing, f"MetricName members without a docs/METRICS.md entry: {missing}"


def test_every_stream_channel_is_documented() -> None:
    """Every canonical stream channel has a reference entry."""
    missing = sorted(_stream_channels() - _documented_keys())
    assert not missing, f"stream channels without a docs/METRICS.md entry: {missing}"


def test_every_canonical_activity_column_is_documented() -> None:
    """Every canonical, non-plumbing ``activity`` column has a reference entry."""
    missing = sorted(_activity_columns() - _documented_keys())
    assert not missing, f"activity columns without a docs/METRICS.md entry: {missing}"


def test_every_canonical_wellness_column_is_documented() -> None:
    """Every canonical, non-plumbing ``daily_wellness`` column has a reference entry."""
    missing = sorted(_wellness_columns() - _documented_keys())
    assert not missing, f"daily_wellness columns without a docs/METRICS.md entry: {missing}"


def test_every_canonical_signature_column_is_documented() -> None:
    """Every canonical, non-plumbing ``fitness_signature`` column has a reference entry."""
    missing = sorted(_signature_columns() - _documented_keys())
    assert not missing, f"fitness_signature columns without a docs/METRICS.md entry: {missing}"


def test_full_inventory_is_covered() -> None:
    """The union of all enforced surfaces is fully covered — the single completeness bar."""
    missing = sorted(_required_keys() - _documented_keys())
    assert not missing, f"canonical keys without a docs/METRICS.md entry: {missing}"


def test_exclusions_are_real_columns() -> None:
    """Every excluded key is an actual column (guards against stale exclusions drifting)."""
    activity = {c.key for c in Activity.__table__.columns}
    wellness = {c.key for c in DailyWellness.__table__.columns}
    signature = {c.key for c in FitnessSignature.__table__.columns}
    assert activity >= _ACTIVITY_EXCLUDE, _ACTIVITY_EXCLUDE - activity
    assert wellness >= _WELLNESS_EXCLUDE, _WELLNESS_EXCLUDE - wellness
    assert signature >= _SIGNATURE_EXCLUDE, _SIGNATURE_EXCLUDE - signature
