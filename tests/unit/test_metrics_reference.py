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

The gate is **bidirectional**. Forward: every live key in the derived inventory must have a
documented entry (a *missing* entry fails). Reverse: every key documented in an *implemented*
section must exist in the derived inventory or in a justified exclusion list (a *phantom*
entry — a key documented as live that the code does not expose — fails and is named).

A document ``#`` heading whose text matches :data:`_UPCOMING_HEADING_RE` opens an
"arriving in an upcoming release" region. Keys documented under such a region describe an
incoming feature that current builds do not yet expose; they are parsed explicitly and are
**exempt** from the reverse (phantom) check, so the doc can fully describe a forthcoming
field without that field having to exist in the code yet.

Each excluded key below is justified in a comment: it is identity/plumbing/audit
machinery a reader never reads as a measured value, not a collected parameter or computed
metric, so documenting it would dilute the reference rather than complete it.

The gate also self-polices doc hygiene: the reference is a public, athlete-facing document,
so it MUST carry **zero** internal spec-requirement identifiers (the ``ABC-R12`` / ``ABC-T3``
shape). :func:`test_reference_doc_has_no_internal_spec_ids` makes that a permanent guard.
"""

from __future__ import annotations

import re
from pathlib import Path

from wattwise_core.agent.capabilities_metrics import MetricName
from wattwise_core.domain.enums import StreamChannelName
from wattwise_core.persistence.models import Activity, FitnessSignature
from wattwise_core.persistence.models.wellness import DailyWellness

_DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "METRICS.md"

# A ``#``-level heading opening the "arriving in an upcoming release" region. Keys documented
# beneath it describe an incoming feature current builds do not expose yet, so they are exempt
# from the reverse (phantom) check. Matched on the heading text, case-insensitively.
_UPCOMING_HEADING_RE = re.compile(r"^#+\s.*\bupcoming\b", re.IGNORECASE)

# The internal spec-requirement-ID shape (e.g. ``TSS-R1``, ``GROUND-T3``). A public,
# athlete-facing doc must carry NONE of these. Allowlist holds tokens that legitimately match
# the pattern but are not spec IDs (kept empty — no such token exists in the reference today;
# e.g. "FIT.GZ" does not match this pattern at all).
_SPEC_ID_RE = re.compile(r"[A-Z]{2,6}-[RT][0-9]+")
_SPEC_ID_ALLOWLIST: frozenset[str] = frozenset()

# Keys that legitimately appear as a documented *implemented-section* heading but are NOT
# members of the derived live inventory enumerated by ``_required_keys`` (the closed metric
# vocabulary, stream channels, and canonical table columns). Each is a real, live,
# reader-facing value the engine computes and exposes through the per-activity analytics
# surface (the load-metrics bundle and the power/HR analytics) — a surface this completeness
# test does not enumerate field-by-field, because those analytics dataclasses are not safe to
# import here (circular import at module load). They are therefore listed explicitly, with a
# justification, and are accepted by the reverse (phantom) check. Anything documented as live
# and NOT here and NOT in ``_required_keys`` fails the reverse check and is named.
_DOCUMENTED_NON_INVENTORY: frozenset[str] = frozenset(
    {
        "tss",  # computed power training-stress score (per-activity load bundle)
        "hr_load",  # computed heart-rate load (per-activity load bundle)
        "hr_load_zonal",  # computed zone-weighted HR load (analytics, on request/default)
        "load_model",  # honest label of which load member produced the load (load bundle)
        "tss_per_hour",  # computed load density (per-activity load bundle)
        "np",  # computed normalized power (per-activity load bundle)
        "if_",  # computed intensity factor (per-activity load bundle)
        "efficiency_factor",  # computed aerobic efficiency (per-activity load bundle)
        "variability_index",  # computed pacing variability (per-activity load bundle)
        "intensity_class",  # computed intensity band label (per-activity load bundle)
        "decoupling",  # computed aerobic decoupling (per-activity analytics)
        "power_curve",  # computed mean-maximal-power curve (power analytics)
        "wbal",  # computed W'balance series (power analytics)
        "trimp",  # computed heart-rate training impulse (HR analytics)
    }
)

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


def _heading_keys_by_section() -> tuple[set[str], set[str]]:
    """Partition documented anchor-heading keys into (implemented, upcoming).

    An anchor heading is a ``###`` line carrying the exact key(s) in backticks, e.g.
    ``### Normalized Power (`np`)`` or a shared heading naming several flag keys
    ``### Channel-presence flags (`has_power` / `has_hr` / ...)``. Every backticked token
    on a heading line counts, so a key is only credited when it is named in a heading, not
    merely mentioned in prose.

    A key belongs to the *upcoming* set when its heading falls under a ``#``-level heading
    matching :data:`_UPCOMING_HEADING_RE`; otherwise it is *implemented*. The upcoming region
    runs from its opening heading to end-of-file (the reference places it last).
    """
    text = _DOC_PATH.read_text(encoding="utf-8")
    implemented: set[str] = set()
    upcoming: set[str] = set()
    in_upcoming = False
    for line in text.splitlines():
        # A top-level (``#``/``##``) heading can switch us into/out of the upcoming region.
        if re.match(r"^#{1,2}\s", line):
            in_upcoming = bool(_UPCOMING_HEADING_RE.match(line))
        if line.startswith("###"):
            target = upcoming if in_upcoming else implemented
            target.update(re.findall(r"`([^`]+)`", line))
    return implemented, upcoming


def _documented_keys() -> set[str]:
    """Every canonical key that appears as an anchor heading anywhere in the reference doc.

    The union of implemented and upcoming heading keys — used by the *forward* completeness
    checks (a live key may be documented in either region and still count as covered).
    """
    implemented, upcoming = _heading_keys_by_section()
    return implemented | upcoming


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


def test_no_phantom_implemented_entry() -> None:
    """Reverse gate: every key documented as IMPLEMENTED is live (or justified-excluded).

    A key documented in the implemented sections must be a member of the derived live
    inventory OR appear in the justified :data:`_DOCUMENTED_NON_INVENTORY` allowlist. A
    *phantom* entry — a key documented as live that the code does not yet expose (e.g. a
    field that only lands in a future release but was placed in the implemented index/body
    instead of the upcoming region) — fails this test and is named, the mirror image of the
    forward completeness checks.
    """
    implemented, _upcoming = _heading_keys_by_section()
    allowed = _required_keys() | _DOCUMENTED_NON_INVENTORY
    phantom = sorted(implemented - allowed)
    assert not phantom, (
        "docs/METRICS.md documents these keys in an IMPLEMENTED section, but they are not in "
        f"the live inventory or the justified allowlist (phantom entries): {phantom}. "
        "Move a not-yet-shipped field under the 'upcoming release' region, or add it to the "
        "code surface it claims to come from."
    )


def test_upcoming_region_is_parsed_and_fenced() -> None:
    """The upcoming region exists, is non-empty, and holds exactly the known incoming keys.

    This anchors the section-marker contract the reverse gate relies on: the parser actually
    recognises the 'arriving in an upcoming release' region, and the three not-yet-live
    self-report / RPE-load keys live there (NOT in the implemented index), so they are
    exempt from the phantom check rather than failing it.
    """
    implemented, upcoming = _heading_keys_by_section()
    assert {"perceived_exertion", "feel", "srpe_load"} <= upcoming, (
        "expected the not-yet-live keys to be documented under the upcoming-release region; "
        f"found upcoming keys: {sorted(upcoming)}"
    )
    leaked_into_implemented = {"perceived_exertion", "feel", "srpe_load"} & implemented
    assert not leaked_into_implemented, (
        "not-yet-live keys must NOT appear as implemented-section headings: "
        f"{sorted(leaked_into_implemented)}"
    )


def test_reference_doc_has_no_internal_spec_ids() -> None:
    """Doc-hygiene guard: the public reference carries ZERO internal spec-requirement IDs.

    ``docs/METRICS.md`` is athlete-facing; internal requirement identifiers (the ``ABC-R12`` /
    ``ABC-T3`` shape) must never leak into a public document. This makes that rule a permanent,
    enforced guard: any future edit reintroducing such a token fails here and is named.
    """
    text = _DOC_PATH.read_text(encoding="utf-8")
    matches = sorted(set(_SPEC_ID_RE.findall(text)) - _SPEC_ID_ALLOWLIST)
    assert not matches, (
        "docs/METRICS.md leaks internal spec-requirement IDs (public docs must carry none): "
        f"{matches}"
    )


def test_exclusions_are_real_columns() -> None:
    """Every excluded key is an actual column (guards against stale exclusions drifting)."""
    activity = {c.key for c in Activity.__table__.columns}
    wellness = {c.key for c in DailyWellness.__table__.columns}
    signature = {c.key for c in FitnessSignature.__table__.columns}
    assert activity >= _ACTIVITY_EXCLUDE, _ACTIVITY_EXCLUDE - activity
    assert wellness >= _WELLNESS_EXCLUDE, _WELLNESS_EXCLUDE - wellness
    assert signature >= _SIGNATURE_EXCLUDE, _SIGNATURE_EXCLUDE - signature


def test_documented_non_inventory_allowlist_is_not_stale() -> None:
    """Every justified non-inventory allowlist entry is actually documented as implemented.

    Guards against the allowlist drifting: an entry that is neither in the live inventory nor
    documented as an implemented heading is dead weight (or a typo) and must be removed.
    """
    implemented, _upcoming = _heading_keys_by_section()
    stale = sorted(_DOCUMENTED_NON_INVENTORY - implemented - _required_keys())
    assert not stale, f"stale _DOCUMENTED_NON_INVENTORY entries (not documented/live): {stale}"
