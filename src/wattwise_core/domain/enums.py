"""Closed canonical enumerations (GBO-R12).

Every enum here is a *closed* canonical vocabulary. Adapters map source vocab onto
these; an unmappable token becomes ``unknown`` / ``other`` plus a lineage note
(MAP-R4) — never a passthrough of the raw source token.

NOTE: ``sport`` and ``sub_sport`` are deliberately NOT modelled as enums — they are
data-driven registries (GBO-R16a / GBO-R16a-ii), so they live in the registry
tables, not here.
"""

from __future__ import annotations

from enum import StrEnum


class Sex(StrEnum):
    """Athlete sex; used only for physiological defaults (GBO-R13)."""

    FEMALE = "female"
    MALE = "male"
    OTHER = "other"
    UNKNOWN = "unknown"


class SampleBasis(StrEnum):
    """Sampling axis of a stream set / channel (GBO-R19/R20).

    ``event`` is reserved for event-spaced channels (e.g. ``rr_intervals_ms``)
    that are exempt from the shared-time-base rule (GBO-R21).
    """

    TIME = "time"
    DISTANCE = "distance"
    EVENT = "event"


class StreamChannelName(StrEnum):
    """Canonical per-sample channel names (GBO-R20). Unit is encoded in the name."""

    POWER_W = "power_w"
    HR_BPM = "hr_bpm"
    CADENCE_RPM = "cadence_rpm"
    SPEED_MPS = "speed_mps"
    ALTITUDE_M = "altitude_m"
    DISTANCE_M = "distance_m"
    LATLNG = "latlng"
    TEMP_C = "temp_c"
    LEFT_RIGHT_BALANCE = "left_right_balance"
    SMO2 = "smo2"
    CORE_TEMP_C = "core_temp_c"
    RESPIRATION_RPM = "respiration_rpm"
    RR_INTERVALS_MS = "rr_intervals_ms"


class DeviceClass(StrEnum):
    """Provenance of measurement — NOT a source name (GBO-R2)."""

    POWERMETER = "powermeter"
    TRAINER = "trainer"
    GPS_WATCH = "gps_watch"
    PHONE = "phone"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class ActivityFileFormat(StrEnum):
    """Verbatim original-file format (RAW-R1)."""

    FIT = "fit"
    GPX = "gpx"
    TCX = "tcx"
    JSON = "json"
    OTHER = "other"


class SourceKind(StrEnum):
    """Data-flow channel of a registered source (LIN-R1).

    Distinct from :class:`AuthArchetype` even though both spell ``file_upload``.
    """

    OAUTH_API = "oauth_api"
    FILE_UPLOAD = "file_upload"
    WEBHOOK = "webhook"
    SCRAPE = "scrape"


class AuthArchetype(StrEnum):
    """Connection auth flow (SCHEMA-R10, doc 60). Consumers branch on this, not on
    the source name (GBO-R48)."""

    OAUTH_REDIRECT = "oauth_redirect"
    API_KEY = "api_key"
    CREDENTIALS = "credentials"
    FILE_UPLOAD = "file_upload"


class ConnectionStatus(StrEnum):
    """Single spec-wide connection-status vocab (GBO-R44)."""

    CONNECTED = "connected"
    REAUTH_REQUIRED = "reauth_required"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class Fidelity(StrEnum):
    """Coverage fidelity (GAP-R2).

    Ranked order (highest first): ``raw_stream > device_computed > platform_computed
    > modeled > summary_only``. ``substituted`` is an *outcome* state, not a ranked
    tier; ``absent_true`` / ``absent_failed`` mean no usable value.
    """

    RAW_STREAM = "raw_stream"
    DEVICE_COMPUTED = "device_computed"
    PLATFORM_COMPUTED = "platform_computed"
    MODELED = "modeled"
    SUMMARY_ONLY = "summary_only"
    SUBSTITUTED = "substituted"
    ABSENT_TRUE = "absent_true"
    ABSENT_FAILED = "absent_failed"


# Ranked trust tiers (CONF-R2 / DM-SUB-R1) — the fidelity ordering used as the
# primary key of resolve_field. ``substituted`` / ``absent_*`` are NOT tiers.
TRUST_TIER_ORDER: tuple[Fidelity, ...] = (
    Fidelity.RAW_STREAM,
    Fidelity.DEVICE_COMPUTED,
    Fidelity.PLATFORM_COMPUTED,
    Fidelity.MODELED,
    Fidelity.SUMMARY_ONLY,
)


def trust_rank(fidelity: Fidelity) -> int:
    """Return a sortable rank where a LOWER number is higher trust (CONF-R2).

    Non-tier fidelities (``substituted`` / ``absent_*``) sort last.
    """
    try:
        return TRUST_TIER_ORDER.index(fidelity)
    except ValueError:
        return len(TRUST_TIER_ORDER)


class Severity(StrEnum):
    """Single severity vocab for Alert + DataHealthIssue (GBO-R12)."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class GapReason(StrEnum):
    """The closed typed-gap reason taxonomy (ING-GAP-R3).

    The complete acquisition/mapping/conflict reason set a partial failure is recorded
    under (ING-GAP-R1/R2). It is exactly the ten members ING-GAP-R3 mandates AT MINIMUM;
    it MUST NOT carry an analytics-dependency reason (those belong to the Analytics
    spec). A field with >=1 contributing candidate is NEVER a gap — material
    disagreement is the coverage ``disputed`` flag, not a reason here (PRV-R5).
    """

    AUTH_REVOKED = "auth_revoked"
    NEEDS_REAUTH = "needs_reauth"
    RATE_LIMITED = "rate_limited"
    SOURCE_UNAVAILABLE = "source_unavailable"
    DISCOVERY_INCOMPLETE = "discovery_incomplete"
    FETCH_FAILED = "fetch_failed"
    SCHEMA_MISMATCH = "schema_mismatch"
    MAPPING_FIELD_MISSING = "mapping_field_missing"
    SOURCE_REMOVED = "source_removed"
    COVERAGE_STALE = "coverage_stale"


class GapState(StrEnum):
    """A typed gap's lifecycle state (ING-GAP-R4).

    A gap opens on a partial failure (ING-GAP-R5) and a transient one is self-healing:
    a later successful sync covering the same range closes it. A consumer MUST be able
    to distinguish ``open`` from ``closed`` (ING-GAP-R4).
    """

    OPEN = "open"
    CLOSED = "closed"


class SignatureOrigin(StrEnum):
    """How a ``fitness_signature`` was obtained (GBO-R26)."""

    MEASURED = "measured"
    MODELED = "modeled"
    USER_ENTERED = "user_entered"
    SOURCE_PROVIDED = "source_provided"


class ZoneKind(StrEnum):
    """Training-zone family (GBO-R13d)."""

    POWER = "power"
    HR = "hr"


class ZoneBasis(StrEnum):
    """How ``training_zone_set`` boundaries are expressed (GBO-R13d)."""

    ABSOLUTE = "absolute"  # watts / bpm
    RELATIVE = "relative"  # fraction of cp_w / ftp_w / threshold_hr_bpm


class StreamSetKind(StrEnum):
    """Which parent owns a ``stream_channel`` row (GBO-R20).

    A single ``stream_channel`` table serves both ``activity_stream_set`` and
    ``wellness_stream_set``; this discriminator records which parent a row belongs
    to (the row's ``stream_set_id`` carries no hard FK because it has two possible
    parents).
    """

    ACTIVITY = "activity"
    WELLNESS = "wellness"


class HrvStatus(StrEnum):
    """Source-reported HRV status (GBO-R24)."""

    BALANCED = "balanced"
    UNBALANCED = "unbalanced"
    LOW = "low"
    POOR = "poor"
    UNKNOWN = "unknown"


class HrvMethod(StrEnum):
    """Headline time-domain HRV variant pointer (GBO-R24c).

    Points to which sibling field holds the primary value; NEVER relabels a field's
    statistic/unit. DISTINCT from doc 40's spectral ``hrv_spectral_method``.
    """

    RMSSD = "rmssd"
    SDNN = "sdnn"
    PNN50 = "pnn50"


class TrainingStatus(StrEnum):
    """Source-reported training state (GBO-R25) — NOT canonical PMC."""

    DETRAINING = "detraining"
    RECOVERY = "recovery"
    MAINTAINING = "maintaining"
    PRODUCTIVE = "productive"
    PEAKING = "peaking"
    OVERREACHING = "overreaching"
    UNPRODUCTIVE = "unproductive"
    UNKNOWN = "unknown"


class AcwrStatus(StrEnum):
    """Source-reported acute:chronic workload-ratio status (GBO-R25)."""

    LOW = "low"
    OPTIMAL = "optimal"
    HIGH = "high"
    UNKNOWN = "unknown"


class ReadinessVerdict(StrEnum):
    """Canonical readiness/form assessment VERDICT (SCHEMA-R3 / COACH-R1 #2).

    A typed coaching state, NOT a numeric score: there is deliberately no numeric
    ``readiness`` metric. The verdict is a deterministic function of canonical
    metrics (notably TSB/form), so the engine — not the LLM — decides it
    (QA-EVAL-R2.4 / COACH-R3 / EVAL-R5).
    """

    GO = "go"
    MAINTAIN = "maintain"
    EASE = "ease"
    REST = "rest"


class WorkoutTargetType(StrEnum):
    """Target type of a single workout step (GBO-R29)."""

    POWER_W = "power_w"
    POWER_PCT_CP = "power_pct_cp"
    HR_BPM = "hr_bpm"
    HR_PCT_THRESHOLD = "hr_pct_threshold"
    CADENCE_RPM = "cadence_rpm"
    RPE = "rpe"
    OPEN = "open"


class WorkoutStepIntent(StrEnum):
    """Step-level intent of a single workout step (GBO-R29).

    DISTINCT from the day-level :class:`PlanDayIntent`.
    """

    WARMUP = "warmup"
    WORK = "work"
    RECOVERY = "recovery"
    COOLDOWN = "cooldown"
    STEADY = "steady"
    VO2 = "vo2"
    THRESHOLD = "threshold"
    SPRINT = "sprint"
    REST = "rest"


class PlanStatus(StrEnum):
    """Lifecycle of a :class:`~wattwise_core.persistence.models.planning.Plan` (GBO-R30a)."""

    ACTIVE = "active"
    COMPLETED = "completed"
    SUPERSEDED = "superseded"


class PlanDayIntent(StrEnum):
    """Day-level intent of a ``plan_day`` (GBO-R30b).

    DISTINCT from the step-level :class:`WorkoutStepIntent`.
    """

    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    THRESHOLD = "threshold"
    VO2 = "vo2"
    SPRINT = "sprint"
    RACE = "race"
    REST = "rest"
    RECOVERY = "recovery"


class GoalType(StrEnum):
    """Kind of a user-authored training objective (GBO-R36)."""

    EVENT = "event"
    TARGET_METRIC = "target_metric"
    DISTANCE = "distance"
    PROCESS = "process"
    OTHER = "other"


class GoalTargetMetric(StrEnum):
    """Canonical target metric of a :class:`Goal` (GBO-R38)."""

    CP_W = "cp_w"
    FTP_W = "ftp_w"
    DISTANCE_M = "distance_m"
    ELAPSED_TIME_S = "elapsed_time_s"
    VO2MAX = "vo2max"


class GoalStatus(StrEnum):
    """Lifecycle of a :class:`Goal` (GBO-R39); closing sets a terminal status."""

    ACTIVE = "active"
    ACHIEVED = "achieved"
    MISSED = "missed"
    ABANDONED = "abandoned"


class AdjustmentType(StrEnum):
    """Kind of a ``schedule_adjustment`` override (GBO-R40/R42)."""

    MOVE = "move"
    SWAP_WORKOUT = "swap_workout"
    SHORTEN = "shorten"
    LENGTHEN = "lengthen"
    SKIP = "skip"
    REST = "rest"
    OTHER = "other"


class AdjustmentOrigin(StrEnum):
    """Who authored a ``schedule_adjustment`` (GBO-R40)."""

    ATHLETE = "athlete"
    AGENT = "agent"


class AdjustmentStatus(StrEnum):
    """Lifecycle of a ``schedule_adjustment`` (GBO-R42)."""

    PROPOSED = "proposed"
    APPLIED = "applied"
    REJECTED = "rejected"
    REVERTED = "reverted"


class DigestCadence(StrEnum):
    """Digest schedule cadence (GBO-R46)."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class Weekday(StrEnum):
    """Mon-first weekday token (GBO-R46b); IDENTICAL token on the doc 60 wire.

    NO second representation (no ISO 1-7 smallint), NO remapping at projection.
    """

    MON = "mon"
    TUE = "tue"
    WED = "wed"
    THU = "thu"
    FRI = "fri"
    SAT = "sat"
    SUN = "sun"


class DeliveryChannel(StrEnum):
    """Sole canonical delivery-channel vocab (GBO-R46c).

    Referenced verbatim by doc 60's ``DigestSubscribeRequest.channels`` and used by
    :class:`NotificationRoute` (GBO-R49). ``web`` is always-on.
    """

    WEB = "web"
    EMAIL = "email"
    TELEGRAM = "telegram"


class DigestStatus(StrEnum):
    """Lifecycle of a :class:`DigestSubscription` (GBO-R47)."""

    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class GboType(StrEnum):
    """Which canonical type a ``source_candidate`` row maps to (LIN-R2).

    Part of the per-source candidate key; NEVER appears in a canonical key.
    """

    ACTIVITY = "activity"
    ACTIVITY_LAP = "activity_lap"
    ACTIVITY_FILE = "activity_file"
    STREAM_CHANNEL = "stream_channel"
    DAILY_WELLNESS = "daily_wellness"
    WELLNESS_STREAM_SET = "wellness_stream_set"
    FITNESS_SIGNATURE = "fitness_signature"


# The canonical sport codes the OSS engine seeds the registry with (GBO-R16a).
# The registry is data-driven and extensible; these are only the built-in seeds.
SEED_SPORTS: tuple[tuple[str, str, bool], ...] = (
    # (sport_code, display_name, has_mechanical_power)
    ("cycling", "Cycling", True),
    ("running", "Running", False),
    ("swimming", "Swimming", False),
    ("rowing", "Rowing", True),
    ("xc_ski", "Cross-country skiing", False),
    ("strength", "Strength", False),
    ("other", "Other", False),
)
