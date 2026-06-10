"""Metric-equivalence class registry (doc 20 §7.5 DM-SUB-R1/R2/R3/R4).

For each canonical channel/metric there may be a DECLARED equivalence class: an
ORDERED set of provider-metrics that can satisfy it, each tagged with a fidelity tier
from the SINGLE ranked GAP-R2 ``fidelity`` enum, a documented semantic-equivalence
note, and a fidelity penalty applied when it substitutes for the top member
(DM-SUB-R1). The classes are EXTERNALIZED configuration — loaded from the packaged
``defaults.toml`` ``[[canonical.equivalence_class]]`` tables, overridable by an
operator file via ``WATTWISE_EQUIVALENCE_CLASSES_FILE`` — never inline engine
constants.

In-class resolution rides the SAME CONF-R2 total order over the SAME tier tokens
(DM-SUB-R2): this module adds no second vocabulary, it only declares which members a
channel admits and what the class's TOP tier is. :func:`substitution_for` is the
DM-SUB-R4 surfacing hook the canonical writer calls: a winner below the class top tier
yields the ``substituted`` coverage marker carrying ``{class, from_fidelity}`` (the
displaced top tier) — never silently badged at the winner's own tier. A channel with
NO declared class is its own degenerate single-member class: no substitution marker is
ever fabricated for it.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from wattwise_core.domain.coverage import Substitution
from wattwise_core.domain.enums import TRUST_TIER_ORDER, Fidelity, trust_rank

_DEFAULTS_PATH = Path(__file__).parents[1] / "config" / "defaults.toml"
_OVERRIDE_ENV = "WATTWISE_EQUIVALENCE_CLASSES_FILE"


@dataclass(frozen=True, slots=True)
class ClassMember:
    """ONE provider-metric admitted by an equivalence class (DM-SUB-R1).

    ``tier`` is drawn from the ranked GAP-R2 ``fidelity`` enum (never ``substituted``
    or ``absent_*`` — those are resolution outcomes, not class-member tiers);
    ``note`` is the mandated semantic-equivalence documentation; ``penalty`` is the
    documented fidelity penalty applied when this member substitutes for the top one.
    """

    metric: str
    tier: Fidelity
    note: str
    penalty: str


@dataclass(frozen=True, slots=True)
class EquivalenceClass:
    """The declared, ORDERED member set for one canonical channel (DM-SUB-R1)."""

    channel: str
    members: tuple[ClassMember, ...]

    @property
    def top_tier(self) -> Fidelity:
        """The class's highest declared fidelity tier (the DM-SUB-R4 reference point)."""
        return min(self.members, key=lambda m: trust_rank(m.tier)).tier


def _classes_path() -> Path:
    """The classes file: the operator override when set, else the packaged defaults."""
    override = os.environ.get(_OVERRIDE_ENV)
    return Path(override) if override else _DEFAULTS_PATH


def _parse_member(raw: dict[str, object], channel: str) -> ClassMember:
    """Parse + validate one declared member; reject a non-ranked tier (DM-SUB-R1)."""
    tier = Fidelity(str(raw["tier"]))
    if tier not in TRUST_TIER_ORDER:
        raise ValueError(
            f"equivalence class {channel!r}: member tier {tier.value!r} is a resolution "
            "outcome, not a ranked class-member tier (DM-SUB-R1)"
        )
    return ClassMember(
        metric=str(raw["metric"]),
        tier=tier,
        note=str(raw["note"]),
        penalty=str(raw["penalty"]),
    )


@lru_cache(maxsize=4)
def _load(path: Path) -> dict[str, EquivalenceClass]:
    """Load + validate every declared class from ``path`` (fail-closed on bad config)."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    out: dict[str, EquivalenceClass] = {}
    for raw in data.get("canonical", {}).get("equivalence_class", []):
        channel = str(raw["channel"])
        members = tuple(_parse_member(m, channel) for m in raw.get("members", []))
        if not members:
            raise ValueError(f"equivalence class {channel!r} declares no members (DM-SUB-R1)")
        out[channel] = EquivalenceClass(channel=channel, members=members)
    return out


def equivalence_classes() -> dict[str, EquivalenceClass]:
    """Every declared equivalence class, keyed by canonical channel (DM-SUB-R1)."""
    return _load(_classes_path())


def class_for(channel: str) -> EquivalenceClass | None:
    """The declared class for ``channel``, or ``None`` (degenerate own-class channel)."""
    return equivalence_classes().get(channel)


def substitution_for(channel: str, winning_tier: Fidelity) -> Substitution | None:
    """The DM-SUB-R4 substitution marker for a resolved winner, or ``None``.

    When ``channel`` carries a declared class and the winner's tier ranks BELOW the
    class's top tier, the canonical value is a SUBSTITUTION: coverage ``fidelity`` must
    become ``substituted`` and carry ``{class, from_fidelity}`` recording the displaced
    higher tier, so a client badges "reduced precision" (DM-SUB-R4). A top-tier winner,
    or a channel with no declared class, yields ``None`` — a substitution marker is
    surfaced only when real, never fabricated.
    """
    cls = class_for(channel)
    if cls is None:
        return None
    top = cls.top_tier
    if trust_rank(winning_tier) <= trust_rank(top):
        return None
    return Substitution(equivalence_class=cls.channel, from_fidelity=top)


__all__ = [
    "ClassMember",
    "EquivalenceClass",
    "class_for",
    "equivalence_classes",
    "substitution_for",
]
