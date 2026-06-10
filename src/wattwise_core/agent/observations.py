"""Stable-id observation projection for conversational follow-ups (doc 50 COACH-R8).

A LEAF helper of the agent family (QUAL-R9 module-size split off :mod:`graph_state`): it projects
the grounder's grounded survivors into the stable-id ``observations`` the production ``ground`` node
writes, so a later turn can ``drill``/``reveal_numbers`` against a specific prior observation
WITHOUT re-stating the question (COACH-R8). It depends only on the closed contracts, never on a
sibling in-flight agent file (ARCH-R21), so it sits strictly BELOW :mod:`graph_state` in imports.

This module adds NO number and certifies no groundedness — it only projects what the deterministic
grounder already grounded (GROUND-R5/-R7); the stable ids it mints are an opaque follow-up handle.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from wattwise_core.agent.contracts import GroundedClaim


def build_observations(survivors: Sequence[GroundedClaim]) -> list[dict[str, Any]]:
    """Project the grounded survivors into stable-id observations a follow-up can target (COACH-R8).

    Every distinct athlete-facing observation MUST carry a STABLE ``observation_id`` so a later turn
    can ``drill``/``reveal_numbers`` against it WITHOUT re-stating the question (COACH-R8): the
    engine attaches the id the deliverable's ``_reveal_observation`` matches by. Each grounded
    survivor (GROUND-R5) becomes one observation carrying its athlete-facing ``text`` and the
    grounded ``{metric, value, as_of}`` citation behind it — the verbatim numbers a ``reveal``
    follow-up surfaces on demand (VOICE-R9), never a new claim. A survivor with no citation is NOT
    observable (no grounded number to reveal), so it is skipped (fail-closed: only grounded, citable
    claims get a drillable handle).

    The id is DETERMINISTIC and stable across turns (:func:`_observation_id`): the SAME grounded
    claim (same canonical record) yields the SAME id, so a follow-up that passes the prior id back
    targets the same observation on the durable thread.
    """
    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for survivor in survivors:
        citation = survivor.citation
        if citation is None:
            continue
        obs_id = _observation_id(citation, survivor.claim.text)
        if obs_id in seen:
            continue
        seen.add(obs_id)
        observations.append(
            {
                "observation_id": obs_id,
                "text": survivor.claim.text,
                "citations": [dict(citation)],
            }
        )
    return observations


def _observation_id(citation: Mapping[str, Any], fallback_text: str) -> str:
    """A stable, deterministic observation id for a grounded claim (COACH-R8).

    Derived from the citation's canonical ``record_id`` when present (a stable canonical reference,
    so the SAME grounded fact gets the SAME id across turns); otherwise from the citation's
    ``canonical_id``/``metric`` or, last resort, the claim text — always a stable function of the
    grounded claim, never a random per-run uuid (a random id would change every turn and a follow-up
    could never target it). The id is an opaque token (no internal metric code is exposed in it,
    VOICE-R2 — it is a hash), prefixed ``obs-`` so a client treats it as a follow-up handle.
    """
    key = (
        citation.get("record_id")
        or citation.get("canonical_id")
        or citation.get("metric")
        or fallback_text
        or "observation"
    )
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]
    return f"obs-{digest}"


__all__ = ["build_observations"]
