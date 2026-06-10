"""Metric-equivalence resolution: natural label -> canonical MetricName (§16 / GROUND-R2).

The focused sibling of :mod:`wattwise_core.agent.capabilities` (QUAL-R9 size split) that owns the
metric-equivalence layer the spec mandates (§16 metric-equivalence classes; SKILL-R1). A real model
answers in natural/aggregate terms — ``"Chronic Training Load (CTL)"``, ``"fitness"``, ``"form"`` —
not as the exact canonical key (``ctl``/``tsb``); without this bridge those names match no
:class:`~wattwise_core.agent.capabilities.MetricName` member, so every NUMBER claim scrubs and the
grounder ABSTAINS on a CORRECT answer (the headline bug). The alias map is loaded CONTENT
(``defaults.toml`` ``[agent.metric_aliases]``, CFG-R1a); this module only normalizes + looks up, so
an unknown label still resolves to ``None`` (fail-closed, GROUND-R3).
"""

from __future__ import annotations

from collections.abc import Mapping

from wattwise_core.agent.capabilities_metrics import MetricName


def _normalize_metric_label(label: str) -> str:
    """Fold a claimed metric label for alias lookup (case/whitespace-insensitive)."""
    return " ".join(label.casefold().split())


class MetricEquivalence:
    """Resolve a natural metric label to a canonical :class:`MetricName` (§16 / GROUND-R2).

    The alias map is loaded CONTENT (``[agent.metric_aliases]``, CFG-R1a), NOT hardcoded here —
    this class only normalizes + looks up. An already-canonical label resolves to itself; an
    unknown label resolves to ``None`` so the grounder still fails closed (GROUND-R3) for a metric
    the engine genuinely does not expose.
    """

    def __init__(self, aliases: Mapping[str, str]) -> None:
        # Pre-fold the loaded alias keys once; values are canonical MetricName members.
        self._aliases: dict[str, str] = {_normalize_metric_label(k): v for k, v in aliases.items()}

    def canonical_key(self, metric: str) -> str | None:
        """Return the canonical metric key for ``metric``, or ``None`` if unrecognized.

        Resolution tries, in order, the WHOLE folded label then a small set of structural
        variants a real model emits around a canonical/aliased term — because the claim
        extractor often captures the metric label WITH the model's inline gloss, e.g.
        ``"ctl (chronic training load / fitness)"`` or ``"chronic training load (ctl)"``:

        1. the folded label is itself a canonical :class:`MetricName` member;
        2. the folded label is a loaded alias (``[agent.metric_aliases]``, §16);
        3. with a trailing ``(...)`` gloss stripped, the remaining head resolves (1)/(2);
        4. a ``(...)`` gloss whose own content resolves (1)/(2) — e.g. ``"fitness (ctl)"``.

        Every variant resolves ONLY through the canonical enum or the loaded alias map, so an
        unknown metric still yields ``None`` (fail-closed, GROUND-R3); the variants only let a
        glossed restatement of a KNOWN metric ground, never invent a new one.
        """
        folded = _normalize_metric_label(metric)
        for candidate in self._variants(folded):
            resolved = self._resolve_exact(candidate)
            if resolved is not None:
                return resolved
        return None

    def _resolve_exact(self, folded: str) -> str | None:
        """Resolve one already-folded candidate via the canonical enum / loaded alias map."""
        try:
            return MetricName(folded).value
        except ValueError:
            pass
        mapped = self._aliases.get(folded)
        if mapped is None:
            return None
        try:
            # A misconfigured alias pointing at a non-canonical key fails closed (GROUND-R3).
            return MetricName(mapped).value
        except ValueError:
            return None

    @staticmethod
    def _variants(folded: str) -> tuple[str, ...]:
        """Structural variants of a folded label to try (whole, gloss-stripped, gloss-body)."""
        variants = [folded]
        head, sep, rest = folded.partition("(")
        if sep:
            stripped = head.strip()
            if stripped:
                variants.append(stripped)
            gloss = rest.split(")", 1)[0].strip()
            if gloss:
                variants.append(gloss)
        return tuple(variants)


__all__ = ["MetricEquivalence"]
