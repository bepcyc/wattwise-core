"""Chart-ready wire shapes shared by the performance + activities surfaces (SCHEMA-R8/R9).

Extracted so both read routers compose the same source-agnostic envelope without
re-declaring it (and to keep each router within the module-size ceiling, QUAL-R9). No
field is source-shaped or carries a provider name (AUTH-R15/ANL-R1); data fidelity is
the SCHEMA-R9 ``coverage`` only.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from pydantic import BaseModel


class CoverageDescriptor(BaseModel):
    """Source-agnostic per-point/scalar coverage descriptor (SCHEMA-R9; no source name)."""

    present: bool
    fidelity: str
    gap_fraction: float = 0.0
    disputed: bool = False
    provisional: bool = False
    substitution: dict[str, Any] | None = None


class SeriesPoint(BaseModel):
    """One chart point (SCHEMA-R8): an X-axis key, ``label``, named values, coverage.

    The per-activity variant also carries ``activity_id`` so two activities on the
    same calendar day are uniquely addressable (Coggan/W'balance/decoupling/TRIMP).
    """

    local_date: _dt.date | None = None
    duration_s: int | None = None
    activity_id: str | None = None
    label: str
    values: dict[str, float | None]
    coverage: CoverageDescriptor


class ChartSeries(BaseModel):
    """Chart-ready time-series envelope (API-R31): items + precomputed ``summary``."""

    items: list[SeriesPoint]
    x_axis: str
    method: str
    summary: dict[str, Any]
    coverage: CoverageDescriptor
    computed_at: _dt.datetime


__all__ = ["ChartSeries", "CoverageDescriptor", "SeriesPoint"]
