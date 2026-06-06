"""Chart decimation that preserves the canonical extrema/shape (API-R48/R49).

Two simplifiers the activities surface uses to bound chart payloads without lying:

- :func:`minmax_index` — a uniform-stride index set with each channel's global
  min/max forced in, so the decimated series never drops the canonical extremum and a
  rendered chart never contradicts ``max_power_w`` (API-R48 / ANL-R8b).
- :func:`rdp_simplify` — Ramer-Douglas-Peucker polyline simplification that keeps the
  corners/turns of a GPS track (API-R49), with the tolerance derived from the
  ``max_points`` budget.

Keeping these out of the router both honours the size ceiling (QUAL-R9) and makes the
``decimation.algorithm`` label match what actually ran.
"""

from __future__ import annotations

from collections.abc import Sequence


def uniform_index(length: int, max_points: int) -> list[int]:
    """A uniform-stride index set of at most ``max_points`` over ``[0, length)``."""
    if length == 0:
        return []
    if max_points >= length:
        return list(range(length))
    step = length / max_points
    return sorted({int(i * step) for i in range(max_points)} | {length - 1})


def _extrema_indices(values: Sequence[object]) -> set[int]:
    """The indices of the global min and global max numeric sample (extrema, API-R48)."""
    nums = [(i, float(v)) for i, v in enumerate(values) if isinstance(v, int | float)]
    if not nums:
        return set()
    return {min(nums, key=lambda p: p[1])[0], max(nums, key=lambda p: p[1])[0]}


def minmax_index(
    length: int, max_points: int, channels: Sequence[Sequence[object]]
) -> list[int]:
    """A decimation index set that PRESERVES every channel's global min/max (API-R48).

    Starts from a uniform-stride sample and unions in the global-extrema indices of each
    channel, so the canonical max/min sample of every channel survives decimation (the
    rendered chart never contradicts the scalar ``max_power_w``). Returned sorted and
    de-duplicated.
    """
    idx = set(uniform_index(length, max_points))
    if not idx:
        return []
    for channel in channels:
        idx |= _extrema_indices(channel)
    return sorted(i for i in idx if 0 <= i < length)


def _perp_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    """Perpendicular distance from ``point`` to the segment ``start``→``end``."""
    (px, py), (ax, ay), (bx, by) = point, start, end
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return float(((px - ax) ** 2 + (py - ay) ** 2) ** 0.5)
    return float(abs(dy * px - dx * py + bx * ay - by * ax) / ((dx * dx + dy * dy) ** 0.5))


def _rdp(points: Sequence[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Recursive Ramer-Douglas-Peucker on a coordinate sequence (shape-preserving)."""
    if len(points) < 3:
        return list(points)
    start, end = points[0], points[-1]
    dmax, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _perp_distance(points[i], start, end)
        if d > dmax:
            dmax, index = d, i
    if dmax <= epsilon:
        return [start, end]
    left = _rdp(points[: index + 1], epsilon)
    right = _rdp(points[index:], epsilon)
    return left[:-1] + right


def rdp_simplify(
    points: Sequence[tuple[float, float]], max_points: int
) -> list[tuple[float, float]]:
    """RDP-simplify a GPS polyline to roughly ``max_points`` while keeping corners (API-R49).

    The tolerance ``epsilon`` is grown until the simplified track fits the point budget,
    so turns are kept (unlike uniform stride) and the advertised ``rdp`` algorithm is
    truthful. A track already within budget is returned unchanged.
    """
    pts = list(points)
    if len(pts) <= max_points or len(pts) < 3:
        return pts
    epsilon = 1e-6
    simplified = _rdp(pts, epsilon)
    while len(simplified) > max_points and epsilon < 1.0:
        epsilon *= 2
        simplified = _rdp(pts, epsilon)
    return simplified


__all__ = ["minmax_index", "rdp_simplify", "uniform_index"]
