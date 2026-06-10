"""Schema-migration state probe for readiness (RUN-R6 / OBS-R6.2).

RUN-R6: schema changes are applied by versioned migrations GATED IN READINESS — a
service MUST NOT serve traffic against an unmigrated schema. This module gives the
readiness probe that gate:

- :func:`expected_head` resolves the repository's newest migration revision by parsing
  the versioned migration scripts (the ``revision`` / ``down_revision`` linkage), without
  importing Alembic's runtime. The scripts directory is located from (in order) the
  ``WATTWISE_MIGRATIONS_DIR`` environment override, the process working directory
  (``./migrations/versions`` — the deployment layout next to ``alembic.ini``), and the
  source checkout relative to this package. When no scripts are locatable (a stripped
  runtime image), the probe degrades to "some revision is stamped".
- :func:`migrations_applied` reads the database's stamped ``alembic_version`` through
  the portable query-builder (no vendor SQL, RUN-R7) and compares it to the expected
  head. An unstamped database (no version table / no row) is NEVER ready.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Column, MetaData, String, Table, select
from sqlalchemy.ext.asyncio import AsyncSession

_REVISION_RE = re.compile(r'^revision:\s*str\s*=\s*"(?P<rev>[^"]+)"', re.MULTILINE)
_DOWN_RE = re.compile(
    r'^down_revision:\s*str\s*\|\s*None\s*=\s*(?P<down>"[^"]+"|None)', re.MULTILINE
)

#: The Alembic bookkeeping table, declared portably (a probe-only metadata; never created here).
_ALEMBIC_VERSION = Table(
    "alembic_version", MetaData(), Column("version_num", String(32), primary_key=True)
)


def _candidate_dirs() -> list[Path]:
    """The ordered locations the versioned migration scripts may live in."""
    candidates: list[Path] = []
    override = os.environ.get("WATTWISE_MIGRATIONS_DIR")
    if override:
        candidates.append(Path(override))
    candidates.append(Path.cwd() / "migrations" / "versions")
    # The source checkout: src/wattwise_core/persistence/ -> repo root /migrations/versions.
    candidates.append(Path(__file__).resolve().parents[3] / "migrations" / "versions")
    return candidates


@lru_cache(maxsize=1)
def expected_head() -> str | None:
    """The newest migration revision id, or ``None`` when no scripts are locatable.

    The head is the revision that no other script names as its ``down_revision`` —
    the same linkage Alembic resolves. Cached for the process lifetime (the scripts
    are immutable build artifacts).
    """
    for directory in _candidate_dirs():
        if not directory.is_dir():
            continue
        revisions: set[str] = set()
        downs: set[str] = set()
        for script in directory.glob("*.py"):
            text = script.read_text(encoding="utf-8")
            rev = _REVISION_RE.search(text)
            if rev is None:
                continue
            revisions.add(rev.group("rev"))
            down = _DOWN_RE.search(text)
            if down is not None and down.group("down") != "None":
                downs.add(down.group("down").strip('"'))
        heads = revisions - downs
        if len(heads) == 1:
            return heads.pop()
    return None


async def migrations_applied(session: AsyncSession) -> bool:
    """True iff the database is stamped at the expected migration head (RUN-R6).

    An unreadable/absent ``alembic_version`` (unmigrated schema) is ``False`` — the
    instance reports not-ready and never serves an unmigrated schema. When the scripts
    are not locatable at runtime, the gate degrades to "a revision is stamped" (still
    refusing a never-migrated database).
    """
    try:
        stamped = (
            await session.execute(select(_ALEMBIC_VERSION.c.version_num))
        ).scalar_one_or_none()
    except Exception:  # no alembic_version table → unmigrated → not ready (RUN-R6)
        return False
    if stamped is None:
        return False
    head = expected_head()
    return True if head is None else stamped == head


__all__ = ["expected_head", "migrations_applied"]
