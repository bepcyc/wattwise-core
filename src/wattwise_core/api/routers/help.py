"""Help router — public in-product help content (``/v1/help``, API-R10 / AUTH-R10).

``GET /v1/help/topics`` and ``GET /v1/help/topics/{topic_id}`` are two of the few
endpoints that MAY be unauthenticated (AUTH-R10): static, athlete-native help content
with no per-user data. The topic set is a small, naturally-bounded static catalog, so
the list returns the standard page envelope with a server-bounded ``limit`` (PAGE-R3)
and never an unbounded list. Copy is warm and jargon-free (API-R21) and names no
source/provider internals beyond what the connections surface itself shows.

Requirement IDs: API-R10 (the Help group), AUTH-R10 (public), API-R21 (athlete-native
copy), PAGE-R3 (server-bounded limit), LIMIT-R1 (rate-limited via the public bucket).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Final

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from wattwise_core.api.deps import PublicRateLimit
from wattwise_core.api.pagination import clamp_limit
from wattwise_core.api.problems import not_found

router = APIRouter(prefix="/v1/help", tags=["help"], dependencies=[PublicRateLimit])


class HelpTopic(BaseModel):
    """One static help topic (API-R10 / AUTH-R10): public, athlete-native copy."""

    model_config = ConfigDict(extra="forbid")

    topic_id: str
    title: str
    body_text: str


class HelpPage(BaseModel):
    """The PAGE-R4 page block of the help-topic list."""

    limit: int
    next_cursor: str | None = None
    has_more: bool


class HelpTopicList(BaseModel):
    """``GET /v1/help/topics``: the bounded help-topic page (PAGE-R3/R4)."""

    data: list[HelpTopic]
    page: HelpPage


#: The externalized, keyed help-topic catalog (QUAL-R13): titles/bodies live in
#: ``api/copy/help.copy.toml``, never inline in logic. Loaded once at import.
_CATALOG_PATH: Final = Path(__file__).resolve().parents[1] / "copy" / "help.copy.toml"


def _load_topics() -> tuple[HelpTopic, ...]:
    """Load the ordered static topic catalog from the externalized copy file."""
    entries = tomllib.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return tuple(
        HelpTopic(topic_id=key, title=str(entry["title"]), body_text=str(entry["text"]))
        for key, entry in entries.items()
    )


TOPICS: Final[tuple[HelpTopic, ...]] = _load_topics()


@router.get("/topics", response_model=HelpTopicList, operation_id="listHelpTopics")
async def list_topics(
    limit: Annotated[int, Query(ge=1, json_schema_extra={"maximum": 200})] = 50,
) -> HelpTopicList:
    """The public help-topic list (AUTH-R10): static content, server-bounded page.

    The catalog is naturally bounded, so one page covers it; ``limit`` is still
    clamped/rejected per PAGE-R3 (a ``limit < 1`` is ``422``, ``> 200`` is clamped) so
    no code path can return an unbounded list.
    """
    bounded = clamp_limit(int(limit))
    rows = list(TOPICS[:bounded])
    has_more = len(TOPICS) > bounded
    return HelpTopicList(
        data=rows, page=HelpPage(limit=bounded, next_cursor=None, has_more=has_more)
    )


@router.get("/topics/{topic_id}", response_model=HelpTopic, operation_id="getHelpTopic")
async def get_topic(topic_id: str) -> HelpTopic:
    """One public help topic by id (AUTH-R10); an unknown id → ``404 not-found``."""
    for topic in TOPICS:
        if topic.topic_id == topic_id:
            return topic
    raise not_found()


__all__ = ["TOPICS", "HelpTopic", "router"]
