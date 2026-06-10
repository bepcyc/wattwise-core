"""The fixed OSS connectable-source catalog (API-R42).

OSS connects exactly two archetypes — a connectionless activity-file importer
(``file_upload``) and one ``api_key`` source (Intervals.icu). OAuth-redirect connectors
are a commercial overlay (COMM-R18) and are deliberately absent. Held here as data so the
connections router stays within the module-size ceiling (QUAL-R9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from wattwise_core.domain.enums import AuthArchetype

#: The single ``api_key`` source key the OSS catalog connects (Intervals.icu, doc 30).
INTERVALS_SOURCE_KEY: Final = "intervals_icu"

#: The built-in connectionless file-upload source key (LIN-R1.1, doc 30).
FILE_IMPORT_SOURCE_KEY: Final = "file_import"

#: The activity-file formats the OSS importer accepts (API-R33; routes to imports).
ACCEPTED_FILE_FORMATS: Final[tuple[str, ...]] = (".fit", ".fit.gz", ".gpx", ".tcx")


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One connectable source in the OSS catalog (API-R42).

    ``source`` is the machine key (the AUTH-R15 source-name exception applies on this
    surface). ``connect_hint`` is short athlete-facing copy (API-R21) telling the athlete
    what connecting this source does — never a URL and never jargon.
    """

    source: str
    display_name: str
    auth_archetype: AuthArchetype
    connect_hint: str


#: The fixed OSS catalog (API-R42): a ``file_upload`` importer + one ``api_key`` source.
OSS_CATALOG: Final[tuple[CatalogEntry, ...]] = (
    CatalogEntry(
        source=FILE_IMPORT_SOURCE_KEY,
        display_name="Activity files",
        auth_archetype=AuthArchetype.FILE_UPLOAD,
        connect_hint="Upload a ride or run file from your watch or another app.",
    ),
    CatalogEntry(
        source=INTERVALS_SOURCE_KEY,
        display_name="Intervals.icu",
        auth_archetype=AuthArchetype.API_KEY,
        connect_hint="Connect with your Intervals.icu key to bring your training in.",
    ),
)

#: Catalog index by source key for O(1) lookup on initiate/complete/reconnect.
CATALOG_BY_SOURCE: Final[dict[str, CatalogEntry]] = {e.source: e for e in OSS_CATALOG}


__all__ = [
    "ACCEPTED_FILE_FORMATS",
    "CATALOG_BY_SOURCE",
    "FILE_IMPORT_SOURCE_KEY",
    "INTERVALS_SOURCE_KEY",
    "OSS_CATALOG",
    "CatalogEntry",
]
