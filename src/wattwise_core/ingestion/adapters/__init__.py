"""OSS ingestion adapters (ADP-R*).

The OSS engine ships exactly two ingestion surfaces (doc 30, COMM-R18):
``file_upload`` (FIT/GPX/TCX) and Intervals.icu (``api_key``). Each is registered on
the ``wattwise_core.adapters`` entry-point group; adding a source is one adapter
(ROAD-R6) with no consumer change.
"""

from __future__ import annotations

__all__: list[str] = []
