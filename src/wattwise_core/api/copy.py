"""Externalized, keyed user-facing copy catalog for the API surface (QUAL-R13(c)/(g)).

Every athlete-facing string the API layer emits (problem ``errors[].message`` copy)
resolves through THIS catalog by a stable key — never an inline literal in logic
(the ``copy-orphan-literal`` lint enforces the call-site side). The catalog lives in
``api/locale/<lang>.copy.json`` (the QUAL-R11 zone (a) i18n path the content/copy
lint scans), one whole, translatable entry per key and locale: localized copy is
assembled from complete entries, never concatenated fragments.

Locale completeness is FAIL-CLOSED at load (QUAL-R13(g)): every key MUST exist in
every supported locale; a missing key raises at first use rather than surfacing an
untranslated string. An unsupported locale falls back to English (LANG-R4).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Final

_LOCALE_DIR: Final = Path(__file__).parent / "locale"

#: The supported copy locales (LANG-R1); ``en`` is the LANG-R4 fallback.
SUPPORTED_LOCALES: Final[tuple[str, ...]] = ("en", "de", "ru")
_FALLBACK: Final = "en"


@lru_cache(maxsize=1)
def _catalogs() -> dict[str, dict[str, dict[str, str]]]:
    """Load every locale catalog once and validate cross-locale completeness."""
    loaded: dict[str, dict[str, dict[str, str]]] = {}
    for locale in SUPPORTED_LOCALES:
        path = _LOCALE_DIR / f"{locale}.copy.json"
        loaded[locale] = json.loads(path.read_text(encoding="utf-8"))
    reference = set(loaded[_FALLBACK])
    for locale, catalog in loaded.items():
        missing = reference.symmetric_difference(catalog)
        if missing:  # fail-closed: a key absent from any locale is a build defect
            raise KeyError(
                f"copy catalog '{locale}' is out of sync with '{_FALLBACK}': "
                f"{sorted(missing)} (QUAL-R13(g): every key in every locale)"
            )
    return loaded


def message(key: str, locale: str = _FALLBACK) -> str:
    """Resolve one user-facing copy entry by stable key (QUAL-R13(c)).

    An unsupported ``locale`` falls back to English (LANG-R4); an unknown ``key``
    raises ``KeyError`` (fail-closed — never an empty or fabricated sentence).
    """
    catalogs = _catalogs()
    catalog = catalogs.get(locale, catalogs[_FALLBACK])
    entry = catalog.get(key)
    if entry is None:
        raise KeyError(f"unknown copy catalog key: {key!r}")
    return entry["text"]


__all__ = ["SUPPORTED_LOCALES", "message"]
