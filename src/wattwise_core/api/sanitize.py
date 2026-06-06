"""Server-side HTML sanitization for athlete-facing rich-text fields (SCHEMA-R7).

Any field this API emits that carries model-generated or otherwise rich HTML —
the agent's ``answer_html``, an insight/briefing/digest body, a callout's
``body_html`` — is sanitized HERE, server-side, before it leaves the API. The API
**never** relies on the client to sanitize (SCHEMA-R7); a client that renders the
returned HTML directly must be safe by construction.

The sanitizer is a strict ALLOW-LIST built on :mod:`nh3` (the Rust ``ammonia``
binding): only the small set of formatting tags a coach narrative needs is kept,
every attribute except a vetted few is dropped, and every dangerous construct is
removed — ``<script>``, event-handler attributes (``onerror``/``onclick``/…),
``<iframe>``/``<object>``/``<embed>``, inline ``style`` (so no ``expression()`` or
``url()`` payload survives), and any ``javascript:`` or ``data:`` URI. Anything not
on the allow-list is stripped, so a never-before-seen injection vector fails closed
(it is removed, not passed through).

This module owns ONLY the HTML allow-list. Untrusted source-derived text
(API-R18) is a different concern: it is never promoted to ``*_html`` and is
returned as escaped plain text in a ``*_text`` field by its producer — this
sanitizer is the last line of defence for the ``*_html`` fields specifically.

Requirement IDs: SCHEMA-R7 (HTML sanitization allow-list), API-R13 (``answer_html``
sanitized server-side before return), API-R18 (no executable HTML from untrusted
text — the ``*_html``/``*_text`` split this backstops).
"""

from __future__ import annotations

from typing import Final

import nh3

#: The strict tag allow-list (SCHEMA-R7): the formatting vocabulary a grounded
#: coach narrative uses and nothing structural. No ``<script>``/``<style>``/
#: ``<iframe>``/``<object>``/``<embed>``/``<form>`` — those are NOT listed, so nh3
#: strips them. Headings/lists/emphasis/links/line-structure only.
ALLOWED_TAGS: Final[frozenset[str]] = frozenset(
    {
        "p",
        "br",
        "strong",
        "b",
        "em",
        "i",
        "u",
        "ul",
        "ol",
        "li",
        "span",
        "a",
        "h1",
        "h2",
        "h3",
        "h4",
        "blockquote",
        "code",
        "pre",
        "hr",
        "small",
        "sub",
        "sup",
    }
)

#: The per-tag attribute allow-list (SCHEMA-R7). Deliberately minimal: only an
#: anchor's ``href`` (URI-scheme-restricted below) plus an accessibility ``title``.
#: NO ``style`` (kills ``expression``/``url`` payloads), NO ``class``/``id``, and NO
#: ``on*`` event handlers anywhere — an attribute absent from this map is dropped.
#: The anchor ``rel`` is managed by nh3 itself via ``link_rel`` (it is intentionally
#: NOT listed here; nh3 rejects a hand-listed ``rel`` when ``link_rel`` is set).
ALLOWED_ATTRIBUTES: Final[dict[str, set[str]]] = {
    "a": {"href", "title"},
    "span": {"title"},
    "abbr": {"title"},
}

#: The ONLY URL schemes permitted on an ``href`` (SCHEMA-R7): no ``javascript:``,
#: no ``data:``, no ``vbscript:``. A link with any other scheme has its ``href``
#: dropped by nh3, neutralizing scripting-via-URI.
ALLOWED_URL_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https", "mailto"})


def sanitize_html(raw: str) -> str:
    """Return ``raw`` reduced to the strict SCHEMA-R7 allow-list (inert HTML).

    Drops every tag/attribute/URL-scheme not explicitly allowed above, so the
    result contains no ``<script>``, no event-handler attribute, no inline style,
    no ``<iframe>``/``<object>``, and no ``javascript:``/``data:`` URI. Comments are
    stripped. The output is safe for a client to render verbatim (SCHEMA-R7 /
    API-R13); the API never defers sanitization to the client.
    """
    return nh3.clean(
        raw,
        tags=set(ALLOWED_TAGS),
        attributes={tag: set(attrs) for tag, attrs in ALLOWED_ATTRIBUTES.items()},
        url_schemes=set(ALLOWED_URL_SCHEMES),
        link_rel="noopener noreferrer nofollow",
        strip_comments=True,
    )


def is_inert(html: str) -> bool:
    """True iff ``html`` carries no executable/dangerous construct (test helper).

    A cheap structural check used by the sanitization contract test (SCHEMA-R7):
    asserts the obvious injection markers are gone. It is NOT the sanitizer (that is
    :func:`sanitize_html`); it only verifies a string is already inert.
    """
    lowered = html.lower()
    forbidden = (
        "<script",
        "</script",
        "javascript:",
        "onerror=",
        "onclick=",
        "onload=",
        "<iframe",
        "<object",
        "<embed",
        " style=",
        "expression(",
    )
    return not any(token in lowered for token in forbidden)


__all__ = [
    "ALLOWED_ATTRIBUTES",
    "ALLOWED_TAGS",
    "ALLOWED_URL_SCHEMES",
    "is_inert",
    "sanitize_html",
]
