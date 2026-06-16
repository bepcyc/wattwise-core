"""Fail-closed <technical_proof> tag strip in the athlete-facing voice pass (COMPOSE-R3, #87).

The compose-node parser already builds a tag-free ``visible_answer``; this is the SECOND,
defense-in-depth strip at the presentation boundary (``enforce_presentation``) — the athlete-facing
guarantee that no ``<technical_proof>`` block, stray fragment, or HTML-escaped tag form ever ships,
on either the text or the HTML channel. A residual ``technical_proof`` token after stripping (a
translated/novel spelling the structured regex missed) trips a construction-diverged bare-substring
backstop that scrubs the body to the warm fallback opener rather than leak.
"""

from __future__ import annotations

import pytest

from wattwise_core.agent.voice import (
    VoicePresentation,
    count_foregrounded_numbers,
    enforce_presentation,
)

pytestmark = pytest.mark.unit

_P = VoicePresentation()


def _enforce(text: str, html: str) -> tuple[str, str]:
    return enforce_presentation(html, text, response_length="standard", presentation=_P)


def test_strips_closed_block_from_text_and_html() -> None:
    """A literal closed block is removed from BOTH the text and the HTML body."""
    text = "<technical_proof>fitness is 5.7 (ctl)</technical_proof>You are building steadily."
    html = "<p>You are building steadily.</p>"
    out_html, out_text = _enforce(text, html)
    assert "technical_proof" not in out_text and "technical_proof" not in out_html
    assert "building steadily" in out_text


def test_strips_html_escaped_tag_form() -> None:
    """The HTML channel escapes '<' → '&lt;'; the strip MUST catch the escaped tag form too.

    answer_html is built as '<p>' + html.escape(text) + '</p>', so a tag that survived into the
    grounded text becomes '&lt;technical_proof&gt;…' in HTML — invisible to a literal-'<' regex.
    """
    escaped_html = (
        "<p>&lt;technical_proof&gt;fitness is 5.7 (ctl)&lt;/technical_proof&gt;You are fresh.</p>"
    )
    out_html, _ = enforce_presentation(
        escaped_html, "You are fresh.", response_length="standard", presentation=_P
    )
    assert "technical_proof" not in out_html
    assert "fresh" in out_html


def test_strips_unclosed_block_and_stray_fragment() -> None:
    """An unclosed opening tag (to end) and an orphan closing fragment are both removed."""
    text = "You are recovered.</technical_proof> <technical_proof>leftover 6.0 (tsb)"
    out_html, out_text = _enforce(text, "<p>You are recovered.</p>")
    assert "technical_proof" not in out_text and "technical_proof" not in out_html


def test_bare_substring_backstop_scrubs_a_surviving_token_to_fallback() -> None:
    """A residual 'technical_proof' token (novel/translated spelling) → scrub to fallback opener.

    The construction-diverged backstop: if the structured strip missed a form but the invariant
    token survives, fail closed to the warm opener rather than leak the evidence layer.
    """
    text = "Your form is good. [technical_proof: tsb 4.8 internal]"
    out_html, out_text = _enforce(text, "<p>Your form is good.</p>")
    assert "technical_proof" not in out_text and "technical_proof" not in out_html
    assert out_text == _P.fallback_lead
    assert "technical_proof" not in out_html


def test_strips_attributed_opener_block() -> None:
    """An attributed opener (`<technical_proof foo>`) has its whole block removed, not leaked."""
    text = (
        '<technical_proof lang="en">internal reasoning here; ctl 5.7'
        "</technical_proof>You are fresh."
    )
    out_html, out_text = _enforce(text, "<p>You are fresh.</p>")
    assert "technical_proof" not in out_text and "technical_proof" not in out_html
    assert "internal reasoning" not in out_text
    assert "fresh" in out_text


def test_numbers_inside_block_are_not_counted_toward_density_cap() -> None:
    """The tag strip runs BEFORE the VOICE-R7 number-density count, so block numbers don't count."""
    block = "<technical_proof>a 1 (m); b 2 (m); c 3 (m); d 4 (m); e 5 (m)</technical_proof>"
    text = f"{block}You are on track."
    _, out_text = _enforce(text, "<p>You are on track.</p>")
    assert count_foregrounded_numbers(out_text) == 0
    assert "technical_proof" not in out_text
