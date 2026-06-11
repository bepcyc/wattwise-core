"""Deterministic prose checks for athlete-facing eval output (QA-EVAL-R3, QUAL-R13).

Two PROGRAMMATIC, network-free graders the eval suites call instead of trusting an LLM:

* :func:`detect_language` — language-of-output detection (QA-EVAL-R2.8 (a)): script
  detection for Cyrillic (ru) and closed stopword/diacritic evidence for de vs en. A tie
  or zero-evidence text reads as ``"unknown"`` and FAILS the calling case (fail-closed:
  never silently assume the requested language was produced).
* :func:`flesch_kincaid_grade` — the plain-language reading-level ceiling
  (QUAL-R13(h)/(i)): graded by code, never by the LLM judge, so a jargon-dense
  regression trips the gate mechanically.

Requirement IDs: QA-EVAL-R2.8, QA-EVAL-R3, QUAL-R13(h), QUAL-R13(i).
"""

from __future__ import annotations

import re

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_DE_DIACRITIC = re.compile(r"[äöüßÄÖÜ]")
_DE_STOPWORDS = frozenset(
    [
        "der",
        "die",
        "das",
        "und",
        "ist",
        "nicht",
        "dein",
        "deine",
        "bei",
        "liegt",
        "heute",
        "morgen",
        "gut",
        "rund",
        "sieht",
        "leicht",
        "stabil",
        "aus",
        "kein",
        "keine",
        "mit",
        "mehr",
        "noch",
        "sehr",
    ]
)
_EN_STOPWORDS = frozenset(
    [
        "the",
        "your",
        "is",
        "and",
        "with",
        "at",
        "a",
        "of",
        "to",
        "looks",
        "today",
        "this",
        "here",
        "so",
        "on",
        "it",
        "for",
        "steady",
        "morning",
        "around",
        "holding",
        "touch",
        "can't",
    ]
)


def detect_language(text: str) -> str:
    """Detect the OUTPUT language of athlete-facing prose: ``en``/``de``/``ru``/``unknown``.

    Programmatic per QA-EVAL-R3 (a language-of-output check MUST be programmatic):
    Cyrillic script decides ``ru``; German diacritics or a German-stopword majority decide
    ``de``; an English-stopword majority decides ``en``; anything else is ``unknown``.
    """
    if _CYRILLIC.search(text):
        return "ru"
    if _DE_DIACRITIC.search(text):
        return "de"
    words = re.findall(r"[a-zA-Z']+", text.lower())
    de_hits = sum(1 for w in words if w in _DE_STOPWORDS)
    en_hits = sum(1 for w in words if w in _EN_STOPWORDS)
    if de_hits > en_hits:
        return "de"
    if en_hits > de_hits:
        return "en"
    return "unknown"


# Plain-language ceiling for athlete-facing body copy: <= 8th-grade Flesch-Kincaid.
MAX_FK_GRADE = 8.0
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


def _syllables(word: str) -> int:
    """Heuristic syllable count (vowel groups, silent trailing 'e' dropped)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    count = len(_VOWEL_GROUP_RE.findall(w))
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ye")):
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid grade level of athlete-facing prose (deterministic).

    The standard formula 0.39*(words/sentences) + 11.8*(syllables/word) - 15.59 with a
    heuristic syllable counter - adequate to catch jargon-dense, run-on copy drifting
    past the plain-language ceiling, without an LLM in the loop.
    """
    sentences = [chunk for chunk in _SENT_SPLIT_RE.split(text) if chunk.strip()]
    words = _WORD_RE.findall(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * len(words) / len(sentences) + 11.8 * syllables / len(words) - 15.59


__all__ = [
    "MAX_FK_GRADE",
    "detect_language",
    "flesch_kincaid_grade",
]
