"""Lightweight, no-dependency Japanese-vs-rest input heuristics.

These are used by ``dazai-chat`` to decide whether to short-circuit the model
path (e.g. "this is English, reply with the canned 'I only speak Japanese'
line"; "this looks like a single unknown word, ask the user to define it").

The functions here are intentionally pure: they receive raw text and return
small primitive values, so they can be tested in isolation without touching
SQLite, the model, or the tokenizer.
"""

from __future__ import annotations

import re

# A character is treated as Japanese if it is hiragana, katakana, kanji,
# or one of the common Japanese-text punctuation marks that we still treat
# as belonging to the language.
_JAPANESE_CHAR_RE = re.compile(
    r"[ぁ-んァ-ヴー一-龥々〆〇]"
)

# Strip punctuation/symbols/whitespace before deciding whether the residual
# text is Japanese. Punctuation alone shouldn't be treated as "non-Japanese".
_NON_TEXT_RE = re.compile(
    r"[\s。、！？!?…「」『』（）()\[\]{}<>:：,，./\\\-_~^*+=#$%&@|;'\"`]+"
)

# Phrases that signal "I made a typo / pretend I never said that".
TYPO_RETRACTION_PATTERNS: tuple[str, ...] = (
    "間違えた",
    "間違いだ",
    "間違いです",
    "間違った",
    "間違って",
    "打ち間違",
    "誤字",
    "タイポ",
    "typo",
    "ごめん間違",
    "ごめん、間違",
    "違った、",
    "やっぱり違",
    "ごめん打ち",
    "言い間違",
    "間違って入力",
)


def has_japanese_char(text: str) -> bool:
    """True if ``text`` contains at least one hiragana / katakana / kanji char."""
    return bool(_JAPANESE_CHAR_RE.search(text))


def is_non_japanese_input(text: str) -> bool:
    """True if the visible (non-punctuation) part of ``text`` is empty of
    Japanese letters and contains at least one alphabetic-ish letter.

    "Hello" → True. "ABC123" → True. "?!?" → False (only punctuation).
    "こんにちは" → False. "Hi こん" → False (one Japanese char wins).
    """
    if not text:
        return False
    if has_japanese_char(text):
        return False
    stripped = _NON_TEXT_RE.sub("", text)
    if not stripped:
        return False
    # Require at least one letter-like character so pure numbers don't count.
    return any(ch.isalpha() for ch in stripped)


# Sentence-final particles / copulas. If the input ends with one of these,
# it's almost certainly a clause, not a single term, so the unknown-word
# flow should not trigger.
_SENTENCE_TAIL_FORMS: tuple[str, ...] = (
    "です", "ます", "だよ", "だね", "だな", "かな",
    "だ", "ね", "よ", "な", "か", "の", "さ", "わ", "ぞ", "ぜ", "ー",
)

# Verbal endings that indicate the input is a clause / verb phrase rather
# than a noun term. Checked after stripping a single optional sentence-final
# particle so e.g. "遅くてさ" still reads as ending in "て".
_VERBAL_TAILS: tuple[str, ...] = (
    "て", "た", "ない", "たい", "てる", "てた",
    "ます", "ました", "ません", "ません",
)

# Particles that strongly suggest a multi-word sentence anywhere in the text.
_SENTENCE_PARTICLE_PATTERNS: tuple[str, ...] = (
    "は", "が", "を", "に", "へ", "で", "と", "から", "まで", "の",
)


def looks_like_single_word(text: str, *, max_chars: int = 12) -> bool:
    """True if ``text`` looks like a single short term we could ask about.

    No internal spaces, no Japanese sentence punctuation, no full stops,
    no obvious sentence-final particles, and short enough that asking
    "what is that?" makes sense.
    """
    cleaned = text.strip()
    if not cleaned or len(cleaned) > max_chars:
        return False
    forbidden = (" ", "\t", "\n", "。", "、", "!", "?", "！", "？", "…", ".", ",")
    if any(ch in cleaned for ch in forbidden):
        return False
    # If the term ends with a sentence-final form (です/ます/だ/ね/よ/...),
    # treat it as a sentence rather than a noun.
    for tail in _SENTENCE_TAIL_FORMS:
        if cleaned.endswith(tail) and len(cleaned) > len(tail):
            return False
    # Verb-phrase endings: even after a stray sentence particle ("遅くてさ"),
    # the remaining stem usually ends in te/ta/nai/tai. That's a clause too.
    stem = cleaned
    for tail in ("さ", "ね", "よ", "な"):
        if stem.endswith(tail) and len(stem) > len(tail) + 1:
            stem = stem[: -len(tail)]
            break
    for tail in _VERBAL_TAILS:
        if stem.endswith(tail) and len(stem) > len(tail) + 1:
            return False
    # If there's a structural particle inside, it's a clause: "今日は天気が…".
    interior = cleaned[1:-1] if len(cleaned) > 2 else ""
    if interior and any(p in interior for p in _SENTENCE_PARTICLE_PATTERNS):
        return False
    return True


def looks_like_typo_correction(text: str) -> bool:
    """True if ``text`` reads like the user is taking back a previous turn."""
    if not text:
        return False
    return any(pattern in text for pattern in TYPO_RETRACTION_PATTERNS)
