"""Supported target languages and cheap, deterministic script checks.

The script check catches one specific, common failure: the model returning a
phrase in the wrong writing system (e.g. romaji where Japanese script was
needed, or English echoed back as the "translation"). It cannot tell Spanish
from Portuguese; that gap is a documented known_gap case in the offline eval, and
the bake-off's teaching accuracy (match against accepted translations) covers it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    script: str  # latin | kana_han | han | hangul
    # Voice Agent API output voice that speaks this language natively, or None if the
    # API cannot speak it yet. Per AssemblyAI's supported-languages page, output is
    # English, Spanish, French, German, Italian and Portuguese only; Japanese and Chinese
    # are "coming soon" and Korean is not on the input list either. The voice is fixed
    # for the life of a session, so the client starts a new session to change language.
    voice: str | None = None


LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in (
        Language("es", "Spanish", "latin", "lola"),
        Language("fr", "French", "latin", "estelle"),
        Language("de", "German", "latin", "juergen"),
        Language("it", "Italian", "latin", "giovanni"),
        Language("pt", "Portuguese", "latin", "rafael"),
        Language("ja", "Japanese", "kana_han"),
        Language("ko", "Korean", "hangul"),
        Language("zh", "Mandarin Chinese", "han"),
    )
}

_ALIASES = {"mandarin": "zh", "chinese": "zh"}


def supported_names() -> list[str]:
    return [lang.name for lang in LANGUAGES.values()]


def spoken_languages() -> list[Language]:
    """Languages the agent can teach by voice today."""
    return [lang for lang in LANGUAGES.values() if lang.voice]


def spoken_names() -> list[str]:
    return [lang.name for lang in spoken_languages()]


# English voices from the docs; they read foreign text with English rules.
ENGLISH_VOICES = frozenset({"alba", "eve", "george", "jane", "jean", "mary", "michael",
                            "anna", "charles", "paul", "vera"})


def resolve(value: object) -> Language | None:
    """Resolve a language name or ISO code (case-insensitive) to a Language."""
    if not isinstance(value, str):
        return None
    key = value.strip().casefold()
    key = _ALIASES.get(key, key)
    if key in LANGUAGES:
        return LANGUAGES[key]
    for lang in LANGUAGES.values():
        if lang.name.casefold() == key:
            return lang
    return None


def _char_ok(script: str, ch: str) -> bool:
    name = unicodedata.name(ch, "")
    if script == "latin":
        return "LATIN" in name
    if script == "hangul":
        return "HANGUL" in name
    if script == "han":
        return "CJK UNIFIED" in name
    if script == "kana_han":
        return any(tag in name for tag in ("HIRAGANA", "KATAKANA", "CJK UNIFIED"))
    return False


def script_matches(lang: Language, text: str) -> bool:
    """True if every letter in `text` belongs to `lang`'s expected script."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return all(_char_ok(lang.script, ch) for ch in letters)


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _fold_latin_accent(ch: str) -> str:
    base = unicodedata.normalize("NFD", ch)[0]
    return base if "LATIN" in unicodedata.name(base, "") else ch


def normalize_text(text: str) -> str:
    """Casefold, fold Latin accents, drop punctuation, collapse whitespace.

    Accents are folded for Latin letters only, so a recognizer that writes
    "donde esta" still matches "dónde está". Other scripts are left intact:
    stripping combining marks there would merge distinct kana (が vs か) and
    split Hangul syllables. The cost is that accent-only errors in Latin-script
    languages (año vs ano) are not counted; scoring that needs them should
    compare the raw strings.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    folded = "".join(_fold_latin_accent(ch) for ch in text)
    return " ".join(_PUNCT.sub(" ", folded).split())
