"""Attempt checking and one-line tips for English speakers.

What this can and cannot do, stated up front: the Voice Agent API gives us a
transcript of what the learner said, not the sound. Comparing that transcript
with the target phrase tells us whether a listener (here, a recognizer) would
have understood each word, so it measures *intelligibility*. It cannot hear an
accent, vowel quality or stress. The tips below are generic hints for common
English-speaker mistakes, shown only for a word that was missed; they are not an
assessment of how the learner actually sounded. True phoneme-level feedback needs
a pronunciation-assessment service fed the learner's audio (see README).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .languages import normalize_text, resolve
from .shadow import align, tokenize

# (regex on the NORMALIZED word, tip). First match wins. Accents are folded away
# by normalize_text, so patterns use plain letters.
_TIPS: dict[str, list[tuple[str, str]]] = {
    "es": [
        (r"ll", "the double l sounds like the y in yes"),
        (r"rr|^r", "roll the r with a quick tap of the tongue"),
        (r"^h", "the h at the start is silent"),
        (r"j|g[ei]", "this j sound is a soft breathy h, like in hot"),
        (r"ñ", "the n with a tilde sounds like ny in canyon"),
    ],
    "fr": [
        (r"r", "the French r comes from the back of the throat, like a soft gargle"),
        (r"(?<![ao])u", "for u, say ee with your lips rounded"),
        (r"(on|an|en|in|un)$", "this is a nasal vowel, so don't pronounce the final n"),
        (r"[sdtxz]$", "the last letter is usually silent"),
    ],
    "de": [
        (r"sch", "sch sounds like sh"),
        (r"ch", "ch is a soft hiss from the throat, not a k"),
        (r"ü", "for ü, say ee with your lips rounded"),
        (r"[äö]", "the two dots change the vowel, so keep your lips rounded and the sound steady"),
        (r"w", "w sounds like an English v"),
        (r"^v", "v at the start sounds like f"),
        (r"z", "z sounds like ts"),
    ],
    "it": [
        (r"gli", "gli sounds like lli in million"),
        (r"gn", "gn sounds like ny in canyon"),
        (r"c[ei]", "c before e or i sounds like ch"),
        (r"([bcdfglmnprstvz])\1", "hold the double consonant a beat longer"),
    ],
    "pt": [
        (r"nh", "nh sounds like ny in canyon"),
        (r"lh", "lh sounds like lli in million"),
        (r"ã|õ", "this is a nasal vowel, so let the sound come through your nose"),
        (r"^r|rr", "this r is a rough h-like sound from the throat"),
    ],
}


def tip_for(language_code: str, word: str) -> str | None:
    for pattern, tip in _TIPS.get(language_code, []):
        if re.search(pattern, word):
            return tip
    return None


def _original_word(target: str, normalized: str) -> str:
    for word in target.split():
        if normalize_text(word) == normalized:
            return word.strip("¿?¡!.,;:")
    return normalized


def check_attempt(args: Mapping[str, Any]) -> dict[str, Any]:
    target = str(args.get("target_phrase", "")).strip()
    heard = str(args.get("heard", "")).strip()
    if not target:
        return {"status": "rejected", "reason": "missing_fields",
                "speak_instead": "Ask which phrase they are practising."}
    if not tokenize(heard):
        return {"status": "ok", "verdict": "no_attempt", "score": 0.0, "words": [],
                "how_to_say": "Nothing was picked up. Invite them to try again, a bit louder or closer to the mic."}

    alignment = align(target, heard)
    words = [{"word": w.word, "state": w.state} for w in alignment.words]
    missed = next((w for w in alignment.words if w.state != "match"), None)
    if alignment.score == 1.0:
        verdict = "clear"
    elif alignment.score >= 0.5:
        verdict = "close"
    else:
        verdict = "unclear"

    result: dict[str, Any] = {
        "status": "ok", "verdict": verdict, "score": round(alignment.score, 2),
        "heard": heard, "words": words,
    }
    if missed:
        lang = resolve(args.get("target_language", ""))
        result["focus_word"] = _original_word(target, missed.word)
        # Match tips on the original word: normalisation folds ñ, ü and ã to plain letters.
        tip = tip_for(lang.code, result["focus_word"].casefold()) if lang else None
        if tip:
            result["tip"] = tip
    result["how_to_say"] = (
        "You know what the recognizer heard, not how it sounded. "
        + ("Say it came through clearly and offer the next phrase or one more go. "
           if verdict == "clear" else
           "Say briefly what you heard, name focus_word, give the tip if there is one, say that word slowly, "
           "and invite another try. ")
        + "Never say the pronunciation or accent was perfect, great or accurate."
    )
    return result
