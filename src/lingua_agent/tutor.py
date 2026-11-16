"""The `teach_phrase` tool: the model proposes a lesson, this code validates it.

Teaching through a tool call gives us a pre-speech checkpoint that a pure
speech-to-speech loop otherwise lacks. Result codes are stable strings so the
offline eval can exact-match them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .guardrails import contains_profanity
from .languages import normalize_text, resolve, script_matches, spoken_names

MAX_PHRASE_CHARS = 200
_RESPELLING = re.compile(r"^[A-Za-z' -]+$")
_EDGE_PUNCT = re.compile(r"^[\s.,;:!?¿¡…\"“”()\[\]/]+|[\s.,;:!?¿¡…\"“”()\[\]/]+$")
_REGISTERS = {"neutral", "informal", "formal", "vulgar"}
_STRETCHED = re.compile(r"(.)\1\1")  # "gaaah": the voice would drag it out
_Y_CONSONANT = re.compile(r"(?<![aeiou])y(?=[aeiou])")  # the y in "yeh" and "nyoh"
_CONSONANT_Y = re.compile(r"(?<=[bcdfghjkmpqrstvwxz])y(?=[aeiou])")  # "syahs" is read as "sigh-az"
_VOWEL_RUN = re.compile(r"[aeiouy]+(?:h|w(?![aeiouy]))?")  # "ah", "eh", "ew", "ow" count as one vowel
# The only vowel spellings the respelling key allows. A lone "e" in "YET" or "es" is read as
# an English word or letter, which is how a live session got "gah-YET-ahs" for "galletas".
ALLOWED_VOWELS = frozenset({"ah", "eh", "ee", "oh", "oo", "ay", "eye", "ow", "oy", "uh", "ew"})


def unclear_vowels(respelling: str) -> list[str]:
    """Vowel groups outside the key, e.g. ["e"] for "gah-YET-ahs"."""
    text = _Y_CONSONANT.sub("Y", respelling.lower())
    return sorted({run for run in _VOWEL_RUN.findall(text) if run not in ALLOWED_VOWELS})

# How the model should respell. A learner heard respellings that did not match the real
# sounds, so the conventions are spelled out per language instead of left to the model.
RESPELLING_GUIDE = (
    "Vowels, always: ah (father), eh (bed), ee (see), oh (go), oo (food), ay (day), eye (my), ow (now), "
    "oy (boy), uh (the). Never write a lone a, e, i, o or u, and never repeat a letter three times. "
    "One chunk per syllable of the TARGET language, one stressed syllable per word in CAPITALS. "
    "SPANISH: h is silent, j and ge/gi = h, ll and y = y, only the letter ñ = ny (a plain n stays n: niño = NEE-nyoh, not NYEE-nyoh), qu = k, gue/gui = g, v = b, z and ce/ci = s "
    "(Latin American), rr = rolled r. Stress falls on the accented vowel; otherwise on the second-to-last syllable "
    "when the word ends in a vowel, n or s, and on the last syllable for other endings. "
    "FRENCH: u = ew, eu/oeu = uh, ou = oo, oi = wah, j and ge/gi = zh, ch = sh, r = a soft throat r (write r), "
    "nasal vowels = ahn, ehn, ohn (am, an, en = ahn; in, ain = ehn; on = ohn), h is silent, and final consonants "
    "are usually silent. Stress the last syllable of each word group. "
    "GERMAN: w = v, v = f, z = ts, s before a vowel = z, ch = kh, sch = sh, st/sp at the start = sht/shp, ü = ew, "
    "ö = uh, ei = eye, ie = ee, eu/äu = oy, final -ig = eekh. "
    "ITALIAN: c before e/i = ch, ch = k, g before e/i = j, gn = ny, gli = ly, sc before e/i = sh, z = ts, "
    "double consonants stay doubled (BAHT-toh). "
    "PORTUGUESE (Brazil): lh = ly, nh = ny, ão = own, õe = oyns, initial r and rr = h, final s = s, "
    "de/di and te/ti = jee/chee. "
    "Never write a y after a consonant except ny and ly: gracias = GRAH-see-ahs, not GRAH-syahs. "
    "Examples: dónde está = DOHN-deh ehs-TAH; galletas = gah-YEH-tahs; niña = NEE-nyah."
)

TOOL_NAME = "teach_phrase"
CHECK_TOOL_NAME = "check_attempt"


def tool_definition(*, inline: bool = False) -> dict[str, Any]:
    """Client-side tool definition (no `http` block, so our code runs it).

    The saved-agent REST schema documents timeout_seconds / execution_mode. For
    inline session tools the docs show only the flat function shape, so the
    inline form sends just that.
    """
    definition = _definition()
    if inline:
        return {"type": "function", **{k: definition[k] for k in ("name", "description", "parameters")}}
    return definition


def _definition() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": (
            "Call this every time the learner asks how to say something in another language, "
            "BEFORE speaking the phrase. Pass the English text, the phrase in the target language "
            "(native script), and a hyphenated pronunciation respelling with stressed syllables in "
            "CAPITALS. Speak only what this tool approves."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_language": {"type": "string", "enum": spoken_names()},
                "source_text": {"type": "string", "description": "What the learner wants to say, in English."},
                "phrase": {"type": "string", "description": "The phrase in the target language, native script."},
                "pronunciation": {
                    "type": "string",
                    "description": (
                        "Respelling for English speakers: letters and hyphens only, no punctuation. "
                        + RESPELLING_GUIDE
                    ),
                },
                "register": {"type": "string", "enum": sorted(_REGISTERS),
                             "description": "Use vulgar for swear words, insults and crude slang, however the learner words the request."},
            },
            "required": ["target_language", "source_text", "phrase", "pronunciation"],
        },
        "timeout_seconds": 10,
        "execution_mode": "interactive",
    }


def clean_respelling(raw: str) -> str:
    """Fix harmless formatting slips instead of rejecting them.

    A trailing "?" cost a real session an extra model round trip (reject, retry),
    which is user-visible latency. Edge punctuation, stray dashes and runs of
    spaces carry no meaning, so they are normalised; digits, IPA and other
    symbols still fail validation because the model has misunderstood the format.
    """
    text = _EDGE_PUNCT.sub("", raw.replace("–", "-").replace("—", "-").replace("·", "-"))
    text = re.sub(r"\s*-\s*", "-", text)       # "DOHN - deh" -> "DOHN-deh"
    text = re.sub(r"-{2,}", "-", text)
    return re.sub(r"\s+", " ", text).strip("- ")


def spoken_breakdown(respelling: str) -> dict[str, str]:
    """The pronunciation lines the agent reads out, built once here so every
    lesson has the same shape. The model used to improvise this part, and
    produced "gah... YET... ahs" in one lesson and "DOHN-deh, ehs-TAH, el, BAHN-yo"
    in the next.

    sounds_like: every syllable as its own short beat ("gah, YEH, tahs"). A live session showed
                 the voice reads a hyphenated chunk like "gah-YEH-tahs" as one English word
                 ("guy-eech tas"), while comma-separated syllables came out right.
    slow:        the same with longer pauses, only for "again" or "slower".
    """
    # Lowercase for the voice: it spells an all-caps chunk out letter by letter ("Y-E-H").
    # The capitals (stress) stay in "pronunciation", which the app prints on screen.
    syllables = [syl for syl in respelling.lower().replace(" ", "-").split("-") if syl]
    return {"sounds_like": ", ".join(syllables), "slow": "... ".join(syllables)}


def _check_attempt_definition() -> dict[str, Any]:
    return {
        "name": CHECK_TOOL_NAME,
        "description": (
            "Call this right after the learner tries to say a phrase you just taught, even if what the "
            "transcript shows looks wrong, like nonsense or another language. Pass the target phrase and "
            "exactly what you heard. Use the result to give feedback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_language": {"type": "string", "enum": spoken_names()},
                "target_phrase": {"type": "string", "description": "The phrase you taught, in the target language."},
                "heard": {"type": "string", "description": "The learner's transcript, exactly as you received it."},
            },
            "required": ["target_phrase", "heard"],
        },
        "timeout_seconds": 10,
        "execution_mode": "interactive",
    }


def tool_definitions(*, inline: bool = False) -> list[dict[str, Any]]:
    check = _check_attempt_definition()
    check = {"type": "function", **{k: check[k] for k in ("name", "description", "parameters")}} if inline else check
    return [tool_definition(inline=inline), check]


def _reject(reason: str, speak_instead: str) -> dict[str, Any]:
    return {"status": "rejected", "reason": reason, "speak_instead": speak_instead}


def teach_phrase(args: Mapping[str, Any], *, profanity_policy: str = "label",
                 course_language: str | None = None, auto_switch: bool = False) -> dict[str, Any]:
    """course_language is the language of the voice speaking right now (None = the English voice).
    With auto_switch, a request for another language returns "switch_language" so the client can
    start a new session with that language's voice; without it, the request is rejected."""
    missing = [k for k in ("target_language", "source_text", "phrase", "pronunciation") if not str(args.get(k, "")).strip()]
    if missing:
        return _reject("missing_fields", f"Ask the learner to repeat what they want to say. Missing: {', '.join(missing)}.")

    lang = resolve(args["target_language"])
    if lang is None:
        return _reject("unsupported_language", f"Say you can't teach that language yet, and offer {', '.join(spoken_names())}.")
    if lang.voice is None:
        # Checked before the lesson itself: a formatting slip must not cost a retry
        # for a language that cannot be spoken anyway.
        return _reject("language_not_spoken_yet",
                       f"Say warmly that you can't speak {lang.name} yet, and offer {', '.join(spoken_names())}.")

    source, phrase = (str(args[k]).strip() for k in ("source_text", "phrase"))
    respelling = clean_respelling(str(args["pronunciation"]))
    if len(phrase) > MAX_PHRASE_CHARS or len(source) > MAX_PHRASE_CHARS:
        return _reject("too_long", "Ask for a shorter phrase, one sentence at a time.")

    if normalize_text(phrase) == normalize_text(source) and lang.script == "latin":
        return _reject("untranslated", "Say you aren't sure of the translation and ask the learner to rephrase.")
    if not script_matches(lang, phrase):
        return _reject("wrong_script", f"The phrase was not valid {lang.name}. Apologize briefly and try again with correct {lang.name}.")
    if not respelling or not _RESPELLING.match(respelling) or _STRETCHED.search(respelling):
        return _reject("bad_pronunciation_format",
                       "Give a respelling using only letters and hyphens, one chunk per syllable, with no letter "
                       "repeated three times, using ah, eh, ee, oh, oo for the vowels.")
    if _CONSONANT_Y.search(respelling.lower()):
        return _reject("unclear_vowels",
                       "A y after a consonant gets read like the English word (syahs sounds like 'sigh-az'). Write "
                       "ee-ah / ee-oh instead, for example GRAH-see-ahs. Only ny and ly may keep the y.")
    odd = unclear_vowels(respelling)
    if odd:
        return _reject("unclear_vowels",
                       f"The respelling has vowel spellings that read as English words or letters ({', '.join(odd)}). "
                       "Try again using only ah, eh, ee, oh, oo, ay, eye, ow, oy, uh, ew for vowels, for example "
                       "gah-YEH-tahs, not gah-YET-ahs.")

    if auto_switch and lang.name != course_language:
        return {"status": "switch_language", "language_code": lang.code, "language": lang.name}
    if course_language and lang.name != course_language:
        return _reject("other_course_language",
                       f"Say this session is for {course_language}, and they can start another session for {lang.name}.")

    register = str(args.get("register", "neutral")).lower()
    register = register if register in _REGISTERS else "neutral"
    vulgar = register == "vulgar" or contains_profanity(source)
    if vulgar and profanity_policy == "block":
        return _reject("profanity_blocked", "Say you can't teach that one, and offer a polite alternative.")

    result: dict[str, Any] = {
        "status": "approved",
        "target_language": lang.name,
        "language_code": lang.code,
        "phrase": phrase,
        "pronunciation": respelling,
        "syllables": [s for s in respelling.replace(" ", "-").split("-") if s],
        "register": "vulgar" if vulgar else register,
        **spoken_breakdown(respelling),
        "how_to_say": (
            "Say the phrase naturally, then say sounds_like once, then invite the learner to try. "
            "Only if they ask for slower or again, say slow."
        ),
    }
    if vulgar:
        result["note"] = "Mention briefly that this is vulgar and not for polite company."
    return result
