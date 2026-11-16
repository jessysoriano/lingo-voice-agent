"""Deterministic guardrails for the three places text crosses a trust boundary.

1. Learner transcript  (input):  PII + prompt-injection heuristics. Used to
   redact before logging and to flag, not to block (audio is already in flight).
2. Tool arguments      (pre-speech): the model proposes a lesson, we dispose.
   This is the only gate that runs BEFORE the phrase is spoken, which is why the
   agent is built to teach through a tool call.
3. Agent transcript    (output, post-hoc): voice-format, leakage and over-claiming checks.
   With a speech-to-speech API the audio is already out when the text arrives,
   so this is monitoring, not blocking. See README "Where guardrails can and
   cannot run".

All checks are heuristics with stable string codes so evals can exact-match them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROMPT_CANARY = "LINGUA-CANARY-7f3a"
MAX_AGENT_WORDS = 80

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SPOKEN_EMAIL = re.compile(r"\b[\w.]+\s+at\s+[\w.]+\s+dot\s+(?:com|net|org|edu|io)\b", re.I)
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

_INJECTION = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|prompts?|rules)"
    r"|(?:reveal|show|repeat|print) (?:me )?(?:your|the) (?:system )?(?:prompt|instructions)"
    r"|you are now\b|developer mode|\bjailbreak\b|disregard your (?:rules|instructions)",
    re.I,
)

_MARKDOWN = re.compile(r"[*#`>_~]|^\s*(?:[-•]|\d+\.)\s", re.M)
_URL = re.compile(r"https?://|www\.", re.I)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
# The agent gets a transcript of the learner, never the sound, so it cannot know the
# pronunciation or accent was good. A live session caught it saying "Your pronunciation
# was perfect"; this flags that claim so it shows up in the report and the eval.
_PRAISE = r"(?:perfect|flawless|spot[- ]on|excellent|impeccable|native[- ]like|just right)"
_OVERCLAIM = re.compile(
    rf"\b(?:pronunciation|accent)\b[^.!?]{{0,40}}\b{_PRAISE}\b|\b{_PRAISE}\b[^.!?]{{0,20}}\b(?:pronunciation|accent)\b"
    rf"|\b(?:pronounced|said) (?:it|that) (?:perfectly|flawlessly)\b",
    re.I,
)

# Placeholder lexicon: mild English terms only. Production should use a
# maintained, multilingual list; the policy knob (label|block) is what matters.
_PROFANITY = frozenset({"damn", "crap", "shit", "fuck", "fucking", "bitch", "asshole"})


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    flags: tuple[str, ...] = ()
    redacted: str | None = None


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if alt:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


def contains_profanity(text: str) -> bool:
    return any(tok in _PROFANITY for tok in re.findall(r"[a-z']+", text.casefold()))


def check_transcript(text: str) -> GuardResult:
    """Flag PII and injection attempts in a learner transcript; return a redacted copy."""
    flags: list[str] = []
    redacted = text

    def _sub(pattern: re.Pattern[str], token: str, flag: str, guard=None) -> None:
        nonlocal redacted
        def repl(m: re.Match[str]) -> str:
            if guard and not guard(m.group(0)):
                return m.group(0)
            if flag not in flags:
                flags.append(flag)
            return token
        redacted = pattern.sub(repl, redacted)

    _sub(_EMAIL, "[EMAIL]", "pii_email")
    _sub(_SPOKEN_EMAIL, "[EMAIL]", "pii_email")
    _sub(_SSN, "[SSN]", "pii_ssn")
    _sub(_CARD, "[CARD]", "pii_card", guard=lambda s: _luhn_ok(re.sub(r"\D", "", s)))
    _sub(_PHONE, "[PHONE]", "pii_phone")
    if _INJECTION.search(text):
        flags.append("prompt_injection")
    # PII is redacted and flagged but the turn is still allowed: the audio has
    # already been processed upstream, and the agent's prompt tells it to decline.
    return GuardResult(allowed="prompt_injection" not in flags, flags=tuple(flags), redacted=redacted)


def check_agent_text(text: str) -> GuardResult:
    """Voice-format and leakage checks on what the agent just said."""
    flags: list[str] = []
    if _MARKDOWN.search(text):
        flags.append("markdown")
    if _URL.search(text):
        flags.append("url")
    if _EMOJI.search(text):
        flags.append("emoji")
    if len(text.split()) > MAX_AGENT_WORDS:
        flags.append("too_long")
    if PROMPT_CANARY.casefold() in text.casefold():
        flags.append("prompt_leak")
    if _OVERCLAIM.search(text):
        flags.append("overclaims_pronunciation")
    return GuardResult(allowed=not flags, flags=tuple(flags))
