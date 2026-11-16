"""Builds the two config shapes the API takes, with limits enforced so a bad
config fails in CI, not on the first live call.

The shapes differ, per AssemblyAI's docs:
  * inline `session.update`: the voice is `output.voice` (a string), tools use
    the flat function shape, and the recognizer accepts `transcription_prompt`
  * saved agent (POST /v1/agents): the voice is `voice.voice_id`, tools take
    `timeout_seconds` / `execution_mode`
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .prompts import (
    DEFAULT_PROMPT_VERSION,
    GREETING,
    KEYTERMS,
    SHADOW_GREETING,
    TRANSCRIPTION_PROMPT,
    get_prompt,
    shadow_prompt,
)
from .tutor import tool_definitions

MAX_KEYTERMS = 100
MAX_KEYTERM_CHARS = 50
MAX_TRANSCRIPTION_PROMPT_CHARS = 1750

# Learners pause mid-request ("how do I say... um... the library"), but every
# millisecond of min_silence is added to every reply. Two live sessions at 1000 ms
# measured about 2.4 s to first audio (p50), so this starts at 700 ms. It is a
# guess, not a tuned value: set TURN_MIN_SILENCE_MS to move it, and watch the
# session report for learners being cut off. The bake-off sweeps it (EVAL_PLAN.md).
DEFAULT_TURN_DETECTION: dict[str, Any] = {
    "vad_threshold": 0.5,
    "min_silence": 700,
    "max_silence": 2500,
    "interrupt_response": True,
}


def turn_detection(settings: Settings) -> dict[str, Any]:
    cfg = dict(DEFAULT_TURN_DETECTION)
    if settings.min_silence_ms is not None:
        cfg["min_silence"] = settings.min_silence_ms
        cfg["max_silence"] = max(cfg["max_silence"], settings.min_silence_ms)
    return cfg


def validate_keyterms(terms: list[str]) -> list[str]:
    if len(terms) > MAX_KEYTERMS:
        raise ValueError(f"at most {MAX_KEYTERMS} keyterms allowed, got {len(terms)}")
    too_long = [t for t in terms if len(t) > MAX_KEYTERM_CHARS]
    if too_long:
        raise ValueError(f"keyterms over {MAX_KEYTERM_CHARS} chars: {too_long}")
    return terms


def _audio_format() -> dict[str, str]:
    return {"encoding": "audio/pcm"}  # 24 kHz PCM16; the rate is implied by the encoding


def _input(*, keyterms: list[str], inline: bool, settings: Settings) -> dict[str, Any]:
    block: dict[str, Any] = {"format": _audio_format(), "turn_detection": turn_detection(settings)}
    if keyterms:
        block["keyterms"] = validate_keyterms(list(keyterms))
    if inline:
        if len(TRANSCRIPTION_PROMPT) > MAX_TRANSCRIPTION_PROMPT_CHARS:
            raise ValueError("TRANSCRIPTION_PROMPT exceeds the API limit")
        block["transcription_prompt"] = TRANSCRIPTION_PROMPT
    return block


def build_session_config(
    settings: Settings,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    *,
    greeting: str | None = GREETING,
    can_switch: bool = True,
) -> dict[str, Any]:
    """Inline session config. Pass greeting=None to have the agent listen first
    (the eval does, so the greeting never overlaps the test utterance), and
    can_switch=False when the client will not switch voices (evals, the probe)."""
    config: dict[str, Any] = {
        "system_prompt": get_prompt(prompt_version, course_language=settings.course_language, can_switch=can_switch),
        "input": _input(keyterms=KEYTERMS, inline=True, settings=settings),
        "output": {"voice": settings.voice_id, "format": _audio_format()},
        "tools": tool_definitions(inline=True),
    }
    if greeting:
        config["greeting"] = greeting
    return config


def build_shadow_config(settings: Settings, phrase: str) -> dict[str, Any]:
    """Shadow mode: no tools, and deliberately no keyterms (boosting the phrase
    being assessed would bias the recognizer toward hearing it)."""
    phrase = " ".join(phrase.split())
    if not phrase:
        raise ValueError("shadow mode needs a phrase to practise")
    return {
        "system_prompt": shadow_prompt(phrase),
        "greeting": SHADOW_GREETING,
        "input": _input(keyterms=[], inline=True, settings=settings),
        "output": {"voice": settings.voice_id, "format": _audio_format()},
    }


def build_agent_request(settings: Settings, prompt_version: str = DEFAULT_PROMPT_VERSION) -> dict[str, Any]:
    """Body for POST /v1/agents and PUT /v1/agents/{id}."""
    return {
        "name": f"Lingo pronunciation coach ({settings.course_language or 'all languages'}, {prompt_version})",
        "system_prompt": get_prompt(prompt_version, course_language=settings.course_language, can_switch=False),
        "greeting": GREETING,
        "voice": {"voice_id": settings.voice_id},
        "input": {"type": "audio", **_input(keyterms=KEYTERMS, inline=False, settings=settings)},
        "output": {"type": "audio", "format": _audio_format()},
        "tools": tool_definitions(),
    }
