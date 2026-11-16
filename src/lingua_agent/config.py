"""Environment-driven settings. Secrets are read from the environment only."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

# AssemblyAI's docs show both /v1/ws (quickstarts, WebSocket reference) and
# /v1/voice (one reference page). /v1/ws is the one in the runnable examples;
# override with ASSEMBLYAI_WS_URL if the probe says otherwise.
DEFAULT_WS_URL = "wss://agents.assemblyai.com/v1/ws"
DEFAULT_REST_URL = "https://agents.assemblyai.com"
PROFANITY_POLICIES = ("label", "block")
_PLACEHOLDERS = {"your_assemblyai_api_key_here", "changeme", "xxx"}


class ConfigError(ValueError):
    """Raised for missing or invalid configuration."""


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)  # repr=False keeps the key out of logs/tracebacks
    ws_url: str = DEFAULT_WS_URL
    rest_url: str = DEFAULT_REST_URL
    agent_id: str | None = None
    voice_id: str = "alba"
    sample_rate: int = 24000
    max_connect_attempts: int = 5
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0
    profanity_policy: str = "block"
    min_silence_ms: int | None = None  # None = the default in agent_spec
    course_language: str | None = None  # set by --language; the voice is fixed per session

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        key = env.get("ASSEMBLYAI_API_KEY", "").strip()
        if not key or key.lower() in _PLACEHOLDERS:
            raise ConfigError("ASSEMBLYAI_API_KEY is not set (copy .env.example to .env)")
        policy = env.get("PROFANITY_POLICY", "block").strip().lower()
        if policy not in PROFANITY_POLICIES:
            raise ConfigError(f"PROFANITY_POLICY must be one of {PROFANITY_POLICIES}, got {policy!r}")
        ws_url = env.get("ASSEMBLYAI_WS_URL", DEFAULT_WS_URL).strip()
        rest_url = env.get("ASSEMBLYAI_REST_URL", DEFAULT_REST_URL).strip().rstrip("/")
        # The API key travels in a header: never over an unencrypted connection.
        if not ws_url.startswith("wss://"):
            raise ConfigError("ASSEMBLYAI_WS_URL must start with wss://")
        if not rest_url.startswith("https://"):
            raise ConfigError("ASSEMBLYAI_REST_URL must start with https://")
        raw_silence = env.get("TURN_MIN_SILENCE_MS", "").strip()
        min_silence: int | None = None
        if raw_silence:
            try:
                min_silence = int(raw_silence)
            except ValueError:
                raise ConfigError(f"TURN_MIN_SILENCE_MS must be a whole number of milliseconds, got {raw_silence!r}") from None
            if not 100 <= min_silence <= 3000:
                raise ConfigError("TURN_MIN_SILENCE_MS must be between 100 and 3000")
        return cls(
            api_key=key,
            ws_url=ws_url,
            rest_url=rest_url,
            agent_id=env.get("ASSEMBLYAI_AGENT_ID", "").strip() or None,
            voice_id=env.get("VOICE_ID", "alba").strip() or "alba",
            profanity_policy=policy,
            min_silence_ms=min_silence,
        )


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Existing environment variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as fh:  # -sig: Notepad on Windows saves a BOM
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))
