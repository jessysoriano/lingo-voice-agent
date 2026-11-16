"""Provider adapters so the same audio and scoring run against any vendor.

The customer's incumbent is a stub on purpose: implementing it takes their API
keys and about an hour, and it is the first thing to do in the real evaluation.
Same audio, same pacing, same scorer, or the comparison is not worth showing.

Latency is measured the same way for every vendor: from the moment the last
speech chunk of the clip was sent to the first reply-audio byte received. It is
NOT measured from a vendor's own "speech stopped" event, because that would hide
endpointing delay and reward whichever vendor reports end-of-speech latest.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lingua_agent import agent_spec, coaching, protocol, tutor
from lingua_agent.config import Settings
from lingua_agent.pcm import paced_pcm
from lingua_agent.session import Connect, VoiceSession

OFFLINE_TAIL_MS = 2000


@dataclass(frozen=True)
class TurnResult:
    transcript: str
    agent_text: str
    ttfa_ms: float | None             # true end of speech -> first reply audio (comparable across vendors)
    endpoint_ttfa_ms: float | None    # vendor's speech-stopped event -> first reply audio (diagnostic)
    tool_calls: tuple[dict, ...] = ()


class Provider(Protocol):
    name: str

    async def run_turn(self, pcm_path: Path, speech_end_ms: int | None = None) -> TurnResult: ...


class IncumbentProvider:
    name = "incumbent"

    async def run_turn(self, pcm_path: Path, speech_end_ms: int | None = None) -> TurnResult:
        raise NotImplementedError(
            "Implement against the customer's current vendor using their keys: stream the same PCM "
            "at real-time pace, capture the final user transcript, the agent text, and the time from "
            "the last speech chunk sent to the first reply audio byte."
        )


class AssemblyAIProvider:
    name = "assemblyai"

    def __init__(self, settings: Settings, *, connect: Connect | None = None, realtime: bool = True,
                 prompt_version: str = agent_spec.DEFAULT_PROMPT_VERSION, turn_timeout_s: float = 30.0) -> None:
        self._settings, self._connect, self._realtime = settings, connect, realtime
        self._prompt_version, self._timeout = prompt_version, turn_timeout_s

    async def run_turn(self, pcm_path: Path, speech_end_ms: int | None = None) -> TurnResult:
        marks: dict[str, float] = {}
        state = {"tool_in_reply": False, "agent_spoke": False}

        def on_event(ev: protocol.Event) -> None:
            if ev.type == "reply.audio" and "speech_end" in marks:
                marks.setdefault("first_audio", time.monotonic())
            elif ev.type == "tool.call":
                state["tool_in_reply"] = True
            elif ev.type == "transcript.agent":
                state["agent_spoke"] = True
            elif ev.type == "reply.done":
                # A reply that ends with a tool call is followed by the real answer:
                # keep listening. Stop on the first spoken reply with no tool pending.
                if state["tool_in_reply"]:
                    state["tool_in_reply"] = False
                elif state["agent_spoke"]:
                    session.stop()

        # No greeting: it would overlap the test utterance and look like a barge-in.
        # No voice switching: one clip is one session, scored on one fixed configuration.
        update = protocol.build_session_update_inline(
            agent_spec.build_session_config(self._settings, self._prompt_version, greeting=None, can_switch=False))
        # Live runs keep sending silence until the reply arrives, as a microphone would.
        tail_ms = int(self._timeout * 1000) if self._realtime else OFFLINE_TAIL_MS
        session = VoiceSession(
            self._settings, session_update=update,
            tools={
                tutor.TOOL_NAME: lambda a: tutor.teach_phrase(
                    a, profanity_policy=self._settings.profanity_policy,
                    course_language=self._settings.course_language),
                tutor.CHECK_TOOL_NAME: coaching.check_attempt,
            },
            audio_source=paced_pcm(pcm_path, self._settings.sample_rate, realtime=self._realtime,
                                   tail_ms=tail_ms, speech_end_ms=speech_end_ms, marks=marks),
            audio_sink=lambda b: None, connect=self._connect, on_event=on_event,
        )
        async with asyncio.timeout(self._timeout):  # not wait_for: see session._send_audio
            report = await session.run()
        if report.reconnects:
            raise RuntimeError("connection dropped mid-clip; result discarded")
        ttfa = (marks["first_audio"] - marks["speech_end"]) * 1000 if "first_audio" in marks else None
        endpoint_ttfa = session.tracker.ttfa_ms[0] if session.tracker.ttfa_ms else None
        return TurnResult(
            transcript=" ".join(t.text for t in report.turns if t.role == "user"),
            agent_text=" ".join(t.text for t in report.turns if t.role == "agent"),
            ttfa_ms=ttfa, endpoint_ttfa_ms=endpoint_ttfa, tool_calls=tuple(report.tool_calls),
        )
