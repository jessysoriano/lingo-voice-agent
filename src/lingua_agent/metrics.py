"""Latency and interruption instrumentation, wall-clock, injectable clock.

The number recorded here is time from the server's `input.speech.stopped` event
to the first audio byte of the reply that answers it. It is useful for live
monitoring, but it starts the clock at the vendor's own end-of-speech event, so
it hides endpointing delay. The bake-off therefore measures from the true end of
speech in the audio file instead (see evals/providers.py).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0-100). Empty input returns NaN."""
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


class LatencyTracker:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._stopped_at: float | None = None
        self._armed = False
        self._agent_speaking = False
        self._turn_stopped_at: float | None = None
        self._tool_sent_at: float | None = None
        self.ttfa_ms: list[float] = []
        # A tool turn has two replies: the first may be only a filler ("One moment"),
        # and the real answer starts after the tool result is sent. ttfa_ms is the
        # first audio of any kind; these two are the lesson itself.
        self.answer_ms: list[float] = []     # end of speech -> first audio after the tool result
        self.post_tool_ms: list[float] = []  # tool result sent -> first audio after it
        self.switch_ms: list[float] = []     # language switch started -> first audio from the new voice
        self._switch_at: float | None = None
        self.interruptions = 0

    def speech_started(self) -> None:
        # The learner is talking (again): any earlier end-of-speech is void.
        self._stopped_at = None
        self._armed = False
        self._tool_sent_at = None  # a new utterance voids the old tool turn

    def speech_stopped(self) -> None:
        self._stopped_at = self._turn_stopped_at = self._clock()
        # If the agent is mid-reply (back-channel, or a barge-in not yet confirmed),
        # audio still arriving belongs to the OLD reply. Wait for a new reply.started.
        self._armed = not self._agent_speaking

    def reply_started(self) -> None:
        self._agent_speaking = True
        if self._stopped_at is not None:
            self._armed = True

    def tool_results_sent(self) -> None:
        self._tool_sent_at = self._clock()

    def switch_started(self) -> None:
        self._switch_at = self._clock()

    def audio_chunk(self) -> None:
        if self._switch_at is not None:
            self.switch_ms.append((self._clock() - self._switch_at) * 1000)
            self._switch_at = None
        if self._tool_sent_at is not None:
            now = self._clock()
            self.post_tool_ms.append((now - self._tool_sent_at) * 1000)
            if self._turn_stopped_at is not None:
                self.answer_ms.append((now - self._turn_stopped_at) * 1000)
            self._tool_sent_at = None
        if self._armed and self._stopped_at is not None:
            self.ttfa_ms.append((self._clock() - self._stopped_at) * 1000)
            self._armed = False
            self._stopped_at = None

    def reply_done(self, interrupted: bool = False) -> None:
        self._agent_speaking = False
        if interrupted:
            self.interruptions += 1

    def summary(self) -> dict[str, float | int | None]:
        """Unmeasured values are None, never NaN: NaN is not valid JSON and the report is a JSON file."""
        n = len(self.ttfa_ms)
        return {
            "turns_measured": n,
            "ttfa_p50_ms": _or_none(percentile(self.ttfa_ms, 50)),
            "ttfa_p95_ms": _or_none(percentile(self.ttfa_ms, 95)),
            "ttfa_mean_ms": sum(self.ttfa_ms) / n if n else None,
            "interruptions": self.interruptions,
            "tool_turns": len(self.post_tool_ms),
            "answer_p50_ms": _or_none(percentile(self.answer_ms, 50)),
            "post_tool_p50_ms": _or_none(percentile(self.post_tool_ms, 50)),
            "language_switches": len(self.switch_ms),
            "switch_p50_ms": _or_none(percentile(self.switch_ms, 50)),
        }


def _or_none(value: float) -> float | None:
    return None if math.isnan(value) else value
