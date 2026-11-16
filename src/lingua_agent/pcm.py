"""Stream a recorded PCM clip the way a microphone would (used by evals and the probe)."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator
from pathlib import Path

CHUNK_MS = 40


async def paced_pcm(path: Path, sample_rate: int, *, realtime: bool, tail_ms: int,
                    speech_end_ms: int | None = None, marks: dict[str, float] | None = None,
                    ) -> AsyncIterator[bytes]:
    """Yield the clip in 40 ms chunks, then silence so turn detection can fire.

    The API drops audio sent faster than real time, so live runs must pace.
    `marks["speech_end"]` is set (monotonic clock) once the last speech chunk has
    been handed over. `speech_end_ms` marks where speech ends inside a clip that
    has trailing silence; by default the whole clip counts as speech.
    """
    chunk = sample_rate * 2 * CHUNK_MS // 1000
    data = path.read_bytes()
    pieces = [data[i : i + chunk] for i in range(0, len(data), chunk)]
    n_speech = len(pieces) if speech_end_ms is None else min(len(pieces), math.ceil(speech_end_ms / CHUNK_MS))
    pieces += [b"\x00" * chunk] * (tail_ms // CHUNK_MS)
    for i, piece in enumerate(pieces):
        yield piece
        if marks is not None and i == n_speech - 1:
            marks["speech_end"] = time.monotonic()
        if realtime:
            await asyncio.sleep(CHUNK_MS / 1000)
