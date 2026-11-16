"""Microphone and speaker plumbing (PCM16, mono, 24 kHz). Needs `pip install .[audio]`.

The hardware wrappers are thin and untested (they need a sound card); the buffer
logic they rely on is `PlaybackBuffer`, which is tested.

There is no echo cancellation here. Use headphones: on open speakers the agent
hears itself and interrupts its own replies.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

CHUNK_MS = 40


class PlaybackBuffer:
    """Thread-safe FIFO of PCM bytes between the event loop and the audio thread.

    `write` never blocks, so reply audio arriving faster than real time cannot
    stall the event loop (and with it the microphone and barge-in). `clear`
    drops queued speech when the learner interrupts.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()

    def write(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)

    def read(self, n: int) -> bytes:
        """Exactly n bytes, zero-padded (silence) when the buffer runs dry."""
        with self._lock:
            out = bytes(self._buf[:n])
            del self._buf[:n]
        return out + b"\x00" * (n - len(out))

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


def _sd():  # pragma: no cover
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("audio support missing: pip install '.[audio]' (needs PortAudio)") from exc
    return sd


async def mic_chunks(sample_rate: int = 24000) -> AsyncIterator[bytes]:  # pragma: no cover
    sd = _sd()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

    def offer(chunk: bytes) -> None:  # runs on the event loop
        if queue.full():
            queue.get_nowait()  # drop the oldest audio, never block
        queue.put_nowait(chunk)

    def callback(indata, frames, time_info, status):  # runs on the audio thread
        loop.call_soon_threadsafe(offer, bytes(indata))

    with sd.RawInputStream(
        samplerate=sample_rate, channels=1, dtype="int16",
        blocksize=sample_rate * CHUNK_MS // 1000, callback=callback,
    ):
        while True:
            yield await queue.get()


class Speaker:  # pragma: no cover
    def __init__(self, sample_rate: int = 24000) -> None:
        self._buffer = PlaybackBuffer()

        def callback(outdata, frames, time_info, status):  # runs on the audio thread
            outdata[:] = self._buffer.read(len(outdata))

        self._stream = _sd().RawOutputStream(
            samplerate=sample_rate, channels=1, dtype="int16", callback=callback)
        self._stream.start()

    def play(self, pcm16: bytes) -> None:
        self._buffer.write(pcm16)

    def flush(self) -> None:
        """Drop queued speech (the learner interrupted)."""
        self._buffer.clear()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
