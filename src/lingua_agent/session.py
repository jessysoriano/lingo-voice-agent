"""Async voice session: connect, stream mic audio, play reply audio, run tools.

Design notes
  * Transport is injected (`connect`) so tests drive it with a scripted fake.
  * The audio source is read by one pump task that outlives any single
    connection, so a reconnect keeps the same microphone stream. Audio captured
    while disconnected is dropped oldest-first: the API wants real-time pacing,
    and replaying stale speech after a gap would be wrong anyway.
  * Transient failures retry with full-jitter exponential backoff, and the retry
    budget resets once a connection reaches session.ready. After a drop we send
    session.resume (held 30 s server-side); if the server says the session is
    gone we fall back to a fresh session.
  * session.error is classified by code: auth is fatal, capacity is retried,
    a rejected client message after session.ready is logged and the call goes on.
  * Tool results are sent on reply.done and discarded if that reply was
    interrupted, as the API requires. On an interruption, queued playback is
    flushed so the learner does not hear stale speech.
  * A tool that raises never kills the loop; the model gets an error result.
  * Learner text is redacted before it is stored or logged.
  * stop() ends the session even if the server is silent, and sends session.end.
    So does cancellation (Ctrl+C), because billing follows connection time.
  * A failing audio source (microphone unplugged) ends the session with an error
    instead of leaving a call that can no longer hear.
  * The voice is fixed per session, so a language change is a new session: see
    _Switch and _begin_switch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any

from websockets.exceptions import ConnectionClosedError, InvalidHandshake, InvalidStatus

from . import guardrails, protocol
from .config import Settings
from .metrics import LatencyTracker
from .protocol import Event, ProtocolError, ToolCall

log = logging.getLogger("lingua_agent.session")

ToolFn = Callable[[Mapping[str, Any]], dict[str, Any]]
Connect = Callable[[], AbstractAsyncContextManager[Any]]

AUDIO_QUEUE_CHUNKS = 25  # ~1 s of 40 ms chunks buffered across a reconnect


class SessionError(RuntimeError):
    """Unrecoverable session failure."""


class _Reconnect(Exception):
    """Internal: the server asked for something a new connection can fix."""


class _Switch(Exception):
    """Internal: the learner asked for a language whose voice differs from this session's.
    The voice is bound to the session at start, so the only way to change it is a new session."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


HANDOFF_INSTRUCTIONS = (
    "The learner's last request is in the conversation. Teach it now, following your teaching steps. "
    "Do not greet them and do not mention the switch."
)
END_TIMEOUT_S = 2.0  # never let a dead socket hold up shutdown


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2**(attempt-1)))."""
    return rng.uniform(0, min(cap, base * 2 ** (attempt - 1)))


@dataclass
class Turn:
    role: str  # "user" | "agent"
    text: str
    flags: tuple[str, ...] = ()
    interrupted: bool = False


@dataclass
class SessionReport:
    session_id: str | None
    turns: list[Turn] = field(default_factory=list)
    guard_events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    reconnects: int = 0
    dropped_audio_chunks: int = 0
    language_switches: list[str] = field(default_factory=list)


class VoiceSession:
    def __init__(
        self,
        settings: Settings,
        *,
        session_update: dict[str, Any],
        tools: Mapping[str, ToolFn],
        audio_source: AsyncIterator[bytes],
        audio_sink: Callable[[bytes], None],
        on_interrupt: Callable[[], None] | None = None,
        connect: Connect | None = None,
        on_event: Callable[[Event], None] | None = None,
        tracker: LatencyTracker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        ready_timeout_s: float = 10.0,
        switch_update: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._update = session_update
        self._tools = dict(tools)
        self._source = audio_source
        self._sink = audio_sink
        self._on_interrupt = on_interrupt
        self._connect = connect or self._default_connect
        self._on_event = on_event
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._ready_timeout = ready_timeout_s
        self.tracker = tracker or LatencyTracker()
        self.report = SessionReport(session_id=None)
        self._pending_results: list[dict[str, Any]] = []
        self._audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=AUDIO_QUEUE_CHUNKS)
        self._ready = asyncio.Event()        # current connection is ready
        self._first_ready = asyncio.Event()  # some connection has been ready: start capturing
        self._stop = asyncio.Event()
        self._attempt = 0
        # Language switching: a tool may answer "switch_language"; we then start a new session
        # with the update this callback returns (new voice, prompt) and hand the request over.
        self._switch_update = switch_update
        self._switch_to: str | None = None
        self._switch_request: str | None = None
        self._handoff: str | None = None
        self._source_error: BaseException | None = None

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> SessionReport:
        """The report so far, with current metrics (usable after a cancellation)."""
        self.report.metrics = self.tracker.summary()
        return self.report

    def _default_connect(self) -> AbstractAsyncContextManager[Any]:
        from websockets.asyncio.client import connect

        return connect(
            self.settings.ws_url,
            additional_headers={"Authorization": f"Bearer {self.settings.api_key}"},
            max_size=2**22,
        )

    # ------------------------------------------------------------------ lifecycle
    async def run(self) -> SessionReport:
        pump = asyncio.create_task(self._pump_audio())
        pump.add_done_callback(self._on_pump_done)
        try:
            while not self._stop.is_set():
                try:
                    await self._run_once()
                    break  # clean close or stop()
                except _Switch as switch:
                    self._begin_switch(switch.code)  # not a failure: no backoff, no retry count
                except Exception as exc:
                    self._raise_if_fatal(exc)
                    self._attempt += 1
                    if self._attempt >= self.settings.max_connect_attempts:
                        raise SessionError(f"giving up after {self._attempt} attempts: {exc!r}") from exc
                    delay = backoff_delay(
                        self._attempt, self.settings.backoff_base_s, self.settings.backoff_cap_s, self._rng)
                    log.warning("connection problem (%r); retry %d in %.2fs", exc, self._attempt, delay)
                    self.report.reconnects += 1
                    await self._sleep(delay)
                    if self._switch_to and self._switch_update:
                        # The drop came between the tool call and reply.done: the old voice's
                        # session is not worth resuming, go straight to the new one.
                        self._begin_switch(self._switch_to)
            if self._source_error is not None:
                raise SessionError(f"audio source failed: {self._source_error!r}") from self._source_error
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        return self.snapshot()

    def _on_pump_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:  # e.g. the microphone went away: end the call, do not sit there deaf
            log.error("audio source failed: %r", exc)
            self._source_error = exc
            self._stop.set()

    def _raise_if_fatal(self, exc: Exception) -> None:
        """Return for errors worth retrying; re-raise everything else."""
        if (isinstance(exc, ConnectionClosedError) and exc.rcvd is not None and exc.rcvd.code == 1008
                and not self.report.session_id):
            # Policy violation before a session existed: the key or the config was refused.
            # Retrying the same thing cannot work, so say so instead of looping.
            raise SessionError(
                "the server refused the session (WebSocket close 1008, no reason given). "
                "Most likely the API key is wrong, revoked or has no Voice Agent access; "
                "run: python scripts/check_connection.py") from exc
        if isinstance(exc, InvalidStatus):  # HTTP error on the WebSocket handshake
            status = exc.response.status_code
            if status in (401, 403):
                raise SessionError(f"authentication failed (HTTP {status}): check ASSEMBLYAI_API_KEY") from exc
            if status == 429 or status >= 500:
                return
            raise SessionError(f"handshake rejected (HTTP {status})") from exc
        if isinstance(exc, (ConnectionClosedError, InvalidHandshake, OSError, _Reconnect)):
            return  # OSError covers TimeoutError and DNS/socket failures
        raise exc

    async def _run_once(self) -> None:
        self._ready.clear()
        self._pending_results.clear()
        async with self._connect() as ws:
            if self.report.session_id:
                await ws.send(json.dumps(protocol.build_session_resume(self.report.session_id)))
            else:
                await ws.send(json.dumps(self._update))
            receiver = asyncio.create_task(self._receive(ws))
            sender = asyncio.create_task(self._send_audio(ws))
            stopper = asyncio.create_task(self._stop.wait())
            pending = {receiver, sender, stopper}
            try:
                # The connection lives until the receiver ends or stop() is called.
                # A sender that finishes (finite audio) is fine; one that raises
                # (the server never sent session.ready) must surface.
                while receiver in pending and stopper in pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
            except (_Switch, asyncio.CancelledError):
                # A language switch, or Ctrl+C: this session is over, so stop its billing now
                # instead of leaving it held open for the resume window.
                await self._end_quietly(ws)
                raise
            finally:
                for task in (receiver, sender, stopper):
                    task.cancel()
                await asyncio.gather(receiver, sender, stopper, return_exceptions=True)
            if self._stop.is_set():
                await self._end_quietly(ws)

    @staticmethod
    async def _end_quietly(ws: Any) -> None:
        try:  # best effort: stop billing now instead of after the resume window
            async with asyncio.timeout(END_TIMEOUT_S):
                await ws.send(json.dumps(protocol.build_session_end()))
        except Exception:  # the socket may already be gone
            log.debug("could not send session.end", exc_info=True)

    def _begin_switch(self, code: str) -> None:
        assert self._switch_update is not None  # only raised when a callback exists
        last_user = next((t.text for t in reversed(self.report.turns) if t.role == "user"), None)
        self._update = self._switch_update(code)
        self._handoff = last_user or self._switch_request
        self.report.session_id = None  # a new session, not a resume of the old voice
        self.report.language_switches.append(code)
        self._attempt = 0
        self._switch_to = None
        self.tracker.switch_started()
        if self._on_interrupt:
            self._on_interrupt()  # cut any filler still playing in the old voice
        log.info("switching to a new session for language %s", code)

    # ------------------------------------------------------------------ audio out
    async def _pump_audio(self) -> None:
        """Read the source once, for the whole session, into a bounded queue."""
        await self._first_ready.wait()  # do not capture before the agent can listen
        async for chunk in self._source:
            self._offer(chunk)
            await asyncio.sleep(0)  # let the sender run even for a non-blocking source
        self._offer(None)  # source exhausted

    def _offer(self, item: bytes | None) -> None:
        if self._audio_q.full():  # nobody is draining (disconnected): drop the oldest chunk
            self._audio_q.get_nowait()
            self.report.dropped_audio_chunks += 1
        self._audio_q.put_nowait(item)

    async def _send_audio(self, ws: Any) -> None:
        # asyncio.timeout, not asyncio.wait_for: on Python 3.11 wait_for swallows a cancellation
        # that lands as the awaited event completes (the socket closing right after
        # session.ready), which left this task alive and shutdown hanging.
        async with asyncio.timeout(self._ready_timeout):
            await self._ready.wait()
        while (chunk := await self._audio_q.get()) is not None:
            await ws.send(json.dumps(protocol.build_input_audio(chunk)))

    # ------------------------------------------------------------------ events in
    async def _receive(self, ws: Any) -> None:
        async for raw in ws:
            try:
                event = protocol.parse_event(raw)
            except ProtocolError as exc:
                log.warning("ignoring malformed server message: %s", exc)
                continue
            await self._handle(event, ws)
            if self._stop.is_set():
                break

    async def _handle(self, ev: Event, ws: Any) -> None:
        if self._on_event:
            self._on_event(ev)
        t = ev.type
        if t == "session.ready":
            self.report.session_id = ev.data.get("session_id") or self.report.session_id
            self._attempt = 0  # a healthy connection earns a fresh retry budget
            self._ready.set()
            self._first_ready.set()
            if self._handoff:  # first session after a language switch: carry the request over
                request, self._handoff = self._handoff, None
                await ws.send(json.dumps(protocol.build_conversation_message("user", request)))
                await ws.send(json.dumps(protocol.build_reply_create(HANDOFF_INSTRUCTIONS)))
        elif protocol.is_error(ev):
            self._on_server_error(ev)
        elif t == "input.speech.started":
            self.tracker.speech_started()
        elif t == "input.speech.stopped":
            self.tracker.speech_stopped()
        elif t == "reply.started":
            self.tracker.reply_started()
        elif t == "reply.audio":
            self.tracker.audio_chunk()
            try:
                self._sink(protocol.decode_audio(ev))
            except ProtocolError as exc:
                log.warning("dropping bad audio chunk: %s", exc)
        elif t == "transcript.user":
            self._record_user(protocol.text_of(ev))
        elif t == "transcript.agent":
            self._record_agent(protocol.text_of(ev), protocol.is_interrupted(ev))
        elif t == "tool.call":
            self._run_tool(ev)
        elif t == "reply.done":
            await self._on_reply_done(ev, ws)

    def _on_server_error(self, ev: Event) -> None:
        code = protocol.error_code(ev)
        message = str(ev.data.get("message", ""))
        self.report.errors.append({"code": code, "message": message})
        if code in protocol.AUTH_ERROR_CODES:
            raise SessionError(f"authentication failed ({code}): check ASSEMBLYAI_API_KEY")
        if code in protocol.RESUME_ERROR_CODES:
            self.report.session_id = None  # the old session is gone: start a new one
            raise _Reconnect(code)
        if code in protocol.RETRYABLE_ERROR_CODES:
            raise _Reconnect(code)
        if not self._ready.is_set():
            # e.g. invalid_config / agent_not_found on the first session.update:
            # retrying the same message cannot succeed.
            raise SessionError(f"session rejected before ready ({code}): {message}")
        log.warning("server rejected a client message (%s): %s", code, message)

    async def _on_reply_done(self, ev: Event, ws: Any) -> None:
        interrupted = protocol.is_interrupted(ev)
        self.tracker.reply_done(interrupted)
        if interrupted:
            self._pending_results.clear()  # results for a reply the learner cut off are void
            self._switch_to = None
            if self._on_interrupt:
                self._on_interrupt()
            return
        if self._switch_to:
            self._pending_results.clear()
            raise _Switch(self._switch_to)
        for result in self._pending_results:
            await ws.send(json.dumps(result))
        if self._pending_results:
            self.tracker.tool_results_sent()
        self._pending_results.clear()

    def _record_user(self, text: str) -> None:
        verdict = guardrails.check_transcript(text)
        self.report.turns.append(Turn("user", verdict.redacted or "", verdict.flags))
        if verdict.flags:
            self.report.guard_events.append({"stage": "input", "flags": list(verdict.flags)})
            log.warning("input guard flags: %s", verdict.flags)

    def _record_agent(self, text: str, interrupted: bool) -> None:
        verdict = guardrails.check_agent_text(text)
        self.report.turns.append(Turn("agent", text, verdict.flags, interrupted))
        if verdict.flags:
            self.report.guard_events.append({"stage": "output", "flags": list(verdict.flags)})
            log.warning("output guard flags: %s", verdict.flags)

    def _run_tool(self, ev: Event) -> None:
        try:
            call: ToolCall = protocol.parse_tool_call(ev)
        except ProtocolError as exc:
            log.error("unparseable tool.call: %s", exc)
            return
        fn = self._tools.get(call.name)
        failed = True
        if fn is None:
            result: dict[str, Any] = {"status": "error", "reason": "unknown_tool"}
        else:
            try:
                result = fn(call.arguments)
                failed = False
            except Exception:  # a buggy tool must not end the session
                log.exception("tool %s raised", call.name)
                result = {"status": "error", "reason": "tool_failed"}
        self.report.tool_calls.append({
            "name": call.name, "arguments": call.arguments,
            "status": result.get("status"), "reason": result.get("reason"),
        })
        if result.get("status") == "rejected":
            self.report.guard_events.append({"stage": "tool", "flags": [str(result.get("reason"))]})
        if result.get("status") == "switch_language" and self._switch_update:
            # No tool.result: this session is about to end. Switch when the filler reply finishes.
            self._switch_to = str(result["language_code"])
            self._switch_request = f"How do you say {call.arguments.get('source_text', 'that')} in {result.get('language')}?"
            return
        self._pending_results.append(protocol.build_tool_result(call, result, is_error=failed))
