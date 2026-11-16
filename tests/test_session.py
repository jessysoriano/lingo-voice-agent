import asyncio
import json
import random

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.http11 import Response

from conftest import FakeWS, b64, chunks, make_connect
from lingua_agent import tutor
from lingua_agent.session import SessionError, VoiceSession, backoff_delay

UPDATE = {"type": "session.update", "session": {"agent_id": "agent-1"}}
ES_ARGS = {"target_language": "Spanish", "source_text": "Where is the library?",
           "phrase": "¿Dónde está la biblioteca?", "pronunciation": "DOHN-deh ehs-TAH lah bee-blee-oh-TEH-kah"}
TOOLS = {tutor.TOOL_NAME: tutor.teach_phrase}
READY = {"type": "session.ready", "session_id": "s1"}


async def _noop(_delay=None):
    return None


def make_session(settings, connect, *, source=None, tools=TOOLS, sink=None, sleep=_noop, **kw):
    return VoiceSession(
        settings, session_update=UPDATE, tools=tools, audio_source=source or chunks(b"\x01\x02", b"\x03\x04"),
        audio_sink=sink or (lambda b: None), connect=connect, sleep=sleep, rng=random.Random(0), **kw,
    )


def happy_script():
    """One teaching turn in the documented event order."""
    return [
        READY,
        {"type": "input.speech.started"},
        {"type": "transcript.user.delta", "text": "how do I"},
        {"type": "input.speech.stopped"},
        {"type": "transcript.user", "text": "how do I say where is the library, my email is jo@example.com"},
        {"type": "reply.started", "reply_id": "fc-c1"},
        {"type": "tool.call", "call_id": "c1", "name": "teach_phrase", "arguments": ES_ARGS},
        {"type": "reply.done", "reply_id": "fc-c1", "status": "completed"},
        {"type": "reply.started", "reply_id": "r2"},
        {"type": "reply.audio", "data": b64(b"PCM!")},
        {"type": "transcript.agent", "text": "Sure. Dónde está la biblioteca.", "interrupted": False},
        {"type": "reply.done", "reply_id": "r2", "status": "completed"},
    ]


async def test_happy_path_runs_tool_and_reports(settings):
    ws, played = FakeWS(happy_script()), []
    report = await make_session(settings, make_connect(ws), sink=played.append).run()

    assert ws.sent[0] == UPDATE
    assert "input.audio" in ws.types()
    results = [m for m in ws.sent if m["type"] == "tool.result"]
    assert len(results) == 1 and results[0]["call_id"] == "c1" and "is_error" not in results[0]
    assert json.loads(results[0]["result"])["status"] == "approved"
    assert played == [b"PCM!"]
    assert report.session_id == "s1"
    assert report.turns[0].text.endswith("[EMAIL]") and "pii_email" in report.turns[0].flags
    assert report.tool_calls[0]["status"] == "approved"
    assert report.tool_calls[0]["arguments"]["phrase"] == "¿Dónde está la biblioteca?"
    assert report.metrics["turns_measured"] == 1 and report.metrics["interruptions"] == 0
    assert report.metrics["tool_turns"] == 1  # post-tool latency is recorded for the lesson itself
    assert {"stage": "input", "flags": ["pii_email"]} in report.guard_events


async def test_tool_result_waits_for_reply_done(settings):
    ws = FakeWS(happy_script()[:7])  # tool.call arrives, reply.done never does
    await make_session(settings, make_connect(ws)).run()
    assert "tool.result" not in ws.types()


async def test_tool_result_echoes_the_id_key_the_server_used(settings):
    script = [READY, {"type": "tool.call", "tool_call_id": "t9", "name": "teach_phrase", "arguments": ES_ARGS},
              {"type": "reply.done", "status": "completed"}]
    ws = FakeWS(script)
    await make_session(settings, make_connect(ws)).run()
    result = next(m for m in ws.sent if m["type"] == "tool.result")
    assert result["tool_call_id"] == "t9" and "call_id" not in result


async def test_interrupted_reply_flushes_playback_and_discards_tool_results(settings):
    flushed = []
    script = [READY, {"type": "reply.started"},
              {"type": "tool.call", "call_id": "c1", "name": "teach_phrase", "arguments": ES_ARGS},
              {"type": "transcript.agent", "text": "One mo", "interrupted": True},
              {"type": "reply.done", "status": "interrupted"},
              {"type": "reply.done", "status": "completed"}]
    ws = FakeWS(script)
    report = await make_session(settings, make_connect(ws), on_interrupt=lambda: flushed.append(1)).run()
    assert flushed == [1]
    assert "tool.result" not in ws.types()  # not sent on the interrupted reply, nor leaked into the next
    assert report.metrics["interruptions"] == 1 and report.turns[0].interrupted


async def test_unknown_and_crashing_tools_return_error_results(settings):
    def boom(args):
        raise RuntimeError("kaput")

    script = [
        READY,
        {"type": "tool.call", "call_id": "a", "name": "nope", "arguments": {}},
        {"type": "tool.call", "call_id": "b", "name": "boom", "arguments": "{}"},
        {"type": "tool.call", "name": "boom"},  # no call id: unparseable, skipped
        {"type": "reply.done", "status": "completed"},
    ]
    ws = FakeWS(script)
    report = await make_session(settings, make_connect(ws), tools={"boom": boom}).run()
    results = {m["call_id"]: m for m in ws.sent if m["type"] == "tool.result"}
    assert {k: json.loads(v["result"])["reason"] for k, v in results.items()} == {"a": "unknown_tool", "b": "tool_failed"}
    assert all(v["is_error"] is True for v in results.values())
    assert len(report.tool_calls) == 2


async def test_rejected_lesson_is_recorded_as_a_guard_event(settings):
    bad = {**ES_ARGS, "phrase": "Where is the library"}
    ws = FakeWS([READY, {"type": "tool.call", "call_id": "c", "name": "teach_phrase", "arguments": bad},
                 {"type": "reply.done", "status": "completed"}])
    report = await make_session(settings, make_connect(ws)).run()
    assert {"stage": "tool", "flags": ["untranslated"]} in report.guard_events
    assert "is_error" not in next(m for m in ws.sent if m["type"] == "tool.result")  # a rejection is a valid result


# ---------------------------------------------------------------- reconnects
async def test_retries_with_backoff_then_succeeds(settings):
    delays = []

    async def sleep(d):
        delays.append(d)

    report = await make_session(settings, make_connect(OSError("x"), OSError("y"), FakeWS([READY])), sleep=sleep).run()
    assert report.reconnects == 2 and len(delays) == 2 and all(0 <= d <= 0.002 for d in delays)


async def test_gives_up_after_max_attempts(settings):
    with pytest.raises(SessionError, match="giving up"):
        await make_session(settings, make_connect(OSError("1"), OSError("2"), OSError("3"))).run()


async def test_retry_budget_resets_after_a_healthy_connection(settings):
    drop = ConnectionClosedError(None, None)
    conns = [FakeWS([READY, drop]) for _ in range(3)] + [FakeWS([READY])]
    report = await make_session(settings, make_connect(*conns)).run()  # max_connect_attempts is 3
    assert report.reconnects == 3


async def test_reconnect_resumes_session_and_keeps_streaming_audio(settings):
    async def slow_mic():
        for i in range(40):
            yield bytes([i])
            await asyncio.sleep(0.002)

    first = FakeWS([READY, ConnectionClosedError(None, None)])
    second = FakeWS([READY] + [{"type": "session.updated"}] * 6)
    report = await make_session(settings, make_connect(first, second), source=slow_mic()).run()
    assert second.sent[0] == {"type": "session.resume", "session_id": "s1"}
    assert "input.audio" in second.types()  # the microphone survives the reconnect
    assert report.reconnects == 1


async def test_expired_session_falls_back_to_a_fresh_one(settings):
    first = FakeWS([READY, ConnectionClosedError(None, None)])
    second = FakeWS([{"type": "session.error", "code": "session_expired", "message": "gone"}])
    third = FakeWS([{"type": "session.ready", "session_id": "s2"}])
    report = await make_session(settings, make_connect(first, second, third)).run()
    assert second.sent[0]["type"] == "session.resume" and third.sent[0] == UPDATE
    assert report.session_id == "s2" and report.errors[0]["code"] == "session_expired"


# ---------------------------------------------------------------- server errors
async def test_auth_error_is_fatal_and_not_retried(settings):
    ws = FakeWS([{"type": "session.error", "code": "UNAUTHORIZED", "message": "bad key"}])
    with pytest.raises(SessionError, match="authentication failed"):
        await make_session(settings, make_connect(ws)).run()


async def test_capacity_error_is_retried(settings):
    busy = FakeWS([{"type": "session.error", "code": "at_capacity", "message": "busy"}])
    report = await make_session(settings, make_connect(busy, FakeWS([READY]))).run()
    assert report.reconnects == 1 and report.session_id == "s1"


async def test_config_rejected_before_ready_is_fatal(settings):
    ws = FakeWS([{"type": "session.error", "code": "invalid_config", "message": "unknown voice"}])
    with pytest.raises(SessionError, match="rejected before ready.*unknown voice"):
        await make_session(settings, make_connect(ws)).run()


async def test_rejected_client_message_after_ready_keeps_the_call_alive(settings):
    ws = FakeWS([READY, {"type": "session.error", "code": "invalid_value", "message": "nope"},
                 {"type": "transcript.agent", "text": "still here"}])
    report = await make_session(settings, make_connect(ws)).run()
    assert [t.text for t in report.turns] == ["still here"] and report.errors[0]["code"] == "invalid_value"


@pytest.mark.parametrize("status, outcome", [(401, "authentication failed"), (404, "handshake rejected")])
async def test_handshake_http_errors_are_fatal(settings, status, outcome):
    exc = InvalidStatus(Response(status, "x", Headers()))
    with pytest.raises(SessionError, match=outcome):
        await make_session(settings, make_connect(exc)).run()


async def test_handshake_503_is_retried(settings):
    exc = InvalidStatus(Response(503, "x", Headers()))
    report = await make_session(settings, make_connect(exc, FakeWS([READY]))).run()
    assert report.reconnects == 1


async def test_never_ready_surfaces_as_timeout_then_gives_up(settings):
    conns = [FakeWS([], hold_open=True) for _ in range(3)]
    with pytest.raises(SessionError, match="giving up"):
        await make_session(settings, make_connect(*conns), ready_timeout_s=0.02).run()


async def test_unexpected_exceptions_are_not_swallowed_as_retries(settings):
    def bad_sink(_):
        raise ZeroDivisionError

    ws = FakeWS([READY, {"type": "reply.audio", "data": b64(b"x")}])
    with pytest.raises(ZeroDivisionError):
        await make_session(settings, make_connect(ws), sink=bad_sink).run()


# ---------------------------------------------------------------- robustness
async def test_malformed_messages_and_bad_audio_are_skipped(settings):
    played = []
    ws = FakeWS([READY, "not json", {"type": "reply.audio"}, {"type": "reply.audio", "data": b64(b"ok")}])
    await make_session(settings, make_connect(ws), sink=played.append).run()
    assert played == [b"ok"]


async def test_stop_ends_a_silent_session_and_sends_session_end(settings):
    ws = FakeWS([READY], hold_open=True)  # the server never says another word
    session = make_session(settings, make_connect(ws))
    asyncio.get_running_loop().call_later(0.05, session.stop)
    report = await asyncio.wait_for(session.run(), timeout=2)
    assert ws.types()[-1] == "session.end" and report.session_id == "s1"


async def test_stop_from_an_event_handler_skips_later_events(settings):
    ws = FakeWS([READY, {"type": "transcript.agent", "text": "hi"}, {"type": "transcript.agent", "text": "never reached"}])
    holder = {}

    def on_event(ev):
        if ev.type == "transcript.agent":
            holder["s"].stop()

    holder["s"] = make_session(settings, make_connect(ws), on_event=on_event)
    report = await holder["s"].run()
    assert [t.text for t in report.turns] == ["hi"]


async def test_audio_backlog_is_dropped_oldest_first_not_replayed(settings):
    async def burst():
        for i in range(60):  # far more than the queue holds, with nobody draining it
            yield bytes([i])

    session = make_session(settings, make_connect(FakeWS([])), source=burst())
    session._first_ready.set()
    await session._pump_audio()
    assert session.report.dropped_audio_chunks > 0 and session._audio_q.full()


async def test_agent_output_flags_are_recorded(settings):
    ws = FakeWS([READY, {"type": "transcript.agent", "text": "**bold** move"}])
    report = await make_session(settings, make_connect(ws)).run()
    assert {"stage": "output", "flags": ["markdown"]} in report.guard_events


def test_backoff_is_bounded_and_grows():
    rng = random.Random(1)
    assert all(0 <= backoff_delay(1, 0.5, 8, rng) <= 0.5 for _ in range(50))
    assert max(backoff_delay(10, 0.5, 8, rng) for _ in range(200)) <= 8
    assert max(backoff_delay(4, 0.5, 8, rng) for _ in range(200)) > 0.5



# ---------- language switching: the voice is fixed per session, so a new language means a new session
FR_ARGS = {"target_language": "French", "source_text": "cookies", "phrase": "biscuits", "pronunciation": "bees-KWEE"}
SWITCH_TOOLS = {tutor.TOOL_NAME: lambda a: tutor.teach_phrase(a, course_language="Spanish", auto_switch=True)}
NEW_UPDATE = {"type": "session.update", "session": {"output": {"voice": "estelle"}}}


def switch_script():
    return [
        READY,
        {"type": "transcript.user", "text": "how do you say cookies in French"},
        {"type": "reply.started", "reply_id": "fc-c1"},
        {"type": "tool.call", "call_id": "c1", "name": "teach_phrase", "arguments": FR_ARGS},
        {"type": "reply.done", "reply_id": "fc-c1", "status": "completed"},
    ]


async def test_language_request_ends_the_session_and_starts_one_with_the_new_voice(settings):
    first = FakeWS(switch_script())
    second = FakeWS([{"type": "session.ready", "session_id": "s2"}] + [{"type": "session.updated"}] * 8)
    flushed, asked = [], []

    async def slow_mic():  # keeps producing audio across the switch
        for i in range(60):
            yield bytes([i])
            await asyncio.sleep(0.002)

    def switch_update(code):
        asked.append(code)
        return NEW_UPDATE

    session = make_session(settings, make_connect(first, second), tools=SWITCH_TOOLS, source=slow_mic(),
                           switch_update=switch_update, on_interrupt=lambda: flushed.append(1))
    report = await session.run()

    assert asked == ["fr"] and report.language_switches == ["fr"] and report.reconnects == 0
    assert "tool.result" not in first.types() and first.types()[-1] == "session.end"  # old voice stops billing
    assert second.sent[0] == NEW_UPDATE  # a fresh session.update, not a session.resume
    assert report.session_id == "s2" and flushed  # filler from the old voice is cut off
    handoff = [m for m in second.sent if m["type"] in ("conversation.message", "reply.create")]
    assert handoff[0] == {"type": "conversation.message", "role": "user", "content": "how do you say cookies in French"}
    assert handoff[1]["type"] == "reply.create" and "Teach it now" in handoff[1]["instructions"]
    assert report.tool_calls[0]["status"] == "switch_language"
    assert "input.audio" in second.types()  # the microphone carries straight over


async def test_switch_is_dropped_if_the_learner_interrupts_the_filler(settings):
    script = switch_script()[:-1] + [{"type": "reply.done", "reply_id": "fc-c1", "status": "interrupted"}]
    ws = FakeWS(script)
    report = await make_session(settings, make_connect(ws), tools=SWITCH_TOOLS,
                                switch_update=lambda c: NEW_UPDATE).run()
    assert report.language_switches == [] and "session.end" not in ws.types()[:-1]


async def test_without_a_switch_callback_the_tool_result_is_sent_as_usual(settings):
    ws = FakeWS(switch_script())
    report = await make_session(settings, make_connect(ws), tools=SWITCH_TOOLS).run()
    result = next(m for m in ws.sent if m["type"] == "tool.result")
    assert json.loads(result["result"])["status"] == "switch_language" and report.language_switches == []


async def test_switch_falls_back_to_the_tool_argument_when_no_transcript_was_heard(settings):
    script = [READY] + switch_script()[2:]  # no transcript.user before the tool call
    first, second = FakeWS(script), FakeWS([{"type": "session.ready", "session_id": "s2"}])
    await make_session(settings, make_connect(first, second), tools=SWITCH_TOOLS, switch_update=lambda c: NEW_UPDATE).run()
    msg = next(m for m in second.sent if m["type"] == "conversation.message")
    assert msg["content"] == "How do you say cookies in French?"


# ---------- final review: shutdown, a dead microphone, and switches that hit a network drop
async def test_cancellation_still_ends_the_session(settings):
    """Ctrl+C cancels the task. Billing follows connection time, so session.end must still go out."""
    ws = FakeWS([READY], hold_open=True)
    task = asyncio.create_task(make_session(settings, make_connect(ws)).run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ws.types()[-1] == "session.end"


async def test_a_dead_socket_cannot_hold_up_shutdown(settings, monkeypatch):
    from lingua_agent import session as session_module

    class StuckWS(FakeWS):
        async def send(self, msg):
            if json.loads(msg)["type"] == "session.end":
                await asyncio.Event().wait()  # never returns
            await super().send(msg)

    monkeypatch.setattr(session_module, "END_TIMEOUT_S", 0.01)
    ws = StuckWS([READY], hold_open=True)
    session = make_session(settings, make_connect(ws))
    asyncio.get_running_loop().call_later(0.05, session.stop)
    await asyncio.wait_for(session.run(), timeout=2)  # returns despite the stuck send


async def test_a_failing_microphone_ends_the_call_with_a_clear_error(settings):
    async def broken_mic():
        yield b"\x01\x02"
        raise OSError("input device unplugged")

    ws = FakeWS([READY], hold_open=True)
    with pytest.raises(SessionError, match="audio source failed"):
        await asyncio.wait_for(make_session(settings, make_connect(ws), source=broken_mic()).run(), timeout=2)
    assert ws.types()[-1] == "session.end"


async def test_switch_still_happens_if_the_connection_drops_before_reply_done(settings):
    first = FakeWS(switch_script()[:-1] + [ConnectionClosedError(None, None)])  # tool.call, then the socket dies
    second = FakeWS([{"type": "session.ready", "session_id": "s2"}])
    report = await make_session(settings, make_connect(first, second), tools=SWITCH_TOOLS,
                                switch_update=lambda c: NEW_UPDATE).run()
    assert second.sent[0] == NEW_UPDATE  # the new voice, not a resume of the old session
    assert report.language_switches == ["fr"] and report.reconnects == 1
    assert any(m["type"] == "reply.create" for m in second.sent)


async def test_switch_latency_is_measured(settings):
    first = FakeWS(switch_script())
    second = FakeWS([{"type": "session.ready", "session_id": "s2"}, {"type": "reply.started"},
                     {"type": "reply.audio", "data": b64(b"hi")}])
    report = await make_session(settings, make_connect(first, second), tools=SWITCH_TOOLS,
                                switch_update=lambda c: NEW_UPDATE).run()
    assert report.metrics["language_switches"] == 1 and report.metrics["switch_p50_ms"] > 0
    assert "greet" in next(m for m in second.sent if m["type"] == "reply.create")["instructions"]


async def test_socket_closing_right_after_ready_does_not_hang_shutdown(settings):
    """Regression for Python 3.11: the sender was cancelled in the same loop turn that
    session.ready arrived, asyncio.wait_for swallowed the cancel, and run() never returned."""
    class InstantWS(FakeWS):
        async def _gen(self):
            for item in self.script:  # no pause: ready and close land in one loop iteration
                yield json.dumps(item)

    async def endless_mic():
        while True:
            await asyncio.sleep(3600)
            yield b""

    ws = InstantWS([READY])
    async with asyncio.timeout(2):
        report = await make_session(settings, make_connect(ws), source=endless_mic()).run()
    assert report.session_id == "s1"
