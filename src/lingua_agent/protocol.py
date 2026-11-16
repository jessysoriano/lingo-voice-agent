"""Voice Agent API wire format, isolated in one module.

Everything the rest of the code knows about message shapes lives here, so a
change in the API (or a correction from a live run) is a one-file fix.

Checked against AssemblyAI's events reference, audio-format and tool-calling docs
(Nov 2026):
  * client events: session.update, session.resume, session.end, input.audio
    ("audio": base64), tool.result ("call_id", "result": JSON string, "is_error"),
    conversation.message ("role", "content"; adds context, no reply) and
    reply.create ("instructions"; makes the agent reply now)
  * server events: session.ready ("session_id"), session.updated, session.ended,
    session.error ("code", "message"), input.speech.started/stopped,
    transcript.user.delta ("text", cumulative), transcript.user ("text"),
    reply.started, reply.audio ("data": base64; NOT "audio"), transcript.agent
    ("text", "interrupted"), reply.done ("status": completed|interrupted),
    tool.call ("call_id", "name", "arguments": object)
  * audio: PCM16 mono 24 kHz, sent at real-time pace

Where the docs disagree with each other, the parsers accept both spellings:
  * tool.call id is "call_id" in the events reference but "tool_call_id" on the
    message-sequence page; tool.result echoes whichever key the server used
  * the final user transcript is "text" in the reference, "transcript" in one example
Confirmed by live sessions: connect, session.update (inline), input.audio,
reply.audio, both transcripts, tool.call and tool.result on reply.done.
Taken from the docs but not yet exercised live: session.resume, conversation.message
and reply.create. scripts/probe_protocol.py records the raw events of a real exchange.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

# session.error codes, lower-cased (the docs mix cases).
AUTH_ERROR_CODES = frozenset({"unauthorized", "forbidden"})
RESUME_ERROR_CODES = frozenset({"session_not_found", "session_forbidden", "session_expired"})
RETRYABLE_ERROR_CODES = frozenset({"at_capacity", "concurrency_exceeded", "internal_error"})


class ProtocolError(ValueError):
    """A message could not be parsed into the shape we expect."""


@dataclass(frozen=True)
class Event:
    type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    id_field: str = "call_id"  # the key the server used; tool.result echoes it


def parse_event(raw: str | bytes) -> Event:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"invalid JSON from server: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        raise ProtocolError("server message has no string 'type'")
    return Event(type=payload["type"], data=payload)


def is_error(event: Event) -> bool:
    return event.type in ("session.error", "error")


def error_code(event: Event) -> str:
    return str(event.data.get("code", "")).strip().lower()


def text_of(event: Event) -> str:
    """Transcript text of a transcript.* event ('' if absent)."""
    value = event.data.get("text", event.data.get("transcript", ""))
    return value if isinstance(value, str) else ""


def is_interrupted(event: Event) -> bool:
    """True for a reply.done / transcript.agent cut short by the user barging in."""
    return event.data.get("status") == "interrupted" or event.data.get("interrupted") is True


def parse_tool_call(event: Event) -> ToolCall:
    d = event.data
    id_field = next((k for k in ("call_id", "tool_call_id", "id") if d.get(k)), None)
    name = d.get("name")
    args = d.get("arguments", {})
    if isinstance(args, str):  # documented as an object; tolerate a JSON string
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"tool arguments are not valid JSON: {exc}") from exc
    if id_field is None or not name:
        raise ProtocolError("tool.call is missing a call id or tool name")
    if not isinstance(args, dict):
        raise ProtocolError("tool arguments must be a JSON object")
    return ToolCall(call_id=str(d[id_field]), name=str(name), arguments=args, id_field=id_field)


def decode_audio(event: Event) -> bytes:
    b64 = event.data.get("data") or event.data.get("audio")
    if not isinstance(b64, str):
        raise ProtocolError("reply.audio has no base64 'data' field")
    try:
        return base64.b64decode(b64, validate=True)
    except ValueError as exc:
        raise ProtocolError(f"reply.audio is not valid base64: {exc}") from exc


def build_session_update_for_agent(agent_id: str) -> dict[str, Any]:
    return {"type": "session.update", "session": {"agent_id": agent_id}}


def build_session_update_inline(config: dict[str, Any]) -> dict[str, Any]:
    """Inline config (system_prompt, greeting, tools, input, output). Mutually
    exclusive with agent_id binding per the docs."""
    if "agent_id" in config:
        raise ProtocolError("agent_id cannot be combined with inline session fields")
    return {"type": "session.update", "session": config}


def build_session_resume(session_id: str) -> dict[str, Any]:
    return {"type": "session.resume", "session_id": session_id}


def build_session_end() -> dict[str, Any]:
    """Ends the session now. Billing follows open connection time, so always send it."""
    return {"type": "session.end"}


def build_conversation_message(role: str, content: str) -> dict[str, Any]:
    """Adds a message to the agent's context. It does not make the agent reply by itself."""
    return {"type": "conversation.message", "role": role, "content": content}


def build_reply_create(instructions: str | None = None) -> dict[str, Any]:
    """Makes the agent reply now, composing the reply from the instructions (not read verbatim)."""
    event: dict[str, Any] = {"type": "reply.create"}
    if instructions:
        event["instructions"] = instructions
    return event


def build_input_audio(pcm16: bytes) -> dict[str, Any]:
    return {"type": "input.audio", "audio": base64.b64encode(pcm16).decode("ascii")}


def build_tool_result(call: ToolCall, result: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "tool.result",
        call.id_field if call.id_field != "id" else "call_id": call.call_id,
        "result": json.dumps(result, ensure_ascii=False),
    }
    if is_error:
        msg["is_error"] = True
    return msg
