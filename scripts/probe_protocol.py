"""Record raw Voice Agent API events from one real exchange.

    ffmpeg -i question.wav -ar 24000 -ac 1 -f s16le question.pcm
    python scripts/probe_protocol.py question.pcm

Record yourself asking "How do I say where is the library in Spanish?" so the
tool-call path is exercised. Writes results/probe-<timestamp>.jsonl with one
event per line (audio payloads truncated) and prints the event types seen.
Compare the field names with src/lingua_agent/protocol.py; if anything differs,
fix that module and the fixtures in tests/. Useful after any API change, and to
confirm the wire format on a new account or region.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

from lingua_agent import agent_spec, coaching, protocol, tutor
from lingua_agent.config import ConfigError, Settings, load_dotenv
from lingua_agent.pcm import paced_pcm
from lingua_agent.session import SessionError, VoiceSession

WAIT_AFTER_AUDIO_S = 20


async def main(pcm_path: Path) -> int:
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    out = Path("results") / f"probe-{int(time.time())}.jsonl"
    out.parent.mkdir(exist_ok=True)
    seen: Counter[str] = Counter()
    clip_s = len(pcm_path.read_bytes()) / (settings.sample_rate * 2)

    with out.open("w", encoding="utf-8") as fh:
        def record(ev: protocol.Event) -> None:
            seen[ev.type] += 1
            row = {k: (v[:32] + "...<truncated>" if k in ("audio", "data") and isinstance(v, str) else v)
                   for k, v in ev.data.items()}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        update = protocol.build_session_update_inline(
            agent_spec.build_session_config(settings, greeting=None, can_switch=False))
        session = VoiceSession(
            settings, session_update=update,
            tools={tutor.TOOL_NAME: tutor.teach_phrase, tutor.CHECK_TOOL_NAME: coaching.check_attempt},
            audio_source=paced_pcm(pcm_path, settings.sample_rate, realtime=True,
                                   tail_ms=WAIT_AFTER_AUDIO_S * 1000),
            audio_sink=lambda b: None, on_event=record,
        )
        asyncio.get_running_loop().call_later(clip_s + WAIT_AFTER_AUDIO_S, session.stop)
        try:
            report = await session.run()
        except SessionError as exc:
            print(f"session failed: {exc}", file=sys.stderr)
            return 1

    print(f"wrote {out}")
    print("events:", dict(seen))
    print("tool calls:", [(c["name"], c["status"], c["reason"]) for c in report.tool_calls])
    expected = {"session.ready", "transcript.user", "reply.audio", "transcript.agent", "reply.done", "tool.call"}
    if missing := expected - set(seen):
        print(f"NOT SEEN (check protocol.py and the prompt): {sorted(missing)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(asyncio.run(main(Path(sys.argv[1]))))
