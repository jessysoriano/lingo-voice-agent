"""Find out why a session will not start: bad key, or a config the server rejects.

    python scripts/check_connection.py

Step 1 sends a minimal session.update (no tools, default voice).
Step 2 sends the full config the agent really uses.
Both print session.ready or the server's reason. Step 1 failing means the key
(or the network); step 1 passing and step 2 failing means the config.
"""

from __future__ import annotations

import asyncio
import json
import sys

import websockets

from lingua_agent import agent_spec, protocol
from lingua_agent.config import ConfigError, Settings, load_dotenv


async def attempt(settings: Settings, label: str, update: dict) -> bool:
    try:
        async with websockets.connect(
            settings.ws_url, additional_headers={"Authorization": f"Bearer {settings.api_key}"}
        ) as ws:
            await ws.send(json.dumps(update))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                ev = json.loads(raw)
                if ev.get("type") == "session.ready":
                    print(f"[{label}] OK: session.ready")
                    return True
                if ev.get("type") in ("session.error", "error"):
                    print(f"[{label}] server error: {json.dumps(ev)[:400]}")
                    return False
    except websockets.ConnectionClosed as exc:
        code = exc.rcvd.code if exc.rcvd else None
        reason = exc.rcvd.reason if exc.rcvd else ""
        print(f"[{label}] server closed the socket: code={code} reason={reason!r}")
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic
        print(f"[{label}] {type(exc).__name__}: {exc}")
    return False


async def main() -> int:
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 2
    key = settings.api_key
    print(f"key loaded: {key[:4]}...{key[-4:]} ({len(key)} chars), url: {settings.ws_url}")
    minimal = protocol.build_session_update_inline({"system_prompt": "You are a test.", "greeting": "Hello."})
    ok1 = await attempt(settings, "minimal", minimal)
    full = protocol.build_session_update_inline(
        agent_spec.build_session_config(settings, can_switch=True))
    ok2 = await attempt(settings, "full config", full)
    if not ok1:
        print("-> Even the minimal session failed: check the API key (and that the account has Voice Agent access).")
    elif not ok2:
        print("-> Key works, full config rejected: send me this output.")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
