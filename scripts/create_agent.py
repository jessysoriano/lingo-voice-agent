"""Create or update the saved Lingo agent and print its id.

    python scripts/create_agent.py --language es        # create a Spanish-course agent (native voice), prints its id
    python scripts/create_agent.py                      # create an all-language agent on the English voice
    python scripts/create_agent.py --update <agent-id>  # change an existing agent in place
    python scripts/create_agent.py --dry-run            # print the request body only

A saved agent keeps prompt, voice, keyterms and tool definitions server-side, so
prompt changes ship without redeploying clients and the same agent serves the
browser, WebSocket and telephony. Use --update for prompt changes: creating
again makes a second agent and leaves clients bound to the old one.

A saved agent has ONE voice, and each native voice speaks only its own language
plus English. So production is one saved agent per course language (--language),
and the client binds to the agent for the course the learner is in. Run the client
with the same --language so the lesson tool enforces it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

from lingua_agent import agent_spec
from lingua_agent.cli import apply_language
from lingua_agent.config import ConfigError, Settings, load_dotenv
from lingua_agent.prompts import DEFAULT_PROMPT_VERSION, PROMPTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    ap.add_argument("--language", metavar="CODE", help="course language (es, fr, de, it, pt): sets the native voice and prompt")
    ap.add_argument("--update", metavar="AGENT_ID", help="update this agent instead of creating a new one")
    ap.add_argument("--dry-run", action="store_true", help="print the request body and exit")
    args = ap.parse_args()
    load_dotenv()
    try:
        settings = apply_language(Settings.from_env(), args.language)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    body = agent_spec.build_agent_request(settings, args.prompt_version)
    if args.dry_run:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    url = f"{settings.rest_url}/v1/agents"
    if args.update:
        url += "/" + urllib.parse.quote(args.update, safe="")
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Authorization": settings.api_key, "Content-Type": "application/json"},
        method="PUT" if args.update else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # https enforced by Settings
            agent = json.load(resp)
    except urllib.error.HTTPError as exc:
        # A 400 here lists the valid voice ids if VOICE_ID is wrong.
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"network error: {exc.reason}", file=sys.stderr)
        return 1
    print(agent.get("id", json.dumps(agent)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
