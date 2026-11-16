"""Command line entry point: talk to Lingo from your terminal. Ctrl+C ends the call."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from . import agent_spec, coaching, guardrails, languages, protocol, shadow, tutor
from .config import ConfigError, Settings, load_dotenv
from .prompts import DEFAULT_PROMPT_VERSION, PROMPTS
from .session import SessionError, SessionReport, VoiceSession


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lingo-agent", description=__doc__)
    p.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    p.add_argument("--native-voices", action="store_true", help="experimental: switch to each language's native voice "
                   "(default: one English voice, alba, says everything and the respelling is shown on screen)")
    p.add_argument("--language", metavar="CODE", help="with --native-voices: the starting language (es, fr, de, it or pt). "
                   "With --shadow, it picks the voice that gives feedback.")
    p.add_argument("--inline", action="store_true", help="send config inline instead of binding ASSEMBLYAI_AGENT_ID")
    p.add_argument("--shadow", metavar="PHRASE", help="shadow mode: read PHRASE aloud and watch it light up")
    p.add_argument("--report-dir", default="results", help="where to write the session report JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="also show log lines in the terminal (they are always saved to the report dir)")
    return p


def apply_language(settings: Settings, code: str | None) -> Settings:
    """--language picks the course language and its native voice; ConfigError if it can't be spoken."""
    if not code:
        return settings
    lang = languages.resolve(code)
    if lang is None:
        raise ConfigError(f"unknown language {code!r}; choose from {', '.join(languages.spoken_names())}")
    if lang.voice is None:
        raise ConfigError(f"{lang.name} can't be spoken by the Voice Agent API yet; choose from {', '.join(languages.spoken_names())}")
    return replace(settings, voice_id=lang.voice, course_language=lang.name)


def make_update(settings: Settings, args: argparse.Namespace) -> dict:
    if args.shadow:  # shadow mode has its own prompt, so it is always inline
        return protocol.build_session_update_inline(agent_spec.build_shadow_config(settings, args.shadow))
    if settings.agent_id and not args.inline:
        return protocol.build_session_update_for_agent(settings.agent_id)
    return protocol.build_session_update_inline(agent_spec.build_session_config(settings, args.prompt_version))


def can_switch_voice(settings: Settings, args: argparse.Namespace) -> bool:
    """Inline sessions can start a new session with another language's voice. A saved agent
    has one voice fixed server-side, and shadow mode never speaks the target language."""
    return args.native_voices and not args.shadow and (args.inline or not settings.agent_id)


def run_paths(directory: str, stamp: int) -> tuple[Path, Path, Path]:
    base = Path(directory)
    return base / f"session-{stamp}.json", base / f"session-{stamp}.log", base / f"session-{stamp}.events.jsonl"


def setup_logging(log_path: Path, verbose: bool) -> list[logging.Handler]:
    """Everything goes to a file in results/. The terminal only gets errors, or everything with -v.
    The websockets logger is capped at INFO because its DEBUG output dumps every audio frame."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.ERROR)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    logging.getLogger("websockets").setLevel(logging.INFO)
    handlers: list[logging.Handler] = [file_handler, console]
    for handler in handlers:
        root.addHandler(handler)
    return handlers


def teardown_logging(handlers: list[logging.Handler]) -> None:
    for handler in handlers:
        logging.getLogger().removeHandler(handler)
        handler.close()


def event_record(ev: protocol.Event, t: float) -> dict:
    """One server event as a JSON-safe record. Audio payloads are replaced by their size, and
    learner transcripts are redacted, so the log is as safe to keep as the session report."""
    rec = {"t": round(t, 3), **{k: v for k, v in ev.data.items() if k not in ("data", "audio")}}
    rec["type"] = ev.type
    for key in ("data", "audio"):
        if isinstance(ev.data.get(key), str):
            rec["audio_b64_chars"] = len(ev.data[key])
    if ev.type in ("transcript.user", "transcript.user.delta") and "text" in rec:
        rec["text"] = guardrails.check_transcript(str(rec["text"])).redacted
    return rec


def save_report(report: SessionReport, directory: str, stamp: int | None = None) -> Path:
    path = run_paths(directory, stamp if stamp is not None else int(time.time()))[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: the report must stay valid JSON for any parser (jq, JavaScript).
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str, allow_nan=False),
                    encoding="utf-8")
    return path


def print_summary(report: SessionReport, path: Path, extra: tuple[Path, ...] = ()) -> None:
    summary = {
        "turns": len(report.turns),
        "language_switches": report.language_switches,
        "tool_calls": [{k: c[k] for k in ("name", "status", "reason")} for c in report.tool_calls],
        "guard_events": report.guard_events,
        "reconnects": report.reconnects,
        "metrics": report.metrics,
    }
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"full report: {path}")
    for other in extra:
        print(f"{'event log' if other.name.endswith('.jsonl') else 'debug log'}: {other}")


async def run(settings: Settings, args: argparse.Namespace) -> int:
    from .audio_io import Speaker, mic_chunks  # imported late: optional dependency

    try:
        speaker = Speaker(settings.sample_rate)
    except Exception as exc:  # no audio extra, no PortAudio, or no output device
        print(f"audio error: {exc}\nInstall the audio extra (pip install -e \".[audio]\") and check your sound devices.",
              file=sys.stderr)
        return 3
    stamp = int(time.time())
    _, log_path, events_path = run_paths(args.report_dir, stamp)
    handlers = setup_logging(log_path, args.verbose)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_fh = events_path.open("w", encoding="utf-8")
    started = time.monotonic()
    state = {"settings": settings}  # the voice (and so the settings) can change mid-call
    switching = can_switch_voice(settings, args)

    def write_event(rec: dict) -> None:
        events_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        events_fh.flush()

    def on_event(ev: protocol.Event) -> None:
        write_event(event_record(ev, time.monotonic() - started))
        text = protocol.text_of(ev)
        if args.shadow and ev.type in ("transcript.user.delta", "transcript.user"):
            print("\r\033[K" + shadow.render_ansi(shadow.align(args.shadow, text)), end="", flush=True)
        elif ev.type == "transcript.user":
            print(f"\nYou:   {guardrails.check_transcript(text).redacted}")  # never echo raw PII
        elif ev.type == "transcript.agent":
            print(f"\nLingo: {text}")

    def teach(a: dict) -> dict:
        result = tutor.teach_phrase(
            a, profanity_policy=state["settings"].profanity_policy,
            course_language=state["settings"].course_language, auto_switch=switching)
        if result.get("status") == "approved":  # the respelling on screen backs up the voice
            print(f"\n   {result['phrase']}   ->   {result['pronunciation']}")
        return result

    def switch_update(code: str) -> dict:
        lang = languages.resolve(code)
        assert lang is not None and lang.voice  # the tool only asks for languages that have a voice
        state["settings"] = replace(state["settings"], voice_id=lang.voice, course_language=lang.name)
        print(f"\n[switching to the {lang.name} voice]")
        write_event({"t": round(time.monotonic() - started, 3), "type": "client.language_switch",
                     "language": lang.name, "voice": lang.voice})
        config = agent_spec.build_session_config(state["settings"], args.prompt_version, greeting=None)
        return protocol.build_session_update_inline(config)

    session = VoiceSession(
        settings,
        session_update=make_update(settings, args),
        tools={} if args.shadow else {
            tutor.TOOL_NAME: teach,
            tutor.CHECK_TOOL_NAME: coaching.check_attempt,
        },
        audio_source=mic_chunks(settings.sample_rate),
        audio_sink=speaker.play,
        on_interrupt=speaker.flush,
        on_event=on_event,
        switch_update=switch_update if switching else None,
    )
    print(f"Listening. Ctrl+C to end. Logs are saved under {args.report_dir}/ (session-{stamp}.*)")
    code = 0
    try:
        report = await session.run()
    except asyncio.CancelledError:  # Ctrl+C: still show what happened
        report = session.snapshot()
    except SessionError as exc:
        print(f"session failed: {exc}", file=sys.stderr)
        report, code = session.snapshot(), 1
    finally:
        speaker.close()
        events_fh.close()
        teardown_logging(handlers)
    print_summary(report, save_report(report, args.report_dir, stamp), (events_path, log_path))
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # a legacy Windows code page must not crash on "¿" or "ñ"
            stream.reconfigure(errors="replace")
    if args.shadow and os.name == "nt":  # pragma: no cover
        os.system("")  # turns on ANSI colours in the classic Windows console
    load_dotenv()
    try:
        if args.language and not (args.native_voices or args.shadow):
            raise ConfigError("--language only applies with --native-voices (or --shadow)")
        settings = apply_language(Settings.from_env(), args.language)
        make_update(settings, args)  # fail on a bad config before opening audio devices
    except (ConfigError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(run(settings, args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
