import json
import math
import os

import pytest

from lingua_agent import agent_spec, guardrails, languages, prompts, protocol, shadow, tutor
from lingua_agent.audio_io import PlaybackBuffer
from lingua_agent.config import ConfigError, Settings, load_dotenv
from lingua_agent.metrics import LatencyTracker, percentile
from lingua_agent.pcm import paced_pcm


# ---------- config
def test_settings_requires_key_and_hides_it_from_repr():
    with pytest.raises(ConfigError):
        Settings.from_env({})
    with pytest.raises(ConfigError):
        Settings.from_env({"ASSEMBLYAI_API_KEY": "your_assemblyai_api_key_here"})
    s = Settings.from_env({"ASSEMBLYAI_API_KEY": "sk-secret-123", "ASSEMBLYAI_AGENT_ID": "a1"})
    assert "sk-secret-123" not in repr(s) and s.agent_id == "a1"


def test_settings_rejects_bad_policy():
    with pytest.raises(ConfigError, match="PROFANITY_POLICY"):
        Settings.from_env({"ASSEMBLYAI_API_KEY": "k", "PROFANITY_POLICY": "yolo"})


@pytest.mark.parametrize("var, value", [("ASSEMBLYAI_WS_URL", "ws://agents.example.com/v1/ws"),
                                        ("ASSEMBLYAI_REST_URL", "http://agents.example.com")])
def test_settings_refuses_to_send_the_key_unencrypted(var, value):
    with pytest.raises(ConfigError, match=var):
        Settings.from_env({"ASSEMBLYAI_API_KEY": "k", var: value})


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nFOO_TEST_A=from_file\nFOO_TEST_B='quoted'\nbadline\n")
    monkeypatch.setenv("FOO_TEST_A", "from_env")
    monkeypatch.delenv("FOO_TEST_B", raising=False)
    load_dotenv(str(env))
    assert os.environ["FOO_TEST_A"] == "from_env" and os.environ["FOO_TEST_B"] == "quoted"
    monkeypatch.delenv("FOO_TEST_B")
    load_dotenv(str(tmp_path / "missing.env"))  # no file: no error


# ---------- protocol
def test_parse_event_and_errors():
    assert protocol.parse_event('{"type":"reply.done"}').type == "reply.done"
    for bad in ("not json", "[]", '{"no":"type"}', b"\xff\xfe"):
        with pytest.raises(protocol.ProtocolError):
            protocol.parse_event(bad)


@pytest.mark.parametrize("payload, id_field", [
    ({"type": "tool.call", "call_id": "1", "name": "t", "arguments": {"a": 1}}, "call_id"),
    ({"type": "tool.call", "tool_call_id": "1", "name": "t", "arguments": '{"a": 1}'}, "tool_call_id"),
])
def test_parse_tool_call_accepts_both_documented_spellings(payload, id_field):
    call = protocol.parse_tool_call(protocol.Event("tool.call", payload))
    assert (call.call_id, call.name, call.arguments, call.id_field) == ("1", "t", {"a": 1}, id_field)


@pytest.mark.parametrize("payload", [
    {"name": "t"}, {"call_id": "1"}, {"call_id": "1", "name": "t", "arguments": "{bad"},
    {"call_id": "1", "name": "t", "arguments": [1]},
])
def test_parse_tool_call_rejects_malformed(payload):
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_tool_call(protocol.Event("tool.call", payload))


def test_empty_string_arguments_become_empty_dict():
    ev = protocol.Event("tool.call", {"call_id": "1", "name": "t", "arguments": "  "})
    assert protocol.parse_tool_call(ev).arguments == {}


def test_audio_field_names_match_the_docs():
    msg = protocol.build_input_audio(b"\x00\x01")
    assert set(msg) == {"type", "audio"}  # client -> server uses "audio"
    assert protocol.decode_audio(protocol.Event("reply.audio", {"data": msg["audio"]})) == b"\x00\x01"  # server uses "data"
    for bad in ({}, {"data": "!!!"}):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode_audio(protocol.Event("reply.audio", bad))


def test_builders_and_event_helpers():
    assert protocol.build_session_update_for_agent("a")["session"] == {"agent_id": "a"}
    assert protocol.build_session_resume("s") == {"type": "session.resume", "session_id": "s"}
    assert protocol.build_session_end() == {"type": "session.end"}
    with pytest.raises(protocol.ProtocolError):
        protocol.build_session_update_inline({"agent_id": "a", "system_prompt": "x"})
    call = protocol.ToolCall("c", "t", {})
    res = protocol.build_tool_result(call, {"phrase": "¿Dónde?"})
    assert res["call_id"] == "c" and isinstance(res["result"], str)
    assert json.loads(res["result"])["phrase"] == "¿Dónde?" and "\\u" not in res["result"]
    assert protocol.build_tool_result(call, {}, is_error=True)["is_error"] is True

    ev = protocol.Event
    assert protocol.text_of(ev("transcript.user", {"text": "hi"})) == "hi"
    assert protocol.text_of(ev("transcript.user", {"transcript": "hi"})) == "hi"
    assert protocol.text_of(ev("transcript.user", {"text": None})) == ""
    assert protocol.is_interrupted(ev("reply.done", {"status": "interrupted"}))
    assert protocol.is_interrupted(ev("transcript.agent", {"interrupted": True}))
    assert not protocol.is_interrupted(ev("reply.done", {"status": "completed"}))
    assert protocol.error_code(ev("session.error", {"code": "UNAUTHORIZED"})) == "unauthorized"
    assert protocol.is_error(ev("error", {})) and not protocol.is_error(ev("reply.done", {}))


# ---------- languages / tutor
def test_language_resolution_and_scripts():
    assert languages.resolve("ES").code == "es" and languages.resolve("mandarin").code == "zh"
    assert languages.resolve("Klingon") is None and languages.resolve(5) is None
    ja = languages.resolve("ja")
    assert languages.script_matches(ja, "コーヒー") and not languages.script_matches(ja, "kohi")
    assert not languages.script_matches(ja, "123")
    assert "Japanese" in languages.supported_names()


def test_normalize_folds_latin_accents_only():
    assert languages.normalize_text("¿Dónde ESTÁ?") == "donde esta"
    assert languages.normalize_text("Ｈｅｌｌｏ") == "hello"  # full-width forms
    # Voiced and unvoiced kana are different sounds and must stay different.
    assert languages.normalize_text("が") != languages.normalize_text("か")
    assert languages.normalize_text("안녕하세요") == "안녕하세요"  # Hangul syllables stay composed


def test_tool_definition_shapes():
    saved, inline = tutor.tool_definition(), tutor.tool_definition(inline=True)
    assert saved["parameters"]["properties"]["target_language"]["enum"] == languages.spoken_names()
    assert "http" not in saved and saved["execution_mode"] == "interactive"
    assert set(inline) == {"type", "name", "description", "parameters"} and inline["type"] == "function"


def test_teach_phrase_approved_payload_shape():
    out = tutor.teach_phrase({"target_language": "French", "source_text": "Hello", "phrase": "Bonjour",
                              "pronunciation": "bohn-ZHOOR", "register": "weird"})
    assert out["syllables"] == ["bohn", "ZHOOR"] and out["register"] == "neutral" and out["language_code"] == "fr"


def test_vulgar_label_adds_note():
    out = tutor.teach_phrase({"target_language": "German", "source_text": "damn it", "phrase": "Verdammt",
                              "pronunciation": "fehr-DAHMT"})
    assert out["register"] == "vulgar" and "note" in out


# ---------- guardrails
def test_redaction_never_leaks_raw_pii():
    r = guardrails.check_transcript("mail jo@example.com or 555-123-4567, ssn 123-45-6789, card 4111 1111 1111 1111")
    assert not any(x in r.redacted for x in ("jo@example.com", "555-123", "123-45", "4111"))
    assert set(r.flags) == {"pii_email", "pii_phone", "pii_ssn", "pii_card"} and r.allowed


# ---------- prompts / spec
def test_keyterms_respect_api_limits_and_prompt_versions():
    assert agent_spec.validate_keyterms(prompts.KEYTERMS)
    with pytest.raises(ValueError):
        agent_spec.validate_keyterms(["x"] * 101)
    with pytest.raises(ValueError):
        agent_spec.validate_keyterms(["x" * 51])
    assert len(prompts.TRANSCRIPTION_PROMPT) <= agent_spec.MAX_TRANSCRIPTION_PROMPT_CHARS
    for version in ("v2", "v3", "v4", "v5"):
        assert guardrails.PROMPT_CANARY in prompts.get_prompt(version)
    assert "One moment" in prompts.get_prompt("v3") and "One moment" not in prompts.get_prompt("v2")
    with pytest.raises(KeyError):
        prompts.get_prompt("v99")


def test_inline_session_config_uses_the_inline_shapes():
    cfg = agent_spec.build_session_config(Settings(api_key="k"))
    assert cfg["output"]["voice"] == "alba" and "voice" not in cfg  # inline voice lives under output
    assert cfg["tools"][0]["type"] == "function"
    assert cfg["input"]["turn_detection"]["interrupt_response"] is True
    assert cfg["input"]["transcription_prompt"] and cfg["input"]["keyterms"]
    assert "greeting" in cfg and "name" not in cfg
    assert "greeting" not in agent_spec.build_session_config(Settings(api_key="k"), greeting=None)


def test_saved_agent_request_uses_the_rest_shapes():
    body = agent_spec.build_agent_request(Settings(api_key="k"))
    assert {"name", "system_prompt", "voice", "greeting", "input", "output", "tools"} <= set(body)
    assert body["voice"] == {"voice_id": "alba"}
    assert body["tools"][0]["name"] == "teach_phrase" and "type" not in body["tools"][0]
    assert "transcription_prompt" not in body["input"]  # not in the documented REST schema


def test_shadow_config_never_boosts_the_phrase_under_test():
    cfg = agent_spec.build_shadow_config(Settings(api_key="k"), "  ¿Dónde   está la biblioteca? ")
    assert "keyterms" not in cfg["input"] and "tools" not in cfg
    assert "¿Dónde está la biblioteca?" in cfg["system_prompt"]
    with pytest.raises(ValueError):
        agent_spec.build_shadow_config(Settings(api_key="k"), "   ")


# ---------- metrics
def test_percentile_edges():
    assert math.isnan(percentile([], 50)) and percentile([7], 95) == 7
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([1, 2, 3, 4], 100) == 4


def test_ttfa_measured_once_per_turn_from_speech_stop():
    t = iter([10.0, 10.5, 20.0, 20.25])
    tr = LatencyTracker(clock=lambda: next(t))
    tr.speech_stopped()   # 10.0
    tr.reply_started()
    tr.audio_chunk()      # 10.5 -> 500 ms
    tr.audio_chunk()      # same reply: ignored
    tr.reply_done()
    assert tr.ttfa_ms == [500.0]
    tr.speech_started(); tr.speech_stopped()   # 20.0
    tr.audio_chunk()      # 20.25 (no reply.started seen: still counted, the agent was idle)
    assert tr.ttfa_ms == [500.0, 250.0]
    s = tr.summary()
    assert s["turns_measured"] == 2 and s["ttfa_p95_ms"] > s["ttfa_p50_ms"]


def test_ttfa_ignores_audio_from_the_reply_being_interrupted():
    t = iter([1.0, 2.0])
    tr = LatencyTracker(clock=lambda: next(t))
    tr.reply_started()                     # agent is talking
    tr.speech_started(); tr.speech_stopped()   # 1.0: learner talks over it
    tr.audio_chunk()                       # tail of the OLD reply: must not count
    assert tr.ttfa_ms == []
    tr.reply_done(interrupted=True)
    tr.reply_started()
    tr.audio_chunk()                       # 2.0: first audio of the new reply
    assert tr.ttfa_ms == [1000.0] and tr.interruptions == 1


def test_resumed_speech_voids_the_earlier_stop():
    tr = LatencyTracker(clock=lambda: 5.0)
    tr.speech_stopped(); tr.speech_started()
    tr.audio_chunk()
    assert tr.ttfa_ms == []


# ---------- playback buffer
def test_playback_buffer_pads_with_silence_and_clears():
    buf = PlaybackBuffer()
    buf.write(b"\x01\x02\x03")
    assert buf.read(2) == b"\x01\x02" and buf.read(4) == b"\x03\x00\x00\x00" and len(buf) == 0
    buf.write(b"stale speech")
    buf.clear()
    assert buf.read(2) == b"\x00\x00"


# ---------- recorded-clip streaming
async def test_paced_pcm_marks_end_of_speech_and_appends_silence(tmp_path):
    pcm = tmp_path / "q.pcm"
    pcm.write_bytes(b"\x01" * 1920 * 3)  # three 40 ms chunks at 24 kHz
    marks = {}
    out = [c async for c in paced_pcm(pcm, 24000, realtime=False, tail_ms=80, marks=marks)]
    assert len(out) == 5 and out[2][:1] == b"\x01" and out[-1] == b"\x00" * 1920 and "speech_end" in marks

    seen = []
    async for _chunk in paced_pcm(pcm, 24000, realtime=False, tail_ms=0, speech_end_ms=50, marks=(m := {})):
        seen.append("speech_end" in m)
    assert seen == [False, False, True]  # 50 ms of speech ends inside the second chunk


# ---------- shadow mode
def test_shadow_alignment_states_and_score():
    a = shadow.align("¿Dónde está la biblioteca?", "donde esta biblioteca por favor")
    states = {w.word: w.state for w in a.words}
    assert states == {"donde": "match", "esta": "match", "la": "missing", "biblioteca": "match"}
    assert a.extra_words == 2 and a.score == 0.75


def test_shadow_wrong_word_and_cjk_and_empty():
    assert [w.state for w in shadow.align("buenos dias", "buenas dias").words] == ["wrong", "match"]
    assert shadow.align("ありがとう", "ありがとう").score == 1.0
    assert shadow.align("ありがとう", "ありかとう").score == 0.8  # one kana off, not a full match
    assert shadow.align("안녕 하세요", "안녕하세요").score == 1.0  # spacing differences are not errors
    assert shadow.align("", "anything").score == 0.0
    assert "\033[32m" in shadow.render_ansi(shadow.align("hola", "hola"))


# ---------- review of two live sessions
def test_clean_respelling_fixes_punctuation_but_not_wrong_formats():
    assert tutor.clean_respelling("DOHN-deh ehs-TAH el BAHN-yo?") == "DOHN-deh ehs-TAH el BAHN-yo"
    assert tutor.clean_respelling("  gah - YEH - tahs. ") == "gah-YEH-tahs"
    assert tutor.clean_respelling("bohn—ZHOOR") == "bohn-ZHOOR"
    for bad in ("/ˈdon.de/", "do2nde", "—"):  # IPA, digits, nothing left after cleaning: the model misread the format
        out = tutor.teach_phrase({"target_language": "Spanish", "source_text": "Where is the library?",
                                  "phrase": "¿Dónde está la biblioteca?", "pronunciation": bad})
        assert out["reason"] == "bad_pronunciation_format", bad


def test_trailing_question_mark_no_longer_costs_a_retry():
    out = tutor.teach_phrase({"target_language": "Spanish", "source_text": "Where is the bathroom?",
                              "phrase": "¿Dónde está el baño?", "pronunciation": "DOHN-deh ehs-TAH el BAHN-yo?"})
    assert out["status"] == "approved" and out["pronunciation"] == "DOHN-deh ehs-TAH el BAHN-yo"


def test_breakdown_is_built_in_code_and_has_one_shape():
    one = tutor.teach_phrase({"target_language": "Spanish", "source_text": "cookies", "phrase": "galletas",
                              "pronunciation": "gah-YEH-tahs"})
    many = tutor.teach_phrase({"target_language": "Spanish", "source_text": "Where is the bathroom?",
                               "phrase": "¿Dónde está el baño?", "pronunciation": "DOHN-deh ehs-TAH el BAHN-yo"})
    assert one["sounds_like"] == "gah-YEH-tahs" and one["slow"] == "gah, YEH, tahs"
    assert many["sounds_like"] == "DOHN-deh, ehs-TAH, el, BAHN-yo"  # words paused, syllables hyphenated
    assert many["slow"] == "DOHN, deh, ehs, TAH, el, BAHN, yo"
    assert "how_to_say" in one and one["how_to_say"] == many["how_to_say"]


def test_v4_is_the_default_and_fixes_what_the_live_sessions_showed():
    text = prompts.get_prompt()
    assert prompts.DEFAULT_PROMPT_VERSION == "v5"
    assert "sounds_like" in text and "never that the pronunciation or accent was perfect" in text
    assert "the tool does" in text  # profanity is decided by the tool, not pre-empted by the prompt
    assert "perfect" not in prompts.GREETING and len(prompts.GREETING.split()) < 15


def test_turn_detection_default_and_override():
    assert agent_spec.build_session_config(Settings(api_key="k"))["input"]["turn_detection"]["min_silence"] == 700
    td = agent_spec.build_session_config(Settings(api_key="k", min_silence_ms=1200))["input"]["turn_detection"]
    assert td["min_silence"] == 1200 and td["max_silence"] >= 1200
    assert Settings.from_env({"ASSEMBLYAI_API_KEY": "k", "TURN_MIN_SILENCE_MS": "500"}).min_silence_ms == 500
    for bad in ("fast", "50", "99999"):
        with pytest.raises(ConfigError):
            Settings.from_env({"ASSEMBLYAI_API_KEY": "k", "TURN_MIN_SILENCE_MS": bad})


def test_post_tool_latency_is_separate_from_first_audio():
    t = iter([0.0, 1.0, 1.2, 1.5, 2.0])  # stopped, filler audio, tool result sent, answer audio
    tr = LatencyTracker(clock=lambda: next(t))
    tr.speech_stopped(); tr.reply_started(); tr.audio_chunk()   # filler at 1.0 s
    tr.tool_results_sent(); tr.audio_chunk()                    # real answer audio at 1.5 s
    assert tr.ttfa_ms == [1000.0] and tr.post_tool_ms == pytest.approx([300.0]) and tr.answer_ms == [1500.0]
    s = tr.summary()
    assert s["tool_turns"] == 1 and s["answer_p50_ms"] == 1500.0 and s["post_tool_p50_ms"] == pytest.approx(300.0)


# ---------- live finding: wrong attempts, English voice, languages the API cannot speak
def test_v5_treats_the_turn_after_a_lesson_as_an_attempt():
    text = prompts.get_prompt()
    assert "whatever the learner says next is an attempt" in text and "check_attempt" in text
    assert "never answer one with" in text and "Korean" in text  # names what it can't speak, doesn't promise it


def test_only_languages_the_api_can_speak_are_offered_by_voice():
    assert languages.spoken_names() == ["Spanish", "French", "German", "Italian", "Portuguese"]
    assert {lang.code: lang.voice for lang in languages.spoken_languages()} == {
        "es": "lola", "fr": "estelle", "de": "juergen", "it": "giovanni", "pt": "rafael"}
    assert languages.ENGLISH_VOICES >= {"alba", "jane"}
    for name in ("Japanese", "Korean", "Mandarin Chinese"):
        out = tutor.teach_phrase({"target_language": name, "source_text": "Thank you",
                                  "phrase": {"Japanese": "ありがとう", "Korean": "감사합니다", "Mandarin Chinese": "谢谢"}[name],
                                  "pronunciation": "ah-ree-GAH-toh"})
        assert out["reason"] == "language_not_spoken_yet" and "Spanish" in out["speak_instead"]
    assert not any(k in prompts.TRANSCRIPTION_PROMPT for k in ("Japanese", "Korean", "Mandarin"))


FR = {"target_language": "French", "source_text": "Hello", "phrase": "Bonjour", "pronunciation": "bohn-ZHOOR"}


def test_pinned_session_rejects_other_languages_and_prompt_names_the_voice():
    out = tutor.teach_phrase(FR, course_language="Spanish")
    assert out["reason"] == "other_course_language" and "Spanish" in out["speak_instead"]
    assert "You are the Spanish voice" in prompts.get_prompt(course_language="Spanish")
    assert "You are the English voice" in prompts.get_prompt()
    assert "voice" not in prompts.get_prompt(can_switch=False).split("SAFETY")[-1]  # saved agents: no switching text


def test_a_request_for_another_language_asks_for_a_voice_switch():
    out = tutor.teach_phrase(FR, course_language="Spanish", auto_switch=True)
    assert out == {"status": "switch_language", "language_code": "fr", "language": "French"}
    assert tutor.teach_phrase(FR, course_language=None, auto_switch=True)["status"] == "switch_language"  # from the English voice
    assert tutor.teach_phrase(FR, course_language="French", auto_switch=True)["status"] == "approved"  # already the right voice
    bad = dict(FR, pronunciation="bohn2")
    assert tutor.teach_phrase(bad, auto_switch=True)["reason"] == "bad_pronunciation_format"  # a bad lesson never switches
    ja = dict(FR, target_language="Japanese", phrase="こんにちは")
    assert tutor.teach_phrase(ja, auto_switch=True)["reason"] == "language_not_spoken_yet"  # no voice to switch to


def test_conversation_message_and_reply_create_builders():
    assert protocol.build_conversation_message("user", "hi") == {"type": "conversation.message", "role": "user", "content": "hi"}
    assert protocol.build_reply_create() == {"type": "reply.create"}
    assert protocol.build_reply_create("go")["instructions"] == "go"


def test_language_flag_picks_the_native_voice_or_fails_clearly():
    from lingua_agent.cli import apply_language
    s = apply_language(Settings(api_key="k"), "es")
    assert s.voice_id == "lola" and s.course_language == "Spanish"
    cfg = agent_spec.build_session_config(s)
    assert cfg["output"]["voice"] == "lola" and "Spanish voice" in cfg["system_prompt"]
    assert apply_language(Settings(api_key="k"), None).voice_id == "alba"
    for bad in ("ja", "ko", "klingon"):
        with pytest.raises(ConfigError):
            apply_language(Settings(api_key="k"), bad)


def test_both_tools_are_registered_in_both_shapes():
    assert [t["name"] for t in tutor.tool_definitions(inline=True)] == ["teach_phrase", "check_attempt"]
    assert all(t["type"] == "function" for t in tutor.tool_definitions(inline=True))
    assert [t["name"] for t in tutor.tool_definitions()] == ["teach_phrase", "check_attempt"]
    assert [t["name"] for t in agent_spec.build_session_config(Settings(api_key="k"))["tools"]] == ["teach_phrase", "check_attempt"]


def test_check_attempt_clear_close_and_unclear():
    from lingua_agent import coaching
    clear = coaching.check_attempt({"target_language": "Spanish", "target_phrase": "galletas", "heard": "Galletas."})
    assert clear["verdict"] == "clear" and "focus_word" not in clear
    close = coaching.check_attempt({"target_language": "Spanish", "target_phrase": "¿Dónde está el baño?",
                                    "heard": "donde esta el bano"})
    assert close["verdict"] == "clear"  # accents and punctuation are ignored: intelligibility, not spelling
    miss = coaching.check_attempt({"target_language": "Spanish", "target_phrase": "galletas", "heard": "gal etas"})
    assert miss["verdict"] == "unclear" and miss["focus_word"] == "galletas" and "y in yes" in miss["tip"]
    half = coaching.check_attempt({"target_language": "Spanish", "target_phrase": "¿Dónde está el baño?", "heard": "donde esta el bar"})
    assert half["verdict"] == "close" and half["focus_word"] == "baño" and "canyon" in half["tip"]
    assert "Never say the pronunciation or accent was perfect" in half["how_to_say"]


def test_check_attempt_edge_cases():
    from lingua_agent import coaching
    assert coaching.check_attempt({"heard": "hola"})["reason"] == "missing_fields"
    none = coaching.check_attempt({"target_phrase": "hola", "heard": "  "})
    assert none["verdict"] == "no_attempt" and none["score"] == 0.0
    fr = coaching.check_attempt({"target_language": "French", "target_phrase": "Bonjour", "heard": "bonjoy"})
    assert fr["focus_word"] == "Bonjour" and fr["tip"]  # r gets the throat hint
    assert "tip" not in coaching.check_attempt({"target_language": "Klingon", "target_phrase": "qapla", "heard": "kapla"})
    assert coaching.tip_for("es", "casa") is None
    assert "ee with your lips rounded" in coaching.tip_for("de", "über")
    assert coaching.tip_for("de", "danke") is None  # no tip unless a pattern applies
    assert "nasal" in coaching.tip_for("pt", "não") and "double" in coaching.tip_for("it", "pizza")


def test_cli_warns_only_when_a_saved_agent_cannot_switch_voices(monkeypatch, capsys):
    from lingua_agent import cli
    seen = {}

    async def fake_run(settings, args):
        seen["voice"] = settings.voice_id
        return 0

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k")
    monkeypatch.delenv("VOICE_ID", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_AGENT_ID", raising=False)
    assert cli.main([]) == 0 and capsys.readouterr().err == ""  # inline: Lingo switches voices itself
    assert cli.main(["--language", "fr"]) == 0 and seen["voice"] == "estelle"
    assert cli.main(["--language", "ko"]) == 2
    monkeypatch.setenv("ASSEMBLYAI_AGENT_ID", "agent-1")
    assert cli.main([]) == 0 and "fixed voice" in capsys.readouterr().err
    assert cli.main(["--inline"]) == 0 and capsys.readouterr().err == ""


# ---------- final review
def test_metrics_summary_is_valid_json_even_with_nothing_measured():
    summary = LatencyTracker().summary()
    assert summary["ttfa_p50_ms"] is None and summary["answer_p50_ms"] is None and summary["switch_p50_ms"] is None
    json.dumps(summary, allow_nan=False)  # NaN would raise: it is not JSON


def test_dotenv_saved_by_notepad_with_a_bom_still_loads(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_bytes(b"\xef\xbb\xbfASSEMBLYAI_API_KEY=abc123\n")
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    load_dotenv(str(env))
    assert os.environ["ASSEMBLYAI_API_KEY"] == "abc123"


def test_unspeakable_languages_are_refused_before_any_format_check():
    romaji = {"target_language": "Japanese", "source_text": "Thank you", "phrase": "arigatou", "pronunciation": "bad2"}
    out = tutor.teach_phrase(romaji)
    assert out["reason"] == "language_not_spoken_yet"  # no wasted retry on script or respelling
    unknown = tutor.teach_phrase(dict(FR, target_language="Klingon"))
    assert unknown["reason"] == "unsupported_language"
    assert "Japanese" not in unknown["speak_instead"] and "Spanish" in unknown["speak_instead"]


@pytest.mark.parametrize("text,flagged", [
    ("Great job! Your pronunciation was perfect.", True),
    ("Perfect! Your pronunciation was spot on.", True),
    ("Your accent is flawless.", True),
    ("You said it perfectly.", True),
    ("That came through clearly. Want another one?", False),
    ("I can't judge your accent, but every word came through.", False),
    ("The perfect tense comes later. Try galletas again.", False),
])
def test_agent_claims_about_pronunciation_quality_are_flagged(text, flagged):
    assert ("overclaims_pronunciation" in guardrails.check_agent_text(text).flags) is flagged


def test_fixed_voice_prompts_never_promise_a_switch():
    eval_cfg = agent_spec.build_session_config(Settings(api_key="k"), can_switch=False)
    assert "THIS SESSION" not in eval_cfg["system_prompt"]
    pinned = prompts.get_prompt(course_language="Spanish", can_switch=False)
    assert "This is a Spanish course" in pinned and "switches" not in pinned


def test_saved_agent_per_course_language_gets_the_native_voice_and_a_pinned_prompt():
    from lingua_agent.cli import apply_language
    body = agent_spec.build_agent_request(apply_language(Settings(api_key="k"), "es"))
    assert body["voice"] == {"voice_id": "lola"} and "Spanish" in body["name"]
    assert "This is a Spanish course" in body["system_prompt"]
    assert "all languages" in agent_spec.build_agent_request(Settings(api_key="k"))["name"]


def test_shadow_prompt_does_not_praise_pronunciation():
    text = prompts.shadow_prompt("hola")
    assert "praise" not in text and "never the sound" in text


def test_script_checks_for_languages_waiting_on_a_voice_still_work():
    ja, ko, zh = (languages.resolve(c) for c in ("ja", "ko", "zh"))
    assert languages.script_matches(ja, "ありがとう") and not languages.script_matches(ja, "arigatou")
    assert languages.script_matches(ko, "안녕하세요") and not languages.script_matches(ko, "annyeong")
    assert languages.script_matches(zh, "谢谢") and not languages.script_matches(zh, "ありがとう")
    assert not languages.script_matches(ja, "123") and all(lang.voice is None for lang in (ja, ko, zh))
