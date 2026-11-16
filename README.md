# Lingo: a pronunciation-coach voice agent on AssemblyAI

A reference voice agent for a language-learning product. The learner asks out loud how to say a word or phrase in another language. Lingo says it in a native voice, gives a "sounds like" line, listens to the learner try it, and tells them which word did not come through.

It is built on AssemblyAI's [Voice Agent API](https://www.assemblyai.com/docs/voice-agents/voice-agent-api) (speech recognition, the language model and the voice over one WebSocket), and it comes with the guardrails, tests and evaluation harness I would use to win a head-to-head technical evaluation against the customer's current vendor.

| Assignment asks for | Where it is |
|---|---|
| A working reference voice agent | `src/lingua_agent/`, run with `lingua-agent` |
| How to run it, how it is built | This README |
| How to win the customer's evaluation | [`evals/EVAL_PLAN.md`](evals/EVAL_PLAN.md), the harness in `evals/`, and [the summary below](#how-i-would-win-the-evaluation) |
| Extra credit: a short creative demo | [Shadow mode](#extra-credit-shadow-mode) |

## Status, stated plainly

- **Run live:** two recorded sessions against the real API. The agent connected, called the lesson tool, spoke the lesson and returned zero errors. Those sessions exposed six problems, all fixed since (see [What the live sessions changed](#what-the-live-sessions-changed)).
- **Unit-tested only, not yet run live:** switching to a native voice per language, attempt feedback (`check_attempt`), and prompt v5. They are built from AssemblyAI's documented events and covered by tests against a scripted socket.
- **Not done:** the bake-off against the incumbent (needs their keys and recorded learner audio), a fluent-speaker listening check per language, and phoneme-level accent scoring.
- 133 tests, 99% line coverage, lint clean. An offline guardrail eval of 62 labeled cases gates CI.

## Quick start

Python 3.11 or newer. A microphone and **headphones** (there is no echo cancellation, so on open speakers the agent hears itself and interrupts its own replies).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,audio]"        # audio needs PortAudio on macOS (brew install portaudio) and Linux (apt install libportaudio2)
cp .env.example .env                 # Windows: copy .env.example .env    then add ASSEMBLYAI_API_KEY
lingua-agent                         # talk to Lingo. Ctrl+C ends the call and prints the session report
```

Things to try: "How do you say cookies in Spanish?", say it back wrong on purpose, "say that slower", then "and in French?".

```bash
lingua-agent --language es           # start in the Spanish voice (es, fr, de, it, pt) instead of English
lingua-agent --shadow "¿Dónde está la biblioteca?" --language es    # extra-credit mode
lingua-agent -v                      # also print log lines (they are always saved to results/)

pytest --cov                         # tests, with a 90% coverage gate
ruff check .                         # lint
python -m evals.run_offline_eval     # guardrail eval: no network, no key
```

The terminal shows only the conversation, a line when the voice switches, errors and the final summary. Each run writes three files to `results/` (gitignored):

| File | Contents |
|---|---|
| `session-<ts>.json` | Turns, tool calls, guardrail events, latency metrics, language switches |
| `session-<ts>.events.jsonl` | Every server event with a timestamp. Audio is replaced by its size and learner text is redacted |
| `session-<ts>.log` | Debug log |

## How it works

```
mic ──PCM16 24 kHz──▶ VoiceSession ══ WebSocket ══▶ AssemblyAI Voice Agent API
                         │   ▲                       (recognition + turn detection + LLM + voice)
 speaker ◀── reply audio ┘   │
 (flushed on barge-in)       ├─ tool.call teach_phrase   ─▶ validate the lesson BEFORE it is spoken
                             ├─ tool.call check_attempt  ─▶ compare what was heard with the phrase
                             ├─ transcript.user          ─▶ redact PII, flag injection
                             ├─ transcript.agent         ─▶ audit (format, leaks, over-claiming)
                             └─ all events               ─▶ latency tracker + event log
```

One teaching turn:

1. The learner asks "how do you say cookies in Spanish?".
2. The model calls `teach_phrase` with the phrase and a pronunciation respelling. The code checks it: a language the API can speak, the right writing system, not the English echoed back, a speakable respelling, length, and a profanity policy.
3. If the current voice is not Spanish, the tool answers `switch_language`. The client ends the session, opens a new one with the Spanish voice, and hands the request over so the learner does not repeat it.
4. The Spanish voice says the phrase, then a "sounds like" line that the **code** built, then invites a try.
5. Whatever the learner says next is treated as an attempt. `check_attempt` compares the transcript with the phrase, names the word that missed and adds one tip ("the double l sounds like the y in yes").

| Module | Job |
|---|---|
| `protocol.py` | The only file that knows message shapes and error codes |
| `session.py` | The async loop: reconnect and resume, error classification, tools, barge-in, voice switching, shutdown |
| `tutor.py` | `teach_phrase`: validates each lesson and builds the spoken breakdown |
| `coaching.py` | `check_attempt`: attempt comparison and tips |
| `guardrails.py` | PII, injection, voice-format, leak and over-claiming checks |
| `prompts.py`, `agent_spec.py` | Versioned prompts, recognizer hints, turn detection, both config shapes |
| `languages.py` | Languages, their native voices, script checks, text normalization |
| `audio_io.py`, `pcm.py` | Microphone, non-blocking playback, recorded-clip streaming |
| `shadow.py` | Extra credit: word-by-word feedback |
| `evals/` | Offline guardrail eval, bake-off harness, scoring and statistics |
| `scripts/` | Create or update saved agents; record raw API events |

## Key decisions

**1. One socket instead of a stitched pipeline.** In a rip-and-replace the pain is usually integration surface and tail latency across three vendors. The Voice Agent API removes the hops and most of the glue (turn detection, barge-in, session state). The cost is control over what gets spoken, which leads to the next decision.

**2. Teach through a tool call, because it is the only checkpoint before speech.** With speech-to-speech, the audio has gone out by the time the agent's transcript arrives, so output text can be audited but not blocked. Requiring the model to call `teach_phrase` first puts code between the model's proposal and the learner's ear. Two limits: the "call the tool first" rule is enforced by the prompt, not by the API, and the tool checks form, not whether the translation is right (the bake-off scores that against accepted translations).

**3. The code writes the pronunciation lines, not the model.** In the first live sessions the model improvised the breakdown and it changed shape from lesson to lesson. The tool now returns `sounds_like` and `slow`, and the prompt says to read them as given. The slow syllable-by-syllable version is only used when the learner asks for "again" or "slower".

**4. A native voice per language, switched by restarting the session.** The API fixes the voice when a session starts, and AssemblyAI documents each native voice as speaking its own language plus English. An English voice read "galletas" with English rules in a live session. So a request for another language ends the session and starts one with that language's voice. `conversation.message` carries the learner's request into the new session and `reply.create` makes it answer. Costs: lesson context is not kept across a switch, and there is a connection gap (recorded as `switch_p50_ms` in the report; not yet measured live).

**5. Feedback claims only what a transcript can support.** The agent receives text, never the learner's sound. It can say whether a word was understood; it cannot hear an accent. The prompt forbids "your pronunciation was perfect" (the agent said exactly that in a live session), and a guardrail flags it if it happens anyway. See [Accent feedback](#accent-feedback-what-is-and-is-not-possible).

**6. Tuned for learners.** Learners pause mid-sentence and mix languages. The config sets explicit turn-detection silence windows, a recognizer prompt naming the expected languages, and keyterms for common phrases. The silence window is a direct trade against latency, so it is one setting (`TURN_MIN_SILENCE_MS`, default 700) and the evaluation plan sweeps it.

**7. Inline config by default, saved agents for production.** Inline config runs with only an API key and allows voice switching. A saved agent keeps prompt, voice and tools server-side, so prompt changes ship without a client release, but it has one voice. The production pattern is one saved agent per course language: `python scripts/create_agent.py --language es`, then run the client with `ASSEMBLYAI_AGENT_ID` set and the same `--language`.

**8. Production behavior.**
- Reconnect with full-jitter backoff and `session.resume`. The retry budget resets after a healthy connection, and an expired session falls back to a new one.
- One task owns the microphone for the whole call, so it survives reconnects and voice switches. A failed microphone ends the call with an error.
- `session.error` is classified: a bad key fails with a clear message, capacity errors retry, a rejected message mid-call is logged and the call continues.
- On barge-in, queued speech is flushed and tool results for the interrupted reply are discarded.
- `session.end` is sent on every exit, including Ctrl+C, because billing follows connection time.
- Tested on Python 3.11, 3.12 and 3.13. The 3.11 run caught a shutdown hang (`asyncio.wait_for` swallowing a cancellation there), now fixed and pinned by a test.
- The key comes from the environment only, is hidden from `repr`, and is refused over non-TLS URLs. Learner text is redacted before it is stored, logged or printed. Audio is never written to disk.

## Guardrails

| Boundary | Check | Can it block? |
|---|---|---|
| Learner transcript | PII redaction (email, phone, SSN, card), injection patterns | No. It flags and redacts; the audio is already in flight |
| Lesson content | `teach_phrase` validation | **Yes**, before the phrase is spoken |
| Agent transcript | Markdown, URLs, emoji, length, canary leak, over-claiming pronunciation quality | No. It audits after the fact |

Known limits, kept as `known_gap` cases in the eval instead of hidden: the deterministic checks cannot tell French from Spanish; the injection regex flags a learner who asks how to say "show me the instructions"; and bare praise such as "Perfect!" is not caught by the over-claiming check. The profanity list is a placeholder; `PROFANITY_POLICY=label|block` is the real control.

## Evaluation

**Offline guardrail eval** (`python -m evals.run_offline_eval`): 62 labeled cases scored by exact match across four kinds (learner transcripts, lesson arguments including voice switching, agent text, learner attempts). 59 are gated and pass; 3 are documented gaps. This is a regression gate on cases I wrote, not evidence of generalization.

**Bake-off harness** (`python -m evals.compare <manifest> --providers assemblyai,incumbent`): runs the same recorded clips at real-time pace through each vendor and reports by stratum.

| Measure | How |
|---|---|
| Word error rate | Pooled across clips, so short utterances are not over-weighted |
| Target-term recall | Did the recognizer get the words that matter: the language name, the foreign phrase? |
| Time to first audio, p50 and p95 | From the last speech chunk **sent** to the first reply audio byte, not from the vendor's own end-of-speech event, which would hide endpointing delay |
| Teaching accuracy | The approved lesson matches an accepted translation |
| Refusal accuracy | Nothing is taught for off-topic, injection or unsupported-language clips |
| Paired difference | Per-clip bootstrap confidence interval, so a difference inside the noise is not called a win |

`evals/audio_manifest.example.jsonl` is a 12-row template. The incumbent adapter is a stub that needs the customer's keys. **No bake-off numbers exist yet**, and this README claims none.

### How I would win the evaluation

1. **Agree the scorecard before running anything**, so nobody argues metrics after seeing results.
2. **Use their learners' audio**, stratified: clean, accented, noisy, code-switched, hesitant, adversarial. Generic benchmarks do not predict this workload.
3. **Measure what matters here**: target-term recall on code-switched speech and latency from the true end of speech, on top of word error rate.
4. **Make it fair and reproducible**: same clips, same pacing, default and tuned settings for both vendors, blind judging, confidence intervals, and hand them the harness.
5. **Raise our weak spots first.** The largest is language coverage: the Voice Agent API speaks English, Spanish, French, German, Italian and Portuguese today. A language-learning catalog almost certainly includes Japanese, Korean or Mandarin, so the plan maps their catalog to coverage on day one and proposes a route for the rest.
6. **Roll out in phases**: offline bake-off, shadow traffic, canary with automatic rollback, ramp.

The full plan is in [`evals/EVAL_PLAN.md`](evals/EVAL_PLAN.md).

## What the live sessions changed

Two recorded sessions showed what unit tests could not. Each finding is fixed and pinned by a test.

| The session showed | Cause | Fix |
|---|---|---|
| The breakdown changed shape between lessons | The model improvised it | The tool builds the spoken lines; the prompt says to read them as given |
| "Your pronunciation was perfect" | The agent only gets a transcript | Prompt limits feedback to what a transcript shows; a guardrail flags the claim |
| A lesson rejected for a trailing "?" in the respelling | Validator too strict, costing a full extra round trip | Harmless punctuation is cleaned; IPA and digits still fail |
| A rude request refused with no tool call | The prompt pre-empted the tool, so the profanity policy never ran | Every "how do I say" goes through the tool; the policy decides in code |
| First audio about 2.4 s at p50 and 3.0 s at p95 (7 turns) | A 1000 ms silence window plus the tool round trip | Window is 700 ms and configurable; the report separates filler from lesson latency |
| A wrong attempt answered with "I can only help with pronunciation" | A mispronounced word transcribes as nonsense and looked off-topic | The turn after a lesson is an attempt; `check_attempt` handles it |
| "galletas" mispronounced, while its respelling sounded right | The default voice is American English | Native voice per language, with session switching |
| Eight languages offered | The API speaks six languages, and Korean is not on its recognition list either | Five teachable languages; others get a clear "can't speak that yet" |

Still open: the respelling is written by the model and checked only for format. In one session it gave "gah-YET-ahs" for "galletas", where "gah-YEH-tahs" matches the syllables ga-lle-tas. A fluent speaker should spot-check respellings per language, and the bake-off should score them.

## Accent feedback: what is and is not possible

`check_attempt` works from the transcript, so it measures **intelligibility**: was each word understood, and which one was not. It adds a rule-of-thumb tip for a missed word. It cannot hear vowel quality, stress or rhythm, and a strong recognizer will often write an accented word correctly, so "understood" does not mean "pronounced well". The agent is told to say only the former.

Real accent feedback needs pronunciation assessment on the learner's audio. The client already holds that audio, so the extension point is clear: after each attempt, score the clip against the target text with a pronunciation-assessment service or a phoneme-recognition model, and add per-word or per-phoneme scores to the `check_attempt` result. I have not built it. It needs a second vendor and a fluent-speaker validation set, and I would not claim phoneme feedback without both.

## Extra credit: Shadow mode

`lingua-agent --shadow "¿Dónde está la biblioteca?" --language es`

Karaoke for pronunciation. The phrase is on screen, and each word lights up as you say it (green understood, red wrong, grey not yet heard), driven by the streaming partial transcripts. Then the agent gives one sentence of feedback.

Two design choices keep it honest. It measures whether you were understood, not phonetics. And it sends **no keyterms**, because boosting the phrase being tested would bias the recognizer toward hearing it and inflate the score.

## Verified and not verified

- **Confirmed live:** the `wss://agents.assemblyai.com/v1/ws` endpoint, inline session config, microphone audio in, reply audio out, both transcript events, `tool.call`, and `tool.result` sent on `reply.done`.
- **From AssemblyAI's docs, covered by tests, not yet run live:** `session.resume`, `conversation.message` and `reply.create` (the voice-switch handoff), the native voice ids, and the saved-agent REST shape.
- **Docs that disagree with each other:** the tool-call id is `call_id` on one page and `tool_call_id` on another. The code accepts both and echoes whichever arrives.
- **Unknown until heard:** how each native voice sounds giving English instructions and reading the English-style respelling, and how well learner attempts are recognized per language.

`python scripts/probe_protocol.py question.pcm` records every raw event of a real exchange, for checking the wire format after an API change.

## Limitations and next steps

- Five teachable languages, one per session. Japanese and Mandarin wait on AssemblyAI's output voices, which their docs list as coming soon.
- A voice switch drops lesson context and adds a connection gap. Carrying a short summary across in `conversation.message` is the next step.
- Attempt scoring ignores accents in Latin-script text, so "ano" matches "año".
- Terminal client only. A browser client would use the temporary-token flow and gets echo cancellation from the browser.
- Next, in order: run the bake-off on real learner audio, add pronunciation assessment on attempt audio, add an LLM judge with blind human spot checks for translation and respelling quality.
