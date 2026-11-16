# Winning the technical evaluation

Context: a rip-and-replace. The customer's language-learning voice agent runs on another provider. They will run a technical evaluation before moving. The aim is not to show AssemblyAI is good in general. It is to show it is better **on their learners' audio**, and to be the vendor who found the risks first.

Nothing in this document is a bake-off result. The only measurements are from two exploratory sessions, labeled as such in section 7.

## 1. What matters for this customer, and generic benchmarks miss

| Learner reality | Why it breaks voice agents | What to measure |
|---|---|---|
| An English sentence containing a foreign phrase ("how do I say *dónde está*") | Recognizers tuned on one language mangle the foreign words, which are the point | **Target-term recall** on code-switched clips |
| Non-native accents, in both languages | Word error rate on native speech does not predict it | Word error rate and term recall on an accented stratum |
| Pauses mid-request ("how do I say... um... the library") | Turn detection cuts the learner off, or waits too long | False cutoffs against latency, swept over the silence window (`TURN_MIN_SILENCE_MS`) |
| "Say it slower", "again", talking over the agent | Barge-in decides whether the product feels alive | Barge-in success and recovery |
| The phrase spoken back in the target language | An English voice mispronounces it (seen live with "galletas") | Blind fluent-speaker ratings, one native voice per language |
| The learner's attempt | A recognizer that corrects speech hides the learner's errors | Whether feedback names the right word and never claims more than a transcript shows |

## 2. Agree the scorecard before running anything

Propose weights at kickoff and get them signed off, so nobody argues metrics after seeing results. Starting proposal (`evals/scoring.py::DEFAULT_WEIGHTS`): recognition 35%, teaching quality 25%, latency 20%, safety 20%. The harness reports a composite only when all four were measured on that slice; a missing component is shown as missing, never as zero. Thresholds come from the current vendor's numbers on the same clips. A typical win condition: at least as good on every stratum, clearly better on code-switched speech.

## 3. Test set

- **Their data.** A few hundred anonymized production utterances, stratified: clean, accented, noisy, code-switched, hesitant, interrupting, adversarial (off-topic, injection, PII), and requests for languages a vendor cannot speak. Synthetic speech flatters every recognizer and gets no weight.
- Human-verified references, with the must-hear terms, accepted translations and expected behavior (teach or refuse) per clip. `evals/audio_manifest.example.jsonl` shows the format.
- A held-out slice nobody tunes on.
- For attempts: clips of learners saying a phrase correctly, slightly wrong and badly wrong, each labeled with the word that should be flagged.

## 4. Fairness rules

1. Identical audio, real-time pacing and scorer, in one harness (`evals/compare.py`).
2. Latency is timed from the last speech chunk sent to the first reply audio byte, for every vendor. Timing from a vendor's own end-of-speech event rewards whoever reports it latest.
3. Latency is reported for the lesson, not only the first sound. A filler such as "Sure" makes first audio look fast while the lesson is slow, so the session report separates them (`answer_p50_ms`, `post_tool_p50_ms`).
4. Each vendor runs twice: default settings (what a developer gets on day one) and tuned (best honest effort, documented).
5. Anything judged by people or by a model is blind to the vendor.
6. Report n, pooled word error rate, p50 and p95, and paired bootstrap confidence intervals on per-clip differences (`compare.paired_comparison`). No win is claimed inside the noise.
7. Failures count. A timeout or an empty transcript is a miss, never dropped (`compare.evaluate`).
8. The customer gets the harness and can re-run it. A result they can reproduce is the strongest asset in the deal.

## 5. Phases

| Phase | What | Exit criterion |
|---|---|---|
| 0. Kickoff (days) | Success criteria, scorecard, data access, scope, catalog-to-coverage map | Signed-off scorecard |
| 1. Offline bake-off (week 1) | Both vendors on the recorded set | Scorecard filled, failure analysis shared |
| 2. Shadow traffic (week 2) | Mirror live sessions to the new stack; learners never hear it | Parity on the live distribution, no safety regressions |
| 3. Canary (weeks 3 to 4) | About 5% of sessions, automatic rollback on a latency or error guardrail | Learner metrics (completion, retries) no worse |
| 4. Ramp | 25, 50, 100% | Cost and quality dashboards agreed |

## 6. Raise the weak spots first

Credibility comes from naming the risks before the customer finds them.

- **Output language coverage. This is the big one.** Per AssemblyAI's documentation the Voice Agent API speaks English, Spanish, French, German, Italian and Portuguese, with more languages listed as coming soon. A language-learning catalog will be wider. Map their catalog on day one. For uncovered languages, say so and propose a route: keep their current voice for those languages behind the same client, or a stitched pipeline with AssemblyAI recognition and a third-party voice.
- **One voice per session.** Each native voice speaks its language plus English, and the voice cannot change mid-session. This reference agent restarts the session to switch. For their product, one saved agent per course language is the cleaner fit; confirm how their app handles a learner changing course mid-call.
- **Pronunciation feedback.** The API returns transcripts, not phoneme scores. If their product promises accent scoring, that is a separate assessment step on the learner's audio. Agree whether "was understood" is the product requirement, or phonetic accuracy is.
- **Blocking output.** Speech-to-speech cannot stop audio after it is generated. If their policy needs more than the tool checkpoint, design it explicitly.
- **Their existing prompts and model.** They have tuning invested. Confirm the model options and what migrating their prompts involves.
- **Telephony, data residency, compliance.** Get what they need confirmed in writing, not assumed from documentation.

## 7. What exists today

Built and tested offline: the harness, scoring and statistics, the AssemblyAI adapter (against a scripted socket), the guardrail eval, and an example manifest.

Exploratory only: two live sessions, 7 measured turns, with a 1000 ms silence window. First audio was about 2.4 s at p50 and 3.0 s at p95. That is not a benchmark. It did show that the silence window was a full second of the delay, which is why it became a tunable setting.

Pending, and needing keys and audio: live bake-off runs, the incumbent adapter, the recorded clip set, the silence-window sweep, and fluent-speaker ratings. Translation quality is scored by exact match against accepted translations; a model judge with blind human spot checks is the planned next layer, and it should also score the pronunciation respellings.

Vendor-published accuracy, latency and pricing claims are inputs to test on the customer's audio. None were verified here, so none are quoted.
