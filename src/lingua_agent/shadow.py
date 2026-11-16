"""Shadow mode (extra credit): live word-by-word pronunciation feedback.

The learner reads a target phrase aloud. As streaming partials arrive
(`transcript.user.delta`, whose text is cumulative), each target word is marked
heard / wrong / not yet heard, so the learner watches the phrase "light up"
while they speak. It leans on fast streaming partials and multilingual
recognition.

Two measurement caveats, both deliberate design inputs:
  * A recognizer that is very good at "fixing" speech also fixes a learner's
    mispronunciation, so this measures *intelligibility* (would a listener get
    the word?), not phonetic accuracy.
  * The target phrase must NOT be passed as a keyterm in this mode. Boosting the
    expected answer biases the recognizer toward hearing it and inflates the
    score. `agent_spec.build_shadow_config` therefore sends no keyterms.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .languages import normalize_text

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def tokenize(text: str) -> list[str]:
    norm = normalize_text(text)
    if _CJK.search(norm):
        # Japanese and Chinese have no word spaces, and recognizers space Korean
        # inconsistently: compare per character and ignore spacing.
        return [ch for ch in norm if not ch.isspace()]
    return norm.split()


@dataclass(frozen=True)
class WordStatus:
    word: str
    state: str  # match | wrong | missing


@dataclass(frozen=True)
class Alignment:
    words: tuple[WordStatus, ...]
    extra_words: int

    @property
    def score(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.state == "match" for w in self.words) / len(self.words)


def align(target: str, heard: str) -> Alignment:
    t, h = tokenize(target), tokenize(heard)
    out: list[WordStatus] = []
    extra = 0
    for op, t0, t1, h0, h1 in difflib.SequenceMatcher(a=t, b=h, autojunk=False).get_opcodes():
        if op == "equal":
            out += [WordStatus(w, "match") for w in t[t0:t1]]
        elif op == "replace":
            out += [WordStatus(w, "wrong") for w in t[t0:t1]]
            extra += max(0, (h1 - h0) - (t1 - t0))
        elif op == "delete":
            out += [WordStatus(w, "missing") for w in t[t0:t1]]
        else:  # insert
            extra += h1 - h0
    return Alignment(tuple(out), extra)


_ANSI = {"match": "\033[32m", "wrong": "\033[31m", "missing": "\033[90m"}


def render_ansi(alignment: Alignment) -> str:
    return " ".join(f"{_ANSI[w.state]}{w.word}\033[0m" for w in alignment.words)
