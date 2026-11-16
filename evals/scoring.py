"""Scoring functions for the customer bake-off. Pure, dependency-free, unit-tested.

Metrics
  wer             word error rate (per character for CJK). Reported POOLED across
                  a set (total errors / total reference words): averaging
                  per-utterance WER over-weights short utterances.
  term_recall     share of must-hear target-language terms recognized; the metric a
                  language-learning customer actually cares about, because a transcript
                  can have a fine WER and still miss every foreign word
  paired_bootstrap  confidence interval for a per-utterance difference between two
                  vendors on the same audio, so a "win" inside the noise is not claimed
  composite       weighted blend, weights agreed with the customer BEFORE the run

Text is compared after `normalize_text`: case, punctuation and Latin accents are
ignored (recognizers differ in formatting), so accent-only errors are not counted.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from lingua_agent.languages import normalize_text
from lingua_agent.shadow import tokenize

DEFAULT_WEIGHTS: dict[str, float] = {"recognition": 0.35, "teaching": 0.25, "latency": 0.20, "safety": 0.20}


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """(edit errors, reference length) so callers can pool across utterances."""
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    return edit_distance(ref, hyp), len(ref)


def wer(reference: str, hypothesis: str) -> float:
    errors, n = wer_counts(reference, hypothesis)
    if n == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / n


def pooled_wer(counts: Sequence[tuple[int, int]]) -> float:
    total_ref = sum(n for _, n in counts)
    return sum(e for e, _ in counts) / total_ref if total_ref else 0.0


def term_recall(terms: Sequence[str], hypothesis: str) -> float:
    """Fraction of `terms` present (accent/case-insensitive, whole-word aware)."""
    if not terms:
        return 1.0
    hyp_tokens = tokenize(hypothesis)
    hyp = " " + " ".join(hyp_tokens) + " "
    return sum(" " + " ".join(tokenize(t)) + " " in hyp for t in terms) / len(terms)


def phrase_matches(candidate: str, accepted: Sequence[str]) -> bool:
    """True if `candidate` equals any accepted translation after normalization."""
    norm = normalize_text(candidate)
    return bool(norm) and any(norm == normalize_text(a) for a in accepted)


def latency_score(p95_ms: float, target_ms: float = 1000.0, ceiling_ms: float = 3000.0) -> float:
    """1.0 at or under target, 0.0 at or over ceiling, linear between."""
    if p95_ms != p95_ms:  # NaN: no measurements is not a pass
        return 0.0
    if p95_ms <= target_ms:
        return 1.0
    if p95_ms >= ceiling_ms:
        return 0.0
    return 1 - (p95_ms - target_ms) / (ceiling_ms - target_ms)


def paired_bootstrap(diffs: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict[str, float | int | bool]:
    """Mean of paired differences with a percentile bootstrap CI.

    `significant` is True only when the interval excludes zero. With few clips
    the interval is wide; that is the point.
    """
    n = len(diffs)
    if n == 0:
        raise ValueError("no paired observations")
    mean = sum(diffs) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
    lo = means[int(n_boot * alpha / 2)]
    hi = means[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]
    return {"n": n, "mean_diff": mean, "ci_low": lo, "ci_high": hi, "significant": lo > 0 or hi < 0}


def composite(scores: Mapping[str, float], weights: Mapping[str, float] = DEFAULT_WEIGHTS) -> float:
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.0")
    missing = set(weights) - set(scores)
    if missing:
        raise ValueError(f"missing score components: {sorted(missing)}")
    return sum(scores[k] * w for k, w in weights.items())
