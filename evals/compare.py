"""Run a labeled audio manifest through one or more providers and score the result.

    python -m evals.compare evals/audio_manifest.jsonl --providers assemblyai,incumbent

Manifest rows (JSONL). Only id, audio and reference are required:
  {"id": "es-001", "audio": "evals/audio/es-001.pcm",
   "reference": "how do I say where is the library in Spanish",   # what was said
   "terms": ["Spanish", "library"],                # words the recognizer must not miss
   "expected_phrases": ["¿Dónde está la biblioteca?"],  # accepted translations
   "expect": "teach",                              # or "refuse" for off-topic / injection clips
   "speech_end_ms": 2400,                          # where speech ends, if the clip has trailing silence
   "category": "code_switch"}                      # clean | accent | noise | hesitation | adversarial ...

Add --language es (fr, de, it, pt) to run AssemblyAI with that language's native voice;
run the manifest once per course language when voice quality is being rated.

Audio is 24 kHz mono PCM16. Use real learner speech, ideally the customer's own
recorded sessions; synthetic TTS audio flatters every recognizer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from lingua_agent.metrics import percentile

from . import scoring
from .providers import Provider, TurnResult

EXPECTATIONS = ("teach", "refuse")


@dataclass(frozen=True)
class RowResult:
    id: str
    category: str
    provider: str
    wer_errors: int
    ref_words: int
    term_recall: float
    ttfa_ms: float | None
    phrase_correct: bool | None = None       # teach rows with expected_phrases
    refused_correctly: bool | None = None    # refuse rows
    error: str | None = None

    @property
    def wer(self) -> float:
        return self.wer_errors / self.ref_words if self.ref_words else 0.0


def load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    for r in rows:
        for key in ("id", "audio", "reference"):
            if key not in r:
                raise ValueError(f"manifest row {r.get('id', '?')} missing {key!r}")
        if r["id"] in seen:
            raise ValueError(f"duplicate manifest id {r['id']!r}")
        seen.add(r["id"])
        if r.get("expect", "teach") not in EXPECTATIONS:
            raise ValueError(f"manifest row {r['id']}: expect must be one of {EXPECTATIONS}")
    return rows


def _approved_phrases(res: TurnResult) -> list[str]:
    return [str(c.get("arguments", {}).get("phrase", "")) for c in res.tool_calls if c.get("status") == "approved"]


def score_row(row: dict, provider_name: str, res: TurnResult) -> RowResult:
    errors, ref_words = scoring.wer_counts(row["reference"], res.transcript)
    approved = _approved_phrases(res)
    phrase_correct = refused = None
    if row.get("expect", "teach") == "refuse":
        refused = not approved  # nothing was taught in response to an off-topic or hostile request
    elif row.get("expected_phrases"):
        phrase_correct = any(scoring.phrase_matches(p, row["expected_phrases"]) for p in approved)
    return RowResult(
        row["id"], row.get("category", "uncategorized"), provider_name, errors, ref_words,
        scoring.term_recall(row.get("terms", []), res.transcript), res.ttfa_ms, phrase_correct, refused,
    )


async def evaluate(rows: list[dict], provider: Provider) -> list[RowResult]:
    out: list[RowResult] = []
    for r in rows:
        try:
            res = await provider.run_turn(Path(r["audio"]), r.get("speech_end_ms"))
        except NotImplementedError:
            raise
        except Exception as exc:  # one bad clip must not sink the run; it is counted, not hidden
            ref_words = scoring.wer_counts(r["reference"], "")[1]
            refuse = r.get("expect", "teach") == "refuse"
            out.append(RowResult(
                r["id"], r.get("category", "uncategorized"), provider.name, ref_words, ref_words, 0.0, None,
                phrase_correct=False if r.get("expected_phrases") and not refuse else None,
                refused_correctly=False if refuse else None, error=repr(exc)))
            continue
        out.append(score_row(r, provider.name, res))
    return out


def _num(value: float) -> float | None:
    return None if value != value else value  # NaN (nothing measured) is not valid JSON


def _rate(values: list[bool | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def _block(rs: list[RowResult]) -> dict:
    ttfa = [r.ttfa_ms for r in rs if r.ttfa_ms is not None]
    wer = scoring.pooled_wer([(r.wer_errors, r.ref_words) for r in rs])
    recall = sum(r.term_recall for r in rs) / len(rs)
    block = {
        "n": len(rs),
        "errors": sum(r.error is not None for r in rs),
        "wer_pooled": wer,
        "term_recall_mean": recall,
        "ttfa_p50_ms": _num(percentile(ttfa, 50)),
        "ttfa_p95_ms": _num(percentile(ttfa, 95)),
        "teaching_accuracy": _rate([r.phrase_correct for r in rs]),
        "refusal_accuracy": _rate([r.refused_correctly for r in rs]),
    }
    components = {
        "recognition": 0.5 * (1 - min(1.0, wer)) + 0.5 * recall,
        "teaching": block["teaching_accuracy"],
        "latency": scoring.latency_score(percentile(ttfa, 95)) if ttfa else None,
        "safety": block["refusal_accuracy"],
    }
    # No composite unless every component was actually measured: a blank is not a zero.
    block["composite"] = None if None in components.values() else scoring.composite(components)
    return block


def summarize(results: list[RowResult]) -> dict:
    by_provider: dict[str, list[RowResult]] = defaultdict(list)
    for r in results:
        by_provider[r.provider].append(r)
    summary: dict = {}
    for name, rs in by_provider.items():
        by_cat: dict[str, list[RowResult]] = defaultdict(list)
        for r in rs:
            by_cat[r.category].append(r)
        summary[name] = {"overall": _block(rs), "by_category": {c: _block(v) for c, v in by_cat.items()}}
    return summary


def paired_comparison(results: list[RowResult], baseline: str, challenger: str) -> dict:
    """Per-clip differences (challenger minus baseline) with bootstrap intervals.

    Positive term_recall and negative wer / ttfa_ms favor the challenger.
    """
    by_id: dict[str, dict[str, RowResult]] = defaultdict(dict)
    for r in results:
        by_id[r.id][r.provider] = r
    pairs = [(v[baseline], v[challenger]) for v in by_id.values() if baseline in v and challenger in v]
    out: dict = {"baseline": baseline, "challenger": challenger}
    metrics = {
        "term_recall": [c.term_recall - b.term_recall for b, c in pairs],
        "wer": [c.wer - b.wer for b, c in pairs],
        "ttfa_ms": [c.ttfa_ms - b.ttfa_ms for b, c in pairs if b.ttfa_ms is not None and c.ttfa_ms is not None],
    }
    for name, diffs in metrics.items():
        out[name] = scoring.paired_bootstrap(diffs) if diffs else None
    return out


def _build_providers(names: list[str], language: str | None = None) -> list[Provider]:  # pragma: no cover (needs live keys)
    from lingua_agent.cli import apply_language
    from lingua_agent.config import Settings, load_dotenv

    from .providers import AssemblyAIProvider, IncumbentProvider

    load_dotenv()
    settings = apply_language(Settings.from_env(), language)
    table = {"assemblyai": lambda: AssemblyAIProvider(settings), "incumbent": IncumbentProvider}
    return [table[n]() for n in names]


async def _main(args: argparse.Namespace) -> int:  # pragma: no cover
    rows = load_manifest(Path(args.manifest))
    names = args.providers.split(",")
    results: list[RowResult] = []
    for provider in _build_providers(names, args.language):
        results += await evaluate(rows, provider)
    report = {"summary": summarize(results), "rows": [asdict(r) for r in results]}
    if len(names) == 2:
        report["paired"] = paired_comparison(results, baseline=names[1], challenger=names[0])
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--providers", default="assemblyai", help="comma-separated; with two, the second is the baseline")
    ap.add_argument("--language", help="course language for AssemblyAI's native voice: es, fr, de, it or pt")
    ap.add_argument("--out", default="results/compare.json")
    sys.exit(asyncio.run(_main(ap.parse_args())))
