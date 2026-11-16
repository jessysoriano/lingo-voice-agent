"""Offline guardrail eval: labeled cases, exact-match scoring, no network, no keys.

    python -m evals.run_offline_eval

Same scoring idea as the meal-analysis project: a case passes only on an exact
match of the expected flags / status / reason. Cases marked `known_gap` document
where the deterministic layer is knowingly wrong. They are reported separately and
excluded from the CI gate, because hiding them would overstate coverage.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from lingua_agent import coaching, guardrails, tutor

CASES_PATH = Path(__file__).with_name("guard_cases.jsonl")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(case: dict) -> tuple[bool, str]:
    kind, inp, exp = case["kind"], case["input"], case["expect"]
    if kind == "transcript":
        got = guardrails.check_transcript(inp)
        ok = set(got.flags) == set(exp["flags"]) and got.allowed == exp["allowed"]
        return ok, f"flags={sorted(got.flags)} allowed={got.allowed}"
    if kind == "agent_text":
        got = guardrails.check_agent_text(inp)
        return set(got.flags) == set(exp["flags"]), f"flags={sorted(got.flags)}"
    if kind == "tool_args":
        got = tutor.teach_phrase(inp, profanity_policy=case.get("policy", "label"),
                                 course_language=case.get("course_language"),
                                 auto_switch=case.get("auto_switch", False))
        ok = got["status"] == exp["status"] and got.get("reason") == exp.get("reason")
        return ok, f"status={got['status']} reason={got.get('reason')}"
    if kind == "attempt":
        got = coaching.check_attempt(inp)
        ok = got.get("verdict") == exp["verdict"] and got.get("focus_word") == exp.get("focus_word")
        return ok, f"verdict={got.get('verdict')} focus_word={got.get('focus_word')}"
    raise ValueError(f"unknown case kind {kind!r}")


def run(cases: list[dict] | None = None) -> dict:
    cases = cases if cases is not None else load_cases()
    by_kind: dict[str, list[bool]] = defaultdict(list)
    failures, gaps = [], []
    for c in cases:
        ok, detail = run_case(c)
        if c.get("known_gap"):
            gaps.append({"id": c["id"], "expected_ideal": c["expect"], "got": detail, "still_a_gap": not ok})
            continue
        by_kind[c["kind"]].append(ok)
        if not ok:
            failures.append({"id": c["id"], "got": detail, "expect": c["expect"]})
    gated = [ok for v in by_kind.values() for ok in v]
    return {
        "pass_rate": sum(gated) / len(gated) if gated else 0.0,
        "by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
        "n_gated": len(gated),
        "failures": failures,
        "known_gaps": gaps,
    }


if __name__ == "__main__":
    result = run()
    Path("results").mkdir(exist_ok=True)
    Path("results/offline_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if not result["failures"] else 1)
