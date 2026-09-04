"""
Re-score every saved raw transcript (results/raw/*.jsonl) with the current judge
and rewrite the summary CSVs — without re-running any generation. Used after a
judge fix (e.g. adding the NON_ANSWER category) so past runs don't need to be
redone from scratch just to get corrected numbers.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark.judge import attack_success_rate, judge_response, verdict_breakdown

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"


def rescore_baseline():
    path = RAW_DIR / "baseline.jsonl"
    if not path.exists():
        return
    rows = [json.loads(l) for l in path.open()]
    for r in rows:
        r["verdict"] = judge_response(r["response"])

    by_cat: dict[str, list[str]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["verdict"])

    out = ROOT / "results" / "summary_baseline.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "n", "asr", "refusal_rate", "non_answer_rate"])
        for cat, verdicts in sorted(by_cat.items()):
            bd = verdict_breakdown(verdicts)
            writer.writerow([cat, len(verdicts), f"{attack_success_rate(verdicts):.3f}",
                              f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
        overall = [r["verdict"] for r in rows if r["category"] != "benign"]
        bd = verdict_breakdown(overall)
        writer.writerow(["OVERALL (excl. benign)", len(overall), f"{attack_success_rate(overall):.3f}",
                          f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
    print(f"[rescore] rewrote {out}")


def rescore_blackbox():
    files = sorted(RAW_DIR.glob("blackbox_*.jsonl"))
    if not files:
        return

    all_rows = []  # (technique, category, verdict)
    for path in files:
        technique = path.stem.removeprefix("blackbox_")
        rows = [json.loads(l) for l in path.open()]
        for r in rows:
            r["verdict"] = judge_response(r["response"])
            all_rows.append((technique, r["category"], r["verdict"]))

    out = ROOT / "results" / "summary_blackbox.csv"
    techniques = sorted({t for t, _, _ in all_rows}, key=lambda t: [f.stem for f in files].index(f"blackbox_{t}"))
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["technique", "category", "n", "asr", "refusal_rate", "non_answer_rate"])
        for technique in techniques:
            rows = [(cat, v) for (t, cat, v) in all_rows if t == technique]
            by_cat: dict[str, list[str]] = {}
            for cat, v in rows:
                by_cat.setdefault(cat, []).append(v)
            for cat, verdicts in sorted(by_cat.items()):
                bd = verdict_breakdown(verdicts)
                writer.writerow([technique, cat, len(verdicts), f"{attack_success_rate(verdicts):.3f}",
                                  f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
            overall = [v for _, v in rows]
            bd = verdict_breakdown(overall)
            writer.writerow([technique, "OVERALL", len(overall), f"{attack_success_rate(overall):.3f}",
                              f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
    print(f"[rescore] rewrote {out}")
    for technique in techniques:
        overall = [v for (t, _, v) in all_rows if t == technique]
        bd = verdict_breakdown(overall)
        print(f"[rescore] {technique}: ASR={attack_success_rate(overall):.1%} "
              f"refusal={bd['refusal']:.1%} non_answer={bd['non_answer']:.1%}")


def rescore_ablation():
    files = sorted(RAW_DIR.glob("ablation_sweep_alpha*.jsonl")) + sorted(RAW_DIR.glob("ablation_alpha*.jsonl"))
    if not files:
        return

    all_rows = []  # (alpha, category, verdict)
    for path in files:
        # filenames: ablation_sweep_alpha<A>.jsonl or ablation_alpha<A>.jsonl
        alpha = path.stem.rsplit("alpha", 1)[1]
        rows = [json.loads(l) for l in path.open()]
        for r in rows:
            r["verdict"] = judge_response(r["response"])
            all_rows.append((alpha, r["category"], r["verdict"]))

    out = ROOT / "results" / "summary_ablation.csv"
    alphas = sorted({a for a, _, _ in all_rows}, key=float)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "category", "n", "asr", "refusal_rate", "non_answer_rate"])
        for alpha in alphas:
            rows = [(cat, v) for (a, cat, v) in all_rows if a == alpha]
            by_cat: dict[str, list[str]] = {}
            for cat, v in rows:
                by_cat.setdefault(cat, []).append(v)
            for cat, verdicts in sorted(by_cat.items()):
                bd = verdict_breakdown(verdicts)
                writer.writerow([alpha, cat, len(verdicts), f"{attack_success_rate(verdicts):.3f}",
                                  f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
            overall = [v for _, v in rows]
            bd = verdict_breakdown(overall)
            writer.writerow([alpha, "OVERALL", len(overall), f"{attack_success_rate(overall):.3f}",
                              f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
    print(f"[rescore] rewrote {out}")
    for alpha in alphas:
        overall = [v for (a, _, v) in all_rows if a == alpha]
        bd = verdict_breakdown(overall)
        print(f"[rescore] alpha={alpha}: ASR={attack_success_rate(overall):.1%} "
              f"refusal={bd['refusal']:.1%} non_answer={bd['non_answer']:.1%}")


if __name__ == "__main__":
    rescore_baseline()
    rescore_blackbox()
    rescore_ablation()
