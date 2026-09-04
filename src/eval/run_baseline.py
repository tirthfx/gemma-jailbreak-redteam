"""
Phase A: measure the stock model's refusal rate on the benchmark, with no attack
applied. This is the control group everything else in the project compares against.

Usage:
    python -m src.eval.run_baseline [--limit N] [--max-new-tokens N]

Outputs:
    results/raw/baseline.jsonl        - full prompt/response/verdict per item (local only, gitignored)
    results/summary_baseline.csv      - per-category + overall ASR (committed)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.judge import attack_success_rate, judge_response
from src.benchmark.prompts import BENIGN_PROMPTS, load_benchmark
from src.model import generate, load_model

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_PATH = ROOT / "results" / "summary_baseline.csv"


def run(limit: int | None, max_new_tokens: int, include_benign: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] loading model on device...", flush=True)
    t0 = time.time()
    model, tokenizer, processor, device = load_model()
    print(f"[baseline] model loaded on {device} in {time.time() - t0:.1f}s", flush=True)

    prompts = load_benchmark()
    if include_benign:
        prompts = prompts + BENIGN_PROMPTS
    if limit:
        prompts = prompts[:limit]

    raw_path = RAW_DIR / "baseline.jsonl"
    records = []

    with raw_path.open("w") as raw_f:
        for i, p in enumerate(prompts, 1):
            t_start = time.time()
            response = generate(model, tokenizer, p.text, max_new_tokens=max_new_tokens, device=device)
            verdict = judge_response(response)
            dt = time.time() - t_start

            record = {
                "id": p.id,
                "category": p.category,
                "prompt": p.text,
                "response": response,
                "verdict": verdict,
            }
            records.append(record)
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()

            print(
                f"[baseline] {i}/{len(prompts)} [{p.category}] {p.id} -> {verdict} "
                f"({dt:.1f}s)",
                flush=True,
            )

    # Aggregate per category + overall
    by_cat: dict[str, list[str]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r["verdict"])

    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "n", "asr"])
        for cat, verdicts in sorted(by_cat.items()):
            writer.writerow([cat, len(verdicts), f"{attack_success_rate(verdicts):.3f}"])
        overall = [r["verdict"] for r in records if r["category"] != "benign"]
        writer.writerow(["OVERALL (excl. benign)", len(overall), f"{attack_success_rate(overall):.3f}"])

    print(f"\n[baseline] wrote {raw_path} and {SUMMARY_PATH}")
    print(f"[baseline] overall ASR (excl. benign): {attack_success_rate(overall):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N prompts (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-benign", action="store_true", help="Skip the benign sanity-check prompts")
    args = ap.parse_args()
    run(limit=args.limit, max_new_tokens=args.max_new_tokens, include_benign=not args.no_benign)
