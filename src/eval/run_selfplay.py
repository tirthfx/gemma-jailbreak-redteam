"""
Phase G: run self-play automated jailbreak search on a benchmark subset (one
prompt per category by default, matching run_gcg.py's approach — this attack is
several generation calls deep per prompt, so a full 46-prompt run isn't the
right default) and compare ASR-after-N-rounds against the single-shot black-box
baseline (Phase A) on the same prompts.

Usage:
    python -m src.eval.run_selfplay --limit 8 --max-rounds 3

Outputs:
    results/raw/selfplay.jsonl      - full round-by-round transcripts (local only)
    results/summary_selfplay.csv    - ASR after self-play vs. single-shot baseline (committed)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.attacks.selfplay.attacker_loop import run_selfplay
from src.benchmark.judge import attack_success_rate, judge_response
from src.benchmark.prompts import load_benchmark
from src.model import generate, load_model

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_PATH = ROOT / "results" / "summary_selfplay.csv"


def pick_subset(limit: int):
    bench = load_benchmark()
    seen = set()
    subset = []
    for p in bench:
        if p.category not in seen:
            subset.append(p)
            seen.add(p.category)
        if len(subset) >= limit:
            break
    return subset


def run(limit: int, max_rounds: int, max_new_tokens: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[selfplay] loading model on device...", flush=True)
    t0 = time.time()
    model, tokenizer, processor, device = load_model()
    print(f"[selfplay] model loaded on {device} in {time.time() - t0:.1f}s", flush=True)

    subset = pick_subset(limit)
    raw_path = RAW_DIR / "selfplay.jsonl"
    single_shot_verdicts = []
    selfplay_verdicts = []

    with raw_path.open("w") as raw_f:
        for i, p in enumerate(subset, 1):
            print(f"\n[selfplay] === prompt {i}/{len(subset)}: [{p.category}] {p.id} ===", flush=True)

            single_shot_response = generate(model, tokenizer, p.text, max_new_tokens=max_new_tokens, device=device)
            single_shot_verdict = judge_response(single_shot_response)
            single_shot_verdicts.append(single_shot_verdict)

            result = run_selfplay(model, tokenizer, device, p.text,
                                   max_rounds=max_rounds, max_new_tokens=max_new_tokens)
            selfplay_verdicts.append(result["final_verdict"])

            record = {
                "id": p.id, "category": p.category, "prompt": p.text,
                "single_shot_verdict": single_shot_verdict,
                "selfplay_final_verdict": result["final_verdict"],
                "succeeded_at_round": result["succeeded_at_round"],
                "rounds": result["rounds"],
            }
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()

            print(f"[selfplay] {p.id}: single_shot={single_shot_verdict} "
                  f"selfplay_after_{max_rounds}_rounds={result['final_verdict']} "
                  f"(succeeded_at_round={result['succeeded_at_round']})", flush=True)

    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n", "asr"])
        writer.writerow(["single_shot", len(single_shot_verdicts), f"{attack_success_rate(single_shot_verdicts):.3f}"])
        writer.writerow([f"selfplay_{max_rounds}_rounds", len(selfplay_verdicts), f"{attack_success_rate(selfplay_verdicts):.3f}"])

    print(f"\n[selfplay] wrote {SUMMARY_PATH}")
    print(f"[selfplay] single-shot ASR: {attack_success_rate(single_shot_verdicts):.1%}")
    print(f"[selfplay] self-play ASR after {max_rounds} rounds: {attack_success_rate(selfplay_verdicts):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="Number of benchmark prompts (one per category)")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    run(limit=args.limit, max_rounds=args.max_rounds, max_new_tokens=args.max_new_tokens)
