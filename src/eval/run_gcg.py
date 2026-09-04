"""
Phase F: run the GCG suffix search on a scoped-down subset of the benchmark
(default: first prompt from each category, ~8 prompts), then check whether the
found suffix actually produces a compliant *full* response through the normal
chat/generate path (the optimization target is just a forced affirmative
opener — this step verifies it actually carries through to real compliance).

NOT YET RUN — see suffix_attack.py's module docstring. First invocation should
use small --num-steps/--batch-size as a smoke test before the full scoped run.

Usage:
    python -m src.eval.run_gcg --limit 8 --num-steps 100 --batch-size 24

Outputs:
    results/raw/gcg.jsonl        - suffix + before/after transcripts (local only)
    results/summary_gcg.csv      - ASR before vs. after suffix, on the subset (committed)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.attacks.gcg.suffix_attack import run_gcg
from src.benchmark.judge import attack_success_rate, judge_response
from src.benchmark.prompts import load_benchmark
from src.model import generate, load_model

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_PATH = ROOT / "results" / "summary_gcg.csv"


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


def run(limit: int, num_steps: int, batch_size: int, suffix_len: int, topk: int, max_new_tokens: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[gcg] loading model on device...", flush=True)
    t0 = time.time()
    model, tokenizer, processor, device = load_model()
    print(f"[gcg] model loaded on {device} in {time.time() - t0:.1f}s", flush=True)

    subset = pick_subset(limit)
    raw_path = RAW_DIR / "gcg.jsonl"
    records = []

    with raw_path.open("w") as raw_f:
        for i, p in enumerate(subset, 1):
            print(f"\n[gcg] === prompt {i}/{len(subset)}: [{p.category}] {p.id} ===", flush=True)

            before_response = generate(model, tokenizer, p.text, max_new_tokens=max_new_tokens, device=device)
            before_verdict = judge_response(before_response)

            t_start = time.time()
            result = run_gcg(model, tokenizer, device, p.text,
                              suffix_len=suffix_len, num_steps=num_steps,
                              batch_size=batch_size, topk=topk)
            search_time = time.time() - t_start

            after_response = generate(model, tokenizer, result["prompt_with_suffix"],
                                       max_new_tokens=max_new_tokens, device=device)
            after_verdict = judge_response(after_response)

            record = {
                "id": p.id,
                "category": p.category,
                "prompt": p.text,
                "suffix": result["suffix_text"],
                "final_loss": result["final_loss"],
                "search_time_s": search_time,
                "before_response": before_response,
                "before_verdict": before_verdict,
                "after_response": after_response,
                "after_verdict": after_verdict,
            }
            records.append(record)
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()

            print(f"[gcg] {p.id}: before={before_verdict} after={after_verdict} "
                  f"(loss={result['final_loss']:.3f}, {search_time:.1f}s search)", flush=True)

    before_verdicts = [r["before_verdict"] for r in records]
    after_verdicts = [r["after_verdict"] for r in records]

    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n", "asr"])
        writer.writerow(["before_suffix", len(before_verdicts), f"{attack_success_rate(before_verdicts):.3f}"])
        writer.writerow(["after_suffix", len(after_verdicts), f"{attack_success_rate(after_verdicts):.3f}"])

    print(f"\n[gcg] wrote {SUMMARY_PATH}")
    print(f"[gcg] ASR before suffix: {attack_success_rate(before_verdicts):.1%}")
    print(f"[gcg] ASR after suffix:  {attack_success_rate(after_verdicts):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="Number of benchmark prompts (one per category)")
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--suffix-len", type=int, default=20)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    run(limit=args.limit, num_steps=args.num_steps, batch_size=args.batch_size,
        suffix_len=args.suffix_len, topk=args.topk, max_new_tokens=args.max_new_tokens)
