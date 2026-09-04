"""
Phase B: run every black-box prompt-jailbreak technique across the benchmark and
compare each one's Attack Success Rate (ASR) against the Phase A baseline.

Usage:
    python -m src.eval.run_blackbox [--limit N] [--max-new-tokens N] [--techniques a,b,c]

Outputs:
    results/raw/blackbox_<technique>.jsonl   - full transcripts per technique (local only)
    results/summary_blackbox.csv             - ASR per technique x category (committed)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.attacks.blackbox import multiturn, obfuscation, prefix_injection, roleplay
from src.benchmark.judge import attack_success_rate, judge_response
from src.benchmark.prompts import load_benchmark
from src.model import load_model

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_PATH = ROOT / "results" / "summary_blackbox.csv"

TECHNIQUES = {
    roleplay.NAME: roleplay.run,
    obfuscation.NAME: obfuscation.run,          # base64 variant (default)
    "obfuscation_leetspeak": obfuscation.run_leetspeak,
    multiturn.NAME: multiturn.run,
    prefix_injection.NAME: prefix_injection.run,
}


def run(limit: int | None, max_new_tokens: int, techniques: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[blackbox] loading model on device...", flush=True)
    t0 = time.time()
    model, tokenizer, processor, device = load_model()
    print(f"[blackbox] model loaded on {device} in {time.time() - t0:.1f}s", flush=True)

    prompts = load_benchmark()
    if limit:
        prompts = prompts[:limit]

    all_rows = []  # (technique, category, verdict) for the summary

    for technique in techniques:
        run_fn = TECHNIQUES[technique]
        raw_path = RAW_DIR / f"blackbox_{technique}.jsonl"
        print(f"\n[blackbox] === technique: {technique} ({len(prompts)} prompts) ===", flush=True)

        with raw_path.open("w") as raw_f:
            for i, p in enumerate(prompts, 1):
                t_start = time.time()
                try:
                    response = run_fn(model, tokenizer, device, p.text, max_new_tokens=max_new_tokens)
                except Exception as e:  # keep the run going even if one technique/prompt errors
                    response = f"[ERROR: {e}]"
                verdict = judge_response(response)
                dt = time.time() - t_start

                record = {
                    "technique": technique,
                    "id": p.id,
                    "category": p.category,
                    "prompt": p.text,
                    "response": response,
                    "verdict": verdict,
                }
                raw_f.write(json.dumps(record) + "\n")
                raw_f.flush()
                all_rows.append((technique, p.category, verdict))

                print(
                    f"[blackbox:{technique}] {i}/{len(prompts)} [{p.category}] {p.id} -> {verdict} ({dt:.1f}s)",
                    flush=True,
                )

    # Aggregate: per technique overall, and per technique x category
    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["technique", "category", "n", "asr"])
        for technique in techniques:
            rows = [(cat, v) for (t, cat, v) in all_rows if t == technique]
            by_cat: dict[str, list[str]] = {}
            for cat, v in rows:
                by_cat.setdefault(cat, []).append(v)
            for cat, verdicts in sorted(by_cat.items()):
                writer.writerow([technique, cat, len(verdicts), f"{attack_success_rate(verdicts):.3f}"])
            overall = [v for _, v in rows]
            writer.writerow([technique, "OVERALL", len(overall), f"{attack_success_rate(overall):.3f}"])

    print(f"\n[blackbox] wrote {SUMMARY_PATH}")
    for technique in techniques:
        overall = [v for (t, _, v) in all_rows if t == technique]
        print(f"[blackbox] {technique}: overall ASR = {attack_success_rate(overall):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N benchmark prompts")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--techniques",
        type=str,
        default=",".join(TECHNIQUES.keys()),
        help="Comma-separated subset of: " + ",".join(TECHNIQUES.keys()),
    )
    args = ap.parse_args()
    run(limit=args.limit, max_new_tokens=args.max_new_tokens, techniques=args.techniques.split(","))
