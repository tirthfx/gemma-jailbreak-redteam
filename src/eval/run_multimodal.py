"""
Phase H: compare image-channel ASR (harmful instruction rendered as text-in-image
+ innocuous caption) against text-channel ASR (same instruction typed directly)
on the same subset of the benchmark.

NOT YET RUN — see image_injection.py's module docstring. Run with --limit 1
first as a smoke test of the multimodal call path before the full subset.

Usage:
    python -m src.eval.run_multimodal --limit 8

Outputs:
    results/raw/multimodal.jsonl      - full transcripts, both channels (local only)
    results/summary_multimodal.csv    - ASR by channel (committed)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.attacks.multimodal.image_injection import image_injection_attack
from src.benchmark.judge import attack_success_rate, judge_response
from src.benchmark.prompts import load_benchmark
from src.model import generate, load_model

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_PATH = ROOT / "results" / "summary_multimodal.csv"


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


def run(limit: int, max_new_tokens: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[multimodal] loading model on device...", flush=True)
    t0 = time.time()
    model, tokenizer, processor, device = load_model()
    print(f"[multimodal] model loaded on {device} in {time.time() - t0:.1f}s "
          f"(processor available: {processor is not None})", flush=True)

    subset = pick_subset(limit)
    raw_path = RAW_DIR / "multimodal.jsonl"
    text_verdicts, image_verdicts = [], []

    with raw_path.open("w") as raw_f:
        for i, p in enumerate(subset, 1):
            print(f"\n[multimodal] === prompt {i}/{len(subset)}: [{p.category}] {p.id} ===", flush=True)

            text_response = generate(model, tokenizer, p.text, max_new_tokens=max_new_tokens, device=device)
            text_verdict = judge_response(text_response)
            text_verdicts.append(text_verdict)

            try:
                image_response = image_injection_attack(model, tokenizer, processor, device, p.text,
                                                          max_new_tokens=max_new_tokens)
                image_verdict = judge_response(image_response)
            except Exception as e:
                image_response = f"[ERROR: {e}]"
                image_verdict = "NON_ANSWER"
            image_verdicts.append(image_verdict)

            record = {"id": p.id, "category": p.category, "prompt": p.text,
                       "text_response": text_response, "text_verdict": text_verdict,
                       "image_response": image_response, "image_verdict": image_verdict}
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()

            print(f"[multimodal] {p.id}: text={text_verdict} image={image_verdict}", flush=True)

    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "n", "asr"])
        writer.writerow(["text", len(text_verdicts), f"{attack_success_rate(text_verdicts):.3f}"])
        writer.writerow(["image", len(image_verdicts), f"{attack_success_rate(image_verdicts):.3f}"])

    print(f"\n[multimodal] wrote {SUMMARY_PATH}")
    print(f"[multimodal] text-channel ASR:  {attack_success_rate(text_verdicts):.1%}")
    print(f"[multimodal] image-channel ASR: {attack_success_rate(image_verdicts):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    run(limit=args.limit, max_new_tokens=args.max_new_tokens)
