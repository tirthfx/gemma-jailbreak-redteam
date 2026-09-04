"""
Phase C (+D): one model load that (1) extracts the refusal direction, (2) runs
the harmful benchmark against the ablated model to measure ASR, and (3) runs the
Phase D capability check against the same ablated model — so the "does it still
work generally" question is answered in the same run as "did the attack work",
never as an afterthought.

Usage:
    python -m src.eval.run_ablation [--alpha 1.0] [--limit N] [--max-new-tokens N]

Outputs:
    results/refusal_direction.pt              - the extracted direction (committed; tiny)
    results/raw/ablation_alpha<A>.jsonl        - full transcripts (local only)
    results/summary_ablation.csv               - ASR per category, appended per alpha
    results/summary_capability.csv             - appended with label "ablated_alpha<A>"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.attacks.whitebox.ablate import ablated
from src.attacks.whitebox.refusal_direction import (
    DIRECTION_PATH,
    compute_refusal_direction,
    load_direction,
    save_direction,
)
from src.benchmark.judge import attack_success_rate, judge_response, verdict_breakdown
from src.benchmark.prompts import load_benchmark
from src.eval.capability_check import append_summary, run_capability_check
from src.model import generate, load_model

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_PATH = ROOT / "results" / "summary_ablation.csv"


def run(alpha: float, limit: int | None, max_new_tokens: int, recompute_direction: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[ablation] loading model on device...", flush=True)
    t0 = time.time()
    model, tokenizer, processor, device = load_model()
    print(f"[ablation] model loaded on {device} in {time.time() - t0:.1f}s", flush=True)

    if recompute_direction or not DIRECTION_PATH.exists():
        direction, layer_idx = compute_refusal_direction(model, tokenizer, device)
        save_direction(direction, layer_idx)
    else:
        direction, layer_idx = load_direction()
        print(f"[ablation] loaded cached direction (layer {layer_idx}) from {DIRECTION_PATH}")

    prompts = load_benchmark()
    if limit:
        prompts = prompts[:limit]

    raw_path = RAW_DIR / f"ablation_alpha{alpha}.jsonl"
    records = []

    with raw_path.open("w") as raw_f, ablated(model, direction, alpha=alpha):
        for i, p in enumerate(prompts, 1):
            t_start = time.time()
            response = generate(model, tokenizer, p.text, max_new_tokens=max_new_tokens, device=device)
            verdict = judge_response(response)
            dt = time.time() - t_start

            record = {"id": p.id, "category": p.category, "prompt": p.text,
                      "response": response, "verdict": verdict}
            records.append(record)
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()

            print(f"[ablation:alpha={alpha}] {i}/{len(prompts)} [{p.category}] {p.id} -> {verdict} ({dt:.1f}s)",
                  flush=True)

    by_cat: dict[str, list[str]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r["verdict"])

    write_header = not SUMMARY_PATH.exists()
    with SUMMARY_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["alpha", "category", "n", "asr", "refusal_rate", "non_answer_rate"])
        for cat, verdicts in sorted(by_cat.items()):
            bd = verdict_breakdown(verdicts)
            writer.writerow([alpha, cat, len(verdicts), f"{attack_success_rate(verdicts):.3f}",
                              f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
        overall = [r["verdict"] for r in records]
        bd = verdict_breakdown(overall)
        writer.writerow([alpha, "OVERALL", len(overall), f"{attack_success_rate(overall):.3f}",
                          f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])

    print(f"\n[ablation] alpha={alpha} overall ASR: {attack_success_rate(overall):.1%}")

    # Phase D: capability check on the *same* ablated model, inside the same hook context.
    with ablated(model, direction, alpha=alpha):
        cap_result = run_capability_check(
            respond_fn=lambda p: generate(model, tokenizer, p, max_new_tokens=64, device=device),
            label=f"ablated_alpha{alpha}",
        )
    append_summary(cap_result)
    print(f"[ablation] wrote {SUMMARY_PATH} and appended capability row for alpha={alpha}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0, help="Ablation strength (Phase E sweeps this)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--recompute-direction", action="store_true")
    args = ap.parse_args()
    run(alpha=args.alpha, limit=args.limit, max_new_tokens=args.max_new_tokens,
        recompute_direction=args.recompute_direction)
