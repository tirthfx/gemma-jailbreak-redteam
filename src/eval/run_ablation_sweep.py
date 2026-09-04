"""
Phase E: sweep the ablation strength (alpha) and record ASR + capability
retention at each point, producing the ASR-vs-capability tradeoff curve. One
model load, one direction extraction, looped over every alpha — reusing
run_ablation.py's per-alpha logic would reload the model each time (~15-20s
overhead per point), which isn't worth paying for a sweep.

Usage:
    python -m src.eval.run_ablation_sweep --alphas 0,0.25,0.5,1.0,1.5

Outputs (appended, same files run_ablation.py writes to, so a sweep and a
single-alpha run compose into one growing dataset rather than separate ones):
    results/summary_ablation.csv
    results/summary_capability.csv
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


def run(alphas: list[float], limit: int | None, max_new_tokens: int, recompute_direction: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[sweep] loading model on device...", flush=True)
    t0 = time.time()
    model, tokenizer, processor, device = load_model()
    print(f"[sweep] model loaded on {device} in {time.time() - t0:.1f}s", flush=True)

    if recompute_direction or not DIRECTION_PATH.exists():
        direction, layer_idx = compute_refusal_direction(model, tokenizer, device)
        save_direction(direction, layer_idx)
    else:
        direction, layer_idx = load_direction()
        print(f"[sweep] loaded cached direction (layer {layer_idx})")

    prompts = load_benchmark()
    if limit:
        prompts = prompts[:limit]

    write_header = not SUMMARY_PATH.exists()
    results_for_print = []

    with SUMMARY_PATH.open("a", newline="") as summary_f:
        writer = csv.writer(summary_f)
        if write_header:
            writer.writerow(["alpha", "category", "n", "asr", "refusal_rate", "non_answer_rate"])

        for alpha in alphas:
            print(f"\n[sweep] === alpha={alpha} ===", flush=True)
            raw_path = RAW_DIR / f"ablation_sweep_alpha{alpha}.jsonl"
            records = []

            with raw_path.open("w") as raw_f, ablated(model, direction, alpha=alpha):
                for i, p in enumerate(prompts, 1):
                    response = generate(model, tokenizer, p.text, max_new_tokens=max_new_tokens, device=device)
                    verdict = judge_response(response)
                    records.append({"id": p.id, "category": p.category, "verdict": verdict})
                    raw_f.write(json.dumps({**records[-1], "prompt": p.text, "response": response}) + "\n")
                    if i % 10 == 0 or i == len(prompts):
                        print(f"[sweep:alpha={alpha}] {i}/{len(prompts)} done", flush=True)

            by_cat: dict[str, list[str]] = {}
            for r in records:
                by_cat.setdefault(r["category"], []).append(r["verdict"])
            for cat, verdicts in sorted(by_cat.items()):
                bd = verdict_breakdown(verdicts)
                writer.writerow([alpha, cat, len(verdicts), f"{attack_success_rate(verdicts):.3f}",
                                  f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
            overall = [r["verdict"] for r in records]
            bd = verdict_breakdown(overall)
            asr = attack_success_rate(overall)
            writer.writerow([alpha, "OVERALL", len(overall), f"{asr:.3f}", f"{bd['refusal']:.3f}", f"{bd['non_answer']:.3f}"])
            summary_f.flush()

            with ablated(model, direction, alpha=alpha):
                cap_result = run_capability_check(
                    respond_fn=lambda p: generate(model, tokenizer, p, max_new_tokens=64, device=device),
                    label=f"ablated_alpha{alpha}",
                )
            append_summary(cap_result)

            results_for_print.append((alpha, asr, cap_result["qa_accuracy"], cap_result["benign_refusal_rate"]))
            print(f"[sweep] alpha={alpha}: ASR={asr:.1%} QA_accuracy={cap_result['qa_accuracy']:.1%} "
                  f"benign_refusal={cap_result['benign_refusal_rate']:.1%}", flush=True)

    print("\n[sweep] === summary ===")
    print(f"{'alpha':>6} {'ASR':>8} {'QA acc':>8} {'benign refusal':>15}")
    for alpha, asr, qa, br in results_for_print:
        print(f"{alpha:>6} {asr:>8.1%} {qa:>8.1%} {br:>15.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=str, default="0,0.25,0.5,1.0,1.5",
                     help="Comma-separated ablation strengths to sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--recompute-direction", action="store_true")
    args = ap.parse_args()
    run(alphas=[float(a) for a in args.alphas.split(",")], limit=args.limit,
        max_new_tokens=args.max_new_tokens, recompute_direction=args.recompute_direction)
