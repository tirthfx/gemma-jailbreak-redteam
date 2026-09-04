"""
Phase I: measure both defenses against the attacks they target.

  1. Perplexity filter vs. GCG suffixes (Phase F) — how many GCG-suffixed
     prompts get flagged, at a threshold calibrated to a known false-positive
     rate on benign prompts.
  2. Refusal re-fusion vs. the ablated model (Phase C/E) — does adding the
     refusal direction back in bring ASR back down, and does the capability
     check confirm it isn't just refusing everything?

Requires results/raw/gcg.jsonl (from run_gcg.py) and results/refusal_direction.pt
(from run_ablation.py) to exist — run those first.

Usage:
    python -m src.eval.run_defense [--refusion-strength 1.0]

Outputs:
    results/summary_defense.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.attacks.whitebox.ablate import ablated
from src.attacks.whitebox.refusal_direction import DIRECTION_PATH, load_direction
from src.benchmark.judge import attack_success_rate, judge_response
from src.benchmark.prompts import BENIGN_PROMPTS, load_benchmark
from src.defense.perplexity_filter import calibrate_threshold, is_suspicious
from src.defense.refusal_refusion import refused
from src.eval.capability_check import run_capability_check
from src.model import generate, load_model

ROOT = Path(__file__).resolve().parents[2]
GCG_RAW = ROOT / "results" / "raw" / "gcg.jsonl"
SUMMARY_PATH = ROOT / "results" / "summary_defense.csv"


def run_perplexity_defense(model, tokenizer, device) -> dict:
    if not GCG_RAW.exists():
        print(f"[defense] {GCG_RAW} not found — run eval/run_gcg.py first. Skipping perplexity defense.")
        return {}

    threshold = calibrate_threshold(model, tokenizer, device, percentile=0.99)
    print(f"[defense] perplexity threshold (99th pct of benign prompts): {threshold:.1f}")

    # False-positive rate: how often benign prompts themselves get flagged.
    fp = sum(1 for p in BENIGN_PROMPTS if is_suspicious(model, tokenizer, p.text, device, threshold)[0])
    fp_rate = fp / len(BENIGN_PROMPTS)

    rows = [json.loads(l) for l in GCG_RAW.open()]
    flagged = 0
    for r in rows:
        adversarial_text = f"{r['prompt']} {r['suffix']}"
        is_flagged, ppl = is_suspicious(model, tokenizer, adversarial_text, device, threshold)
        flagged += int(is_flagged)
        print(f"[defense:perplexity] {r['id']} ppl={ppl:.1f} flagged={is_flagged}")
    catch_rate = flagged / len(rows) if rows else 0.0

    print(f"[defense] perplexity filter: catch_rate={catch_rate:.1%} on GCG suffixes, "
          f"false_positive_rate={fp_rate:.1%} on benign prompts")
    return {"threshold": threshold, "catch_rate": catch_rate, "benign_false_positive_rate": fp_rate,
             "n_gcg": len(rows)}


def run_refusion_defense(model, tokenizer, device, ablation_alpha: float, strength: float,
                          limit: int | None) -> dict:
    """
    Defends an *ablated* model (attack: ablation_alpha) by adding the refusal
    direction back in on top (defense: strength) — nesting ablated() and
    refused() together, since the whole point of this defense is to patch a
    model that's already had refusal suppressed, not to test refusal-steering
    on an untouched model (an earlier version of this function did the latter
    by mistake — refused() alone, with no ablated() underneath it).
    """
    if not DIRECTION_PATH.exists():
        print(f"[defense] {DIRECTION_PATH} not found — run eval/run_ablation.py first. Skipping refusion defense.")
        return {}

    direction, layer_idx = load_direction()
    prompts = load_benchmark()
    if limit:
        prompts = prompts[:limit]

    verdicts = []
    with ablated(model, direction, alpha=ablation_alpha), refused(model, direction, strength=strength):
        for p in prompts:
            response = generate(model, tokenizer, p.text, max_new_tokens=256, device=device)
            verdicts.append(judge_response(response))
        cap_result = run_capability_check(
            respond_fn=lambda p: generate(model, tokenizer, p, max_new_tokens=64, device=device),
            label=f"ablated_alpha{ablation_alpha}_refused_strength{strength}",
        )

    asr = attack_success_rate(verdicts)
    print(f"[defense] refusal re-fusion (ablation_alpha={ablation_alpha}, strength={strength}): "
          f"ASR on ablated+guardrailed model = {asr:.1%}, "
          f"QA accuracy = {cap_result['qa_accuracy']:.1%}, benign refusal rate = {cap_result['benign_refusal_rate']:.1%}")
    return {"ablation_alpha": ablation_alpha, "strength": strength, "asr": asr,
            "qa_accuracy": cap_result["qa_accuracy"], "benign_refusal_rate": cap_result["benign_refusal_rate"]}


def run(refusion_strength: float, ablation_alpha: float, limit: int | None) -> None:
    model, tokenizer, processor, device = load_model()

    ppl_result = run_perplexity_defense(model, tokenizer, device)
    refusion_result = run_refusion_defense(model, tokenizer, device, ablation_alpha, refusion_strength, limit)

    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["defense", "metric", "value"])
        for k, v in ppl_result.items():
            writer.writerow(["perplexity_filter", k, v])
        for k, v in refusion_result.items():
            writer.writerow(["refusal_refusion", k, v])
    print(f"\n[defense] wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusion-strength", type=float, default=1.0)
    ap.add_argument("--ablation-alpha", type=float, default=1.0,
                     help="Ablation strength to defend against — should match whichever alpha "
                          "the Phase E sweep found to actually produce a coherent jailbroken "
                          "model, not necessarily 1.0.")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    run(refusion_strength=args.refusion_strength, ablation_alpha=args.ablation_alpha, limit=args.limit)
