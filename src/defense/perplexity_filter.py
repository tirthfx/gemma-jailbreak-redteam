"""
Phase I defense #1: flag GCG-style adversarial suffixes before they reach the
model, after Alon & Kamfonas, "Detecting Language Model Attacks with
Perplexity" (2023).

Why this works: GCG's search has no fluency constraint, so the suffixes it finds
are typically statistically bizarre token sequences — high perplexity under any
reasonable language model, including the target model itself. Natural language
(even a weirdly-phrased jailbreak prompt) is comparatively low-perplexity. This
reuses judge.response_perplexity — the same scoring primitive, just pointed at
the *input* here instead of judging an output.

The threshold is calibrated per-model/run rather than hardcoded: compute
perplexity over a batch of known-benign prompts and set the threshold at a high
percentile of that distribution, so a false-positive rate on ordinary text is
known and controllable rather than guessed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.judge import response_perplexity
from src.benchmark.prompts import BENIGN_PROMPTS


def calibrate_threshold(model, tokenizer, device, percentile: float = 0.99,
                         calibration_texts: list[str] | None = None) -> float:
    """
    Returns a perplexity threshold such that `percentile` fraction of benign
    calibration text falls below it (i.e. ~1% false-positive rate at the
    default 0.99). Uses BENIGN_PROMPTS by default.
    """
    texts = calibration_texts or [p.text for p in BENIGN_PROMPTS]
    perplexities = sorted(response_perplexity(model, tokenizer, t, device) for t in texts)
    idx = min(len(perplexities) - 1, int(round(percentile * (len(perplexities) - 1))))
    return perplexities[idx]


def is_suspicious(model, tokenizer, text: str, device, threshold: float) -> tuple[bool, float]:
    """Returns (flagged, perplexity)."""
    ppl = response_perplexity(model, tokenizer, text, device)
    return ppl > threshold, ppl
