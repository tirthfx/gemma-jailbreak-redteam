"""
Phase I defense #2: "refusal re-fusion" — a guardrail for a model that's had its
refusal direction ablated (Phase C/E). Rather than retraining anything, this
reuses the exact same hook mechanism as ablate.py, just with the sign flipped:
instead of subtracting the refusal direction out of the residual stream, it adds
a multiple of it back in at inference time.

This is deliberately the same underlying primitive as the attack (Arditi et
al.'s direction is bidirectional — remove it to suppress refusal, add it to
reinforce it), which is part of the point being made in the paper: understanding
the mechanism well enough to break it also means understanding it well enough to
patch it, with no fine-tuning required.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from src.attacks.whitebox.ablate import ablated


@contextmanager
def refused(model, direction: torch.Tensor, strength: float = 1.0):
    """
    Usage:
        with refused(model, direction, strength=1.0):
            response = generate(model, tokenizer, prompt, device=device)

    `strength` > 0 pushes the model *toward* refusal; this is `ablated(...,
    alpha=-strength)` under the hood — see ablate.py's `_ablation_hook` for why a
    negative alpha adds the direction back in rather than removing it.
    """
    with ablated(model, direction, alpha=-strength) as m:
        yield m
