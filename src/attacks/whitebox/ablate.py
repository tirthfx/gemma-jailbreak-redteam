"""
Phase C (+ E): apply refusal-direction ablation at inference time via forward
hooks — the "simpler first version" from the plan, versus permanently editing
weight matrices. Registers a hook on every decoder layer that projects the
refusal direction out of that layer's output (the residual stream), so the
model's internal "should I refuse" signal is suppressed everywhere it would
normally be written.

`alpha` is the ablation strength (Phase E sweeps this: 0 = no-op, 1 = full
removal, >1 = over-correction). The same hook mechanism run with a *negative*
alpha adds the direction back in rather than removing it — that's exactly
Phase I's refusal-refusion guardrail (see defense/refusal_refusion.py), so the
two features share this one implementation instead of duplicating it.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.model import get_decoder_layers


def _ablation_hook(direction: torch.Tensor, alpha: float):
    """
    direction: unit vector, shape (hidden_dim,).
    alpha: 1.0 removes the full projection onto `direction`; 0 is a no-op;
           negative values *add* the direction back in (used by the defense).
    """
    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        d = direction.to(dtype=hidden.dtype, device=hidden.device)
        # proj: (..., 1) — how much of `hidden` lies along `d`, per token
        proj = torch.matmul(hidden, d).unsqueeze(-1)
        hidden = hidden - alpha * proj * d

        return (hidden,) + output[1:] if is_tuple else hidden

    return hook


@contextmanager
def ablated(model, direction: torch.Tensor, alpha: float = 1.0):
    """
    Usage:
        direction, layer_idx = load_direction()
        with ablated(model, direction, alpha=1.0):
            response = generate(model, tokenizer, prompt, device=device)

    Hooks every decoder layer for the duration of the `with` block and removes
    them afterward — the model is back to its original behavior once the block
    exits, so a single loaded model can be baseline, ablated, and re-fused
    (Phase I) across successive calls without reloading weights.
    """
    handles = []
    try:
        for layer in get_decoder_layers(model):
            handles.append(layer.register_forward_hook(_ablation_hook(direction, alpha)))
        yield model
    finally:
        for h in handles:
            h.remove()
