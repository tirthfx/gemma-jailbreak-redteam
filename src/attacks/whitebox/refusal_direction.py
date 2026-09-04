"""
Phase C: extract the "refusal direction" — the single direction in the model's
residual stream that mediates refusal behavior (Arditi et al., "Refusal in
Language Models Is Mediated by a Single Direction", 2024).

Method: run a batch of harmful prompts and a batch of harmless prompts through the
model, capture the residual-stream activation at the last input token at one
middle-ish layer, and take the difference of means. That vector, normalized, is
the refusal direction — projecting it out of the residual stream (ablate.py) or
adding it back in (defense/refusal_refusion.py) is how Phases C, E, and I all
manipulate refusal behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmark.prompts import load_benchmark
from src.model import _to_input_ids, get_decoder_layers, load_model

ROOT = Path(__file__).resolve().parents[3]
DIRECTION_PATH = ROOT / "results" / "refusal_direction.pt"

# A dedicated harmless set for direction extraction — deliberately separate from
# benchmark.prompts.BENIGN_PROMPTS (which is used for capability-retention scoring
# in Phase D) so the same prompts aren't doing double duty.
HARMLESS_PROMPTS = [
    "What's a good recipe for chocolate chip cookies?",
    "Explain how photosynthesis works.",
    "Write a short poem about the ocean.",
    "What are some tips for staying productive while studying?",
    "Summarize the plot of Romeo and Juliet.",
    "How do I set up a Python virtual environment?",
    "What's the difference between weather and climate?",
    "Give me three ideas for a birthday gift for my mom.",
    "Explain how compound interest works.",
    "What are the main causes of the French Revolution?",
    "How does a car engine convert fuel into motion?",
    "Write a friendly email asking a colleague for a status update.",
    "What's the best way to learn a new language?",
    "Explain the water cycle to a 10-year-old.",
    "What are some good exercises for lower back pain?",
    "How do vaccines work?",
    "Recommend a beginner-friendly houseplant.",
    "What's the difference between a stock and a bond?",
    "Explain how a rainbow forms.",
    "Give me a simple explanation of how the internet works.",
]


def _last_token_activation(model, tokenizer, device, prompt: str, layer_idx: int) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = _to_input_ids(templated).to(device)
    with torch.no_grad():
        out = model(inputs, output_hidden_states=True)
    # hidden_states: tuple of (num_layers + 1) tensors, each (1, seq_len, hidden_dim);
    # index 0 is the embedding output, index i is the output of decoder layer i-1.
    return out.hidden_states[layer_idx][0, -1, :].float().cpu()


def collect_activations(model, tokenizer, device, prompts: list[str], layer_idx: int) -> torch.Tensor:
    return torch.stack([_last_token_activation(model, tokenizer, device, p, layer_idx) for p in prompts])


def compute_refusal_direction(
    model,
    tokenizer,
    device,
    harmful_prompts: list[str] | None = None,
    harmless_prompts: list[str] | None = None,
    layer_frac: float = 0.6,
    n_harmful: int = 32,
) -> tuple[torch.Tensor, int]:
    """
    Returns (unit_direction, layer_idx). layer_idx indexes into hidden_states as
    returned by the model (see _last_token_activation) — 0.6 of the way through
    the network by default, following the original paper's finding that
    middle-to-late layers carry the clearest refusal signal.
    """
    if harmful_prompts is None:
        harmful_prompts = [p.text for p in load_benchmark()[:n_harmful]]
    if harmless_prompts is None:
        harmless_prompts = HARMLESS_PROMPTS

    num_layers = len(get_decoder_layers(model))
    layer_idx = max(1, min(num_layers, int(round(layer_frac * num_layers))))

    print(f"[refusal_direction] extracting at layer {layer_idx}/{num_layers}, "
          f"{len(harmful_prompts)} harmful vs {len(harmless_prompts)} harmless prompts", flush=True)

    harmful_acts = collect_activations(model, tokenizer, device, harmful_prompts, layer_idx)
    harmless_acts = collect_activations(model, tokenizer, device, harmless_prompts, layer_idx)

    direction = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)
    direction = direction / direction.norm()
    return direction, layer_idx


def save_direction(direction: torch.Tensor, layer_idx: int, path: Path = DIRECTION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": direction, "layer_idx": layer_idx}, path)
    print(f"[refusal_direction] saved to {path}")


def load_direction(path: Path = DIRECTION_PATH) -> tuple[torch.Tensor, int]:
    obj = torch.load(path, weights_only=True)
    return obj["direction"], obj["layer_idx"]


if __name__ == "__main__":
    model, tokenizer, processor, device = load_model()
    direction, layer_idx = compute_refusal_direction(model, tokenizer, device)
    save_direction(direction, layer_idx)
    print(f"[refusal_direction] direction norm (should be ~1.0 after normalization): {direction.norm():.4f}")
