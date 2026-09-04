"""
Shared model-loading helper for the whole project.

Design constraint (see plan): this machine has 16GB unified memory and no CUDA GPU.
Two simultaneous 4B-parameter model instances in bf16 (~8GB each) would leave almost
no headroom for activations/KV cache, so every experiment in this project —
including the self-play attacker/target loop in Phase G — uses ONE shared loaded
model, never two concurrent instances. Different "roles" (attacker vs. target,
baseline vs. ablated) are realised by either swapping system prompts on the same
weights, or by loading/unloading one variant at a time.
"""

from __future__ import annotations

import os

# Must be set before huggingface_hub is imported (by transformers below). This
# network has very high latency and the default single-connection download path
# measured ~40 bytes/s to HF's CDN; the Xet high-performance transfer path measured
# ~1.4MB/s in testing (see docs/ notes) — the difference between an ~unusable
# download and one that finishes in under two hours.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"


def get_device() -> torch.device:
    """MPS on this machine (no CUDA); falls back to CPU if MPS is unavailable."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():  # pragma: no cover - not expected on this machine
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_id: str = MODEL_ID, dtype: torch.dtype = torch.bfloat16):
    """
    Loads gemma-3-4b-it plus its tokenizer/processor onto the best available device.

    Requires the user to have already run (one-time, manual — their own HF account,
    their own acceptance of Google's Gemma license):
        huggingface-cli login

    Returns (model, tokenizer, processor, device). `processor` is used for the
    multimodal (image) attacks in Phase H; text-only code can ignore it.
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        # Some checkpoints/environments may not ship a processor config; text-only
        # experiments don't need it.
        processor = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    return model, tokenizer, processor, device


def generate(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 256,
    device: torch.device | None = None,
    **generate_kwargs,
) -> str:
    """Single-turn chat-template generation helper used across eval/attack scripts."""
    if device is None:
        device = next(model.parameters()).device

    # Gemma's chat templates have historically not supported a "system" role turn.
    # Try it; if the template rejects it, fold the system content into the user
    # turn instead rather than failing every attack that uses a persona/system prompt.
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        except Exception:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
    else:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=generate_kwargs.pop("do_sample", False),
            pad_token_id=generate_kwargs.pop("pad_token_id", tokenizer.eos_token_id),
            **generate_kwargs,
        )

    new_tokens = output_ids[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
