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

# Must be set before huggingface_hub is imported (by transformers below).
#
# Tried Xet high-performance transfer first (HF_XET_HIGH_PERFORMANCE=1): fast on a
# small test file (~1.4MB/s vs. ~40 bytes/s plain HTTP at that moment), but on the
# real ~8GB model shard it ran for 90 minutes and then failed with a CAS
# reconstruction error, leaving zero resumable progress on disk. Plain HTTP
# download is slower per-attempt (~560KB/s measured) but resumes properly via
# range requests when interrupted, which matters far more than raw throughput on
# this network. See scripts/download_model.sh for the retry wrapper used to
# actually pull the weights.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    return model, tokenizer, processor, device


def _to_input_ids(templated) -> torch.Tensor:
    """
    Newer `transformers` versions have `apply_chat_template(..., return_tensors="pt")`
    return a BatchEncoding (dict-like, with an "input_ids" key) rather than a bare
    tensor — this normalizes either shape to a plain input_ids tensor.
    """
    if torch.is_tensor(templated):
        return templated
    return templated["input_ids"]


def _safe_apply_template(tokenizer, messages: list[dict], **kwargs):
    """
    Gemma's chat templates have historically not supported a "system" role turn.
    Try the messages as given; if the template rejects a leading system message,
    fold it into the first user turn instead of failing every attack/eval that
    uses a persona/system prompt (roleplay attacks, Phase G self-play, etc.).
    """
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        if messages and messages[0].get("role") == "system":
            folded = dict(messages[1])
            folded["content"] = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            messages = [folded] + messages[2:]
            return tokenizer.apply_chat_template(messages, **kwargs)
        raise


def chat(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 256,
    device: torch.device | None = None,
    **generate_kwargs,
) -> str:
    """
    General multi-turn chat-template generation. `messages` is a list of
    {"role": ..., "content": ...} dicts, e.g. [{"role": "user", "content": "hi"}].
    Used directly by Phase B's multiturn escalation and Phase G's self-play loop;
    `generate()` below is a single-turn convenience wrapper around this.
    """
    if device is None:
        device = next(model.parameters()).device

    templated = _safe_apply_template(
        tokenizer, messages, add_generation_prompt=True, return_tensors="pt"
    )
    inputs = _to_input_ids(templated).to(device)

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
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return chat(model, tokenizer, messages, max_new_tokens=max_new_tokens, device=device, **generate_kwargs)


def generate_with_prefill(
    model,
    tokenizer,
    prompt: str,
    prefill: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 256,
    device: torch.device | None = None,
    **generate_kwargs,
) -> str:
    """
    Prefix-injection / "forced compliance" generation (Phase B): the assistant's
    turn is seeded with `prefill` (e.g. "Sure, here is") before generation starts,
    so the model continues from an already-compliant-sounding opening instead of
    being free to open with a refusal. Returns only the newly generated
    continuation — the caller should prepend `prefill` when logging/judging the
    full response.
    """
    if device is None:
        device = next(model.parameters()).device

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    templated_text = _safe_apply_template(
        tokenizer, messages, add_generation_prompt=True, tokenize=False
    )
    full_text = templated_text + prefill

    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

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
