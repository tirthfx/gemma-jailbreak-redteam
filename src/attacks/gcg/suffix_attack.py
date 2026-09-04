"""
Phase F: GCG (Greedy Coordinate Gradient) adversarial suffix attack, after
Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language
Models" (2023).

Idea: append a short suffix of tokens to a harmful prompt, and iteratively search
for the suffix (via gradients) that maximizes the model's likelihood of starting
its reply with a forced affirmative target (e.g. "Sure, here is..."). Each step:
  1. Compute the gradient of the target loss w.r.t. a one-hot encoding of each
     suffix token (via the embedding matrix, so it's differentiable).
  2. For each suffix position, take the top-k token substitutions the gradient
     says would most reduce the loss.
  3. Randomly build a batch of candidate suffixes, each swapping one position for
     one of its top-k candidates.
  4. Run one batched forward pass (no grad) over all candidates, keep whichever
     has the lowest loss, and use it as the new suffix for the next step.

NOT YET RUN END-TO-END — writing this while the model is busy with Phase B.
Needs a smoke test (small num_steps/batch_size) before trusting it on the full
scoped-down benchmark subset. See eval/run_gcg.py.

Scoped down for a 16GB-RAM, no-CUDA laptop (see project plan): short suffix,
modest step count and batch size, run on a subset of the benchmark rather than
all of it. All batching happens as one forward call with a batch dimension, not
`batch_size` separate calls, which is what keeps this remotely tractable on MPS.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

DEFAULT_SUFFIX_LEN = 20
DEFAULT_NUM_STEPS = 100
DEFAULT_TOPK = 100
DEFAULT_BATCH_SIZE = 24


def default_target(prompt: str) -> str:
    """
    A generic forced-affirmative continuation to optimize toward. Deliberately
    generic (not tailored per-prompt) so the same suffix-search code works for
    every benchmark category without a hand-written target per prompt.
    """
    return "Sure, here is how to do that."


def _split_template_around_suffix(tokenizer, prompt: str):
    """
    Splits the chat-templated text into (prefix_text, postfix_text) around a
    unique placeholder inserted where the suffix goes, so the suffix tokens can
    be optimized independently of however the tokenizer's chat template wraps
    the turn (BOS, role tags, etc.).
    """
    placeholder = "\x00SUFFIX\x00"
    messages = [{"role": "user", "content": f"{prompt} {placeholder}"}]
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if placeholder not in templated:
        raise RuntimeError("chat template did not preserve the suffix placeholder verbatim")
    prefix_text, postfix_text = templated.split(placeholder, 1)
    return prefix_text, postfix_text


def _init_suffix_ids(tokenizer, suffix_len: int, device) -> torch.Tensor:
    bang_id = tokenizer.encode("!", add_special_tokens=False)
    bang_id = bang_id[0] if bang_id else tokenizer.unk_token_id
    return torch.tensor([bang_id] * suffix_len, device=device, dtype=torch.long)


def _build_sequence(tokenizer, device, prefix_text: str, suffix_ids: torch.Tensor,
                     postfix_text: str, target_text: str):
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    postfix_ids = tokenizer(postfix_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

    input_ids = torch.cat([prefix_ids, suffix_ids, postfix_ids, target_ids])
    suffix_slice = slice(len(prefix_ids), len(prefix_ids) + len(suffix_ids))
    target_slice = slice(len(input_ids) - len(target_ids), len(input_ids))
    return input_ids, suffix_slice, target_slice


def _suffix_gradient(model, input_ids: torch.Tensor, suffix_slice: slice, target_slice: slice) -> torch.Tensor:
    """Returns d(loss)/d(one_hot_suffix), shape (suffix_len, vocab_size), float32."""
    embed_layer = model.get_input_embeddings()
    embed_weights = embed_layer.weight  # (vocab_size, hidden_dim), model dtype

    vocab_size = embed_weights.shape[0]
    suffix_len = suffix_slice.stop - suffix_slice.start

    one_hot = torch.zeros(suffix_len, vocab_size, device=input_ids.device, dtype=torch.float32)
    one_hot.scatter_(1, input_ids[suffix_slice].unsqueeze(1), 1.0)
    one_hot.requires_grad_()

    suffix_embeds = (one_hot @ embed_weights.float()).to(embed_weights.dtype)  # (suffix_len, hidden_dim)

    full_embeds = embed_layer(input_ids.unsqueeze(0)).detach().clone()
    full_embeds = torch.cat([
        full_embeds[:, :suffix_slice.start, :],
        suffix_embeds.unsqueeze(0),
        full_embeds[:, suffix_slice.stop:, :],
    ], dim=1)

    logits = model(inputs_embeds=full_embeds).logits  # (1, seq_len, vocab)
    # predicting target_slice tokens means using logits at positions [target_slice - 1]
    shifted_logits = logits[0, target_slice.start - 1: target_slice.stop - 1, :]
    targets = input_ids[target_slice]
    loss = F.cross_entropy(shifted_logits.float(), targets)
    loss.backward()

    grad = one_hot.grad.detach().clone()
    return grad, loss.item()


@torch.no_grad()
def _batch_loss(model, input_ids: torch.Tensor, suffix_slice: slice, target_slice: slice,
                 candidate_suffixes: torch.Tensor) -> torch.Tensor:
    """candidate_suffixes: (batch, suffix_len). Returns per-candidate loss, shape (batch,)."""
    batch_size = candidate_suffixes.shape[0]
    batch_input_ids = input_ids.unsqueeze(0).repeat(batch_size, 1).clone()
    batch_input_ids[:, suffix_slice] = candidate_suffixes

    logits = model(input_ids=batch_input_ids).logits  # (batch, seq_len, vocab)
    shifted_logits = logits[:, target_slice.start - 1: target_slice.stop - 1, :]
    targets = input_ids[target_slice].unsqueeze(0).repeat(batch_size, 1)

    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).view(batch_size, -1).mean(dim=1)
    return losses


def run_gcg(
    model,
    tokenizer,
    device,
    prompt: str,
    target: str | None = None,
    suffix_len: int = DEFAULT_SUFFIX_LEN,
    num_steps: int = DEFAULT_NUM_STEPS,
    topk: int = DEFAULT_TOPK,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
) -> dict:
    """
    Returns {"suffix_text", "suffix_ids", "final_loss", "loss_history", "prompt_with_suffix"}.
    `prompt_with_suffix` is what a caller should feed through the normal
    generate()/chat() path (with the standard chat template) to actually see
    whether the found suffix produces a compliant full response.
    """
    rng = random.Random(seed)
    target = target or default_target(prompt)

    prefix_text, postfix_text = _split_template_around_suffix(tokenizer, prompt)
    suffix_ids = _init_suffix_ids(tokenizer, suffix_len, device)

    loss_history = []
    best_suffix_ids = suffix_ids.clone()
    best_loss = float("inf")

    for step in range(num_steps):
        input_ids, suffix_slice, target_slice = _build_sequence(
            tokenizer, device, prefix_text, suffix_ids, postfix_text, target
        )

        grad, cur_loss = _suffix_gradient(model, input_ids, suffix_slice, target_slice)
        loss_history.append(cur_loss)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_suffix_ids = suffix_ids.clone()

        # Top-k candidate token ids per suffix position (most loss-reducing direction).
        top_indices = (-grad).topk(topk, dim=1).indices  # (suffix_len, topk)

        candidates = suffix_ids.unsqueeze(0).repeat(batch_size, 1).clone()
        for b in range(batch_size):
            pos = rng.randrange(suffix_len)
            new_token = top_indices[pos, rng.randrange(topk)].item()
            candidates[b, pos] = new_token

        losses = _batch_loss(model, input_ids, suffix_slice, target_slice, candidates)
        min_idx = torch.argmin(losses).item()
        if losses[min_idx].item() < cur_loss:
            suffix_ids = candidates[min_idx]

        print(f"[gcg] step {step + 1}/{num_steps} loss={cur_loss:.4f} best={best_loss:.4f}", flush=True)

    suffix_text = tokenizer.decode(best_suffix_ids, skip_special_tokens=True)
    return {
        "suffix_text": suffix_text,
        "suffix_ids": best_suffix_ids.tolist(),
        "final_loss": best_loss,
        "loss_history": loss_history,
        "prompt_with_suffix": f"{prompt} {suffix_text}",
    }
