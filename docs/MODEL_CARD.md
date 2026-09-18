---
license: gemma
base_model: google/gemma-3-4b-it
base_model_relation: finetune
library_name: gguf
pipeline_tag: text-generation
language:
  - en
tags:
  - gemma3
  - gguf
  - llama.cpp
  - lm-studio
  - abliterated
  - refusal-ablation
  - red-teaming
  - ai-safety
  - interpretability
---

# Gemma-3-4B-it (refusal-ablated) — GGUF

A research artifact from a red-teaming study of [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it).
The model's **refusal direction** has been permanently projected out of its weights, so it no longer
refuses the requests the original model refuses. It is otherwise the same model.

> **Read this before using it.** This model has had its safety-refusal behavior surgically removed. It will
> comply with requests the original Gemma refuses, including harmful ones. It is published for
> **AI-safety research, red-teaming, and interpretability work**, not as a general-purpose assistant.
> See [Intended use & limitations](#intended-use--limitations).

| | |
|---|---|
| **Base model** | `google/gemma-3-4b-it` |
| **Method** | Refusal-direction weight orthogonalization ("abliteration"), after [Arditi et al., 2024](https://arxiv.org/abs/2406.11717) |
| **File** | `gemma-3-4b-it-ablated.gguf` — F16, 7.78 GB (7.2 GiB) |
| **Architecture** | `gemma3`, 34 layers, hidden size 2560, 131,072-token context |
| **Modalities** | **Text only.** This GGUF has no vision tower / `mmproj`, unlike the base model. |
| **Editing precision** | Full projection (alpha = 1.0), baked into the weights |
| **Author** | [@tirthfx](https://huggingface.co/tirthfx) |
| **Code, paper, results** | [github.com/tirthfx/gemma-jailbreak-redteam](https://github.com/tirthfx/gemma-jailbreak-redteam) |

## How it was made

1. **Extract the refusal direction.** Run the model on harmful and harmless prompts and take the
   difference of mean residual-stream activations at layer 20 (of 34). This gives a single unit vector `d`
   that mediates refusal.
2. **Project it out of every weight matrix that writes to the residual stream.**
   - `embed_tokens.weight`: `row' = row - (row·d) d`. Gemma-3 ties this to `lm_head`, so the unembedding is edited too.
   - every layer's `self_attn.o_proj.weight` and `mlp.down_proj.weight`: `W' = W - d (dᵀW)`.

   Removing `d` at every write point means the residual stream can never carry a component along `d`.
   No fine-tuning and no gradient steps were used, and no training data is involved beyond the prompts
   used to find the direction. The edit is a pure weight transformation.
3. **Convert to GGUF (F16)** with llama.cpp's converter.

The standalone script that applies this edit is **intentionally not published** in the
[GitHub repo](https://github.com/tirthfx/gemma-jailbreak-redteam). The repo documents the attack-and-defense
study, and the proposed defense was only verified against the runtime-hook variant, not this full-projection
edit (see [Limitations](#intended-use--limitations)).

## Evaluation

All numbers come from the project's own small benchmark and were measured on the **baked safetensors
checkpoint before GGUF conversion**. The GGUF itself was not separately re-evaluated.

| Metric | Original `gemma-3-4b-it` | This model |
|---|---|---|
| Attack success rate, 20-prompt subset (cybercrime, weapons, fraud) | 15.0% | **100%** |
| Benign-prompt refusal rate | 0% | 0% |
| Small factual/reasoning QA battery | 80% | 80% |

(For context, the original model's ASR on the full 46-prompt benchmark is 13.0%.)

**Caveats on these numbers:**

- The benchmark is **self-authored and small** (46 prompts, 8 categories, category-level requests) and
  the QA battery is a **small sanity check**, not a standard benchmark. Treat "capability unchanged" as
  "no obvious damage on a small check", not as a claim about MMLU or other standard evals. No standard
  benchmarks were run.
- Attack success is scored by a **rule-based judge** (refusal / compliance / non-answer), not a learned
  classifier or human review. "Compliance" means "did not refuse and produced an on-topic answer"; it
  says nothing about whether the answer is accurate or actually useful.
- This model uses the *full-projection* edit (alpha = 1.0). The paper's headline sweep used a different,
  runtime forward-hook intervention, where alpha = 0.05 was the sweet spot and alpha ≥ 0.5 collapsed the
  model into incoherent output. Those two are not the same operation and their alpha values are not
  comparable. See the repo's `paper/paper.md`, §4.5 and `scripts/bake_ablation.py`.

## Usage

Any GGUF-compatible runtime works. The Gemma chat template is embedded in the file.

**llama.cpp**

```bash
llama-cli -hf tirthfx/gemma-3-4b-it-ablated-GGUF -cnv
```

**Ollama**

```bash
ollama run hf.co/tirthfx/gemma-3-4b-it-ablated-GGUF
```

**LM Studio**: search for `tirthfx/gemma-3-4b-it-ablated-GGUF` in the model browser, or download the file
and place it under `~/.lmstudio/models/<publisher>/<name>/`.

Suggested sampling (from the GGUF metadata): `top_k = 64`, `top_p = 0.95`.

This is an **F16** file, which needs roughly 8+ GB of memory. To get a smaller file, quantize it yourself:

```bash
llama-quantize gemma-3-4b-it-ablated.gguf gemma-3-4b-it-ablated-Q4_K_M.gguf Q4_K_M
```

Quantized variants have not been evaluated for refusal behavior or quality.

## Intended use & limitations

**Intended for:**

- Studying how refusal is represented and removed in instruction-tuned LLMs (interpretability, mechanistic work).
- Red-teaming research, evaluating safety classifiers, guardrails, and monitoring tools against a model
  that does not refuse.
- Reproducing or extending the attack-and-defense study linked above.

**Not intended for:**

- Deployment in any user-facing product, or use by anyone who should be protected by the base model's safeguards.
- Generating content that is illegal or harmful to others.
- Being presented as a safe or aligned model. It is neither.

**Limitations:**

- Removing refusals does not make the model more capable or more accurate. It is still a 4B model and
  still hallucinates, and any harmful-sounding output it produces may simply be wrong.
- Refusal ablation is a blunt edit. It removes one direction across the whole model, so it can have side
  effects that the small QA check above would not catch.
- Only text is supported in this file.
- The paper's refusal re-fusion defense was calibrated for the runtime-hook variant (alpha = 0.05). It has
  **not** been validated against this baked, alpha = 1.0 model.

## License and terms

This model is a derivative of Gemma and is subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Those terms flow down
to you as a user of this model, including the restrictions on harmful uses. The removal of the model's
refusal behavior does not change your obligations under them. The original model is gated; you should
accept the Gemma license on the [base model page](https://huggingface.co/google/gemma-3-4b-it).

Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms.

## Responsible release notes

- The edit is **fully reproducible from public tools** (the technique is published and widely implemented),
  and the repo pairs it with measured results and a proposed defense (activation-steering "refusal
  re-fusion"), in keeping with an attack-and-defense framing.
- Raw model completions from testing are not published. Only aggregate statistics and charts are.

## Citation

```bibtex
@misc{tirthfx2026gemma3ablated,
  author       = {tirthfx},
  title        = {Red-Teaming Gemma-3-4B-it: An Attack-and-Defense Study of Open-Weight LLM Safety Training},
  year         = {2026},
  howpublished = {\url{https://github.com/tirthfx/gemma-jailbreak-redteam}}
}
```

Technique reference: Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024).
