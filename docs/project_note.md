---
project: Gemma Red Team
type: moc
status: shipped
updated: 2026-09-15
tags: [python, pytorch, llm, adversarial-ml, security, red-teaming, gemma]
---

# Gemma Red Team

From-scratch attack-and-defense red-teaming study on `google/gemma-3-4b-it`, an open-weight instruction-tuned LLM, run entirely locally (Apple Silicon, no cloud GPU). Five independent attack classes measured for attack-success-rate (ASR) and capability retention, plus a working defense for the two most surgical attacks.

**Stack:** Python, PyTorch, Hugging Face Transformers, llama.cpp
**Model:** google/gemma-3-4b-it — both safetensors and a GGUF export loaded into LM Studio locally
**Repo:** [github.com/tirthfx/gemma-jailbreak-redteam](https://github.com/tirthfx/gemma-jailbreak-redteam) (public)

## Headline results
| Result | Value |
|---|---|
| Baseline ASR (no attack) | 13.0% |
| Best black-box technique (prefix injection) | 84.8% |
| Self-play automated search (3 rounds) | 12.5% → 100% |
| Refusal-direction ablation (α=0.05) | 15% → 95% ASR, capability unchanged (80% QA) |
| Refusal-refusion defense (same α, reversed) | 95% → 15% ASR, capability still unchanged |
| Multimodal (image vs. text channel) | 25.0% vs. 12.5% |

## Status
**Shipped (2026-09-15).** All planned attack and defense phases complete; repo is public with full write-up (`paper/paper.md`, `blog/post.md`, `docs/project_plan_explainer.pdf`). No model weights committed to git — `scripts/download_model.sh` reproduces the base model. The ablated (alpha=1.0, full-projection) GGUF is published on Hugging Face: [tirthfx/gemma-3-4b-it-ablated-GGUF](https://huggingface.co/tirthfx/gemma-3-4b-it-ablated-GGUF) (public, F16, text-only, released 2026-09-19). The LM Studio copy was deleted after upload; the standalone bake script stays local-only (gitignored) because the defense was never verified against full projection.

Announced on LinkedIn (2026-09-15) with a 6-slide carousel styled as a redacted field-report dossier — real ASR numbers, no invented data.
