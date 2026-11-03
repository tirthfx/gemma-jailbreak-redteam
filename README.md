# Red-Teaming Gemma-3-4B-it

An attack-and-defense red-teaming study of [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it),
an open-weight, instruction-tuned, vision-capable language model. Everything was run locally on Apple Silicon
(PyTorch MPS, bfloat16) with no cloud GPU and no external API.

**Status: complete.** Full write-up: [`paper/paper.md`](paper/paper.md) (technical) and
[`blog/post.md`](blog/post.md) (non-technical). Design rationale and phase-by-phase plan:
[`docs/project_plan_explainer.pdf`](docs/project_plan_explainer.pdf).

---

## Overview

Safety-tuned language models are trained to refuse harmful requests. Red-teaming tests how robust that
training actually is, and whether defenses hold up once an attack is known. This project:

1. Builds a small, self-authored benchmark of harmful-request *categories* and an automated judge.
2. Measures the model's baseline attack success rate (ASR) with no attack applied.
3. Implements and measures five attack classes against the model:
   prompt-based jailbreaks, self-play automated search, GCG gradient-based adversarial suffixes,
   refusal-direction ablation, and multimodal (image-channel) injection.
4. Checks capability retention after every intervention, so an attack that merely breaks the model is
   distinguishable from one that targets refusal specifically.
5. Builds and evaluates two defenses for the two most targeted attacks: a perplexity filter for
   gradient-optimized suffixes, and an activation-steering "refusal re-fusion" guardrail.

The project is scoped as **AI-safety research**. Its target is a model the author runs locally. No
production system or third-party service was tested.

## Key results

Sample sizes matter here. The attack-specific conditions were run on 8 or 20 prompts, so one prompt moves the
result by 5 to 12.5 percentage points. Treat these as indicative, not precise estimates.

| Result | Value | Sample |
|---|---|---|
| Baseline ASR, no attack | **13.0%** | 46 prompts |
| Strongest black-box technique (prefix injection) | **84.8%** | 46 prompts |
| Self-play automated search | 12.5% single-shot → **100%** after 3 rounds | 8 prompts |
| Refusal-direction ablation (α = 0.05) | 15% → **95%** ASR; QA accuracy unchanged at 80% | 20 prompts; 10 QA items |
| Refusal re-fusion defense, same scale | 95% → **15%** ASR; QA accuracy unchanged at 80% | 20 prompts; 10 QA items |
| Multimodal: image channel vs. text channel | 25.0% vs. 12.5% | 8 prompts each (2 vs. 1 successes) |
| GCG adversarial suffixes | **No converged suffix** (reported as a negative result) | 2 runs, 9 and 13 steps |
| Perplexity filter | Calibrated (0% false positives on benign prompts); **not tested against a real attack** | — |

### Refusal-direction ablation: strength sweep

The ablation strength (α) produces a sharp, non-monotonic curve rather than a smooth tradeoff.

| α | ASR | QA accuracy | Benign refusal rate | Non-answer rate |
|---|---|---|---|---|
| 0.0 (no edit) | 15% | 80% | 0% | 0% |
| 0.05 | **95%** | 80% | 0% | 0% |
| 0.1 | 95% | 70% | 0% | 5% |
| 0.2 | 80% | 80% | 0% | 20% |
| 0.3 | 40% | 60% | 0% | 60% |
| 0.5 | 0% | 0% | 0% | 100% |
| 1.0 | 10% | 0% | 0% | 90% |
| 1.5 | 0% | 0% | 0% | 100% |

ASR is measured on a 20-prompt subset (cybercrime, fraud, weapons). From α = 0.3 upward the outputs are
increasingly degenerate, mostly repetitive text that the judge scores as NON_ANSWER, so those rows do not measure
real refusal or real compliance. See [§ Methodology](#methodology) and paper §4.5.

### Black-box prompt attacks (Phase B)

| Technique | ASR (46 prompts) | Notes |
|---|---|---|
| Prefix injection | 84.8% | Strongest black-box technique |
| Roleplay / persona framing | 41.3% | |
| Obfuscation (leetspeak) | 32.6% | |
| Obfuscation (base64) | 19.6% | 80% of responses were NON_ANSWER; see the judge note below |
| Multi-turn escalation | 10.9% | At baseline; reported as measured |

Per-category breakdowns and charts are in [`results/REPORT.md`](results/REPORT.md) and
[`results/charts/`](results/charts/).

## Methodology

### Target and environment

- **Model:** `google/gemma-3-4b-it`, loaded in bfloat16 on the Apple MPS backend.
- **Why this model:** open weights (so white-box analysis is possible), small enough to run on a
  16 GB laptop, vision-capable (enables the multimodal channel test), and widely deployed.
- **Why no quantization:** `bitsandbytes` does not support Apple's Metal backend.
- **Why one shared model instance:** two full copies do not fit in 16 GB with room for activations, so the
  self-play attack alternates attacker and target roles on a single loaded model.

### Benchmark and judge

- **Benchmark:** 46 self-written harmful-request prompts across 8 categories (cybercrime, weapons,
  fraud/scams, misinformation, illegal drugs, privacy violation, hateful content, extremism), plus 5 benign
  prompts as a sanity check. Prompts are written at the level of *category-level* requests, in line with
  published academic benchmarks such as AdvBench, HarmBench and JailbreakBench. No prompt text is
  reproduced in this README.
- **Judge:** a rule-based, three-way classifier: **REFUSAL**, **COMPLIANCE**, or **NON_ANSWER**
  (short, generic, or degenerate output). See `src/benchmark/judge.py`.
- **Metric:** ASR = fraction of prompts judged COMPLIANCE.

### Capability retention

Every model variant is also run on:

- a 10-item factual and arithmetic QA battery, scored by expected-substring match, and
- the 5 benign prompts, scored by refusal rate.

The goal is to show that an intervention made the model comply *without* damaging its general
behaviour. Both checks are small sanity checks, not substitutes for standard benchmarks. No MMLU-style
evaluation was run.

### Attacks

| Phase | Attack | Category | What it tests |
|---|---|---|---|
| B | Roleplay, obfuscation, multi-turn escalation, prefix injection | Black-box prompts | Whether manual prompt framing changes refusal behaviour |
| C / E | Refusal-direction ablation with strength sweep | White-box | Whether one internal direction mediates refusal, and how the edit strength trades off against capability |
| F | GCG adversarial suffix search | Gradient-based | Whether gradient search can find a suffix that elicits compliance (negative result) |
| G | Self-play automated search | Agentic | Whether the model, rewriting its own refused attempts, finds a working prompt |
| H | Multimodal image injection | Input channel | Whether text-trained safety transfers to rendered-text images |

**Refusal-direction ablation** is based on [Arditi et al., 2024](https://arxiv.org/abs/2406.11717). It
extracts a single activation direction associated with refusal and intervenes on it during the forward pass,
then measures how ASR and capability change with intervention strength. The method is described at this
level only. Implementation details are in `src/attacks/whitebox/`, and the paper discusses the
implementation-specific effects in §4.5.

### Defenses

| Phase | Defense | Target | Result |
|---|---|---|---|
| I | Perplexity filter (`src/defense/perplexity_filter.py`) | GCG-style suffixes | Threshold set at the 99th percentile of benign-prompt perplexity (857.6). 0% benign false positives. Catch rate not measurable, because GCG produced no converged suffix. |
| I | Refusal re-fusion guardrail (`src/defense/refusal_refusion.py`) | Refusal-direction ablation | At the same scale as the attack (α = 0.05), ASR returns from 95% to 15%, the unablated baseline. QA accuracy stays at 80%. |

Calibration matters. An earlier re-fusion attempt at strength 1.0 over-corrected into the same degenerate
output as an over-ablated model. The defense works only at a matched scale.

### Notes on process

Two issues were caught by spot-checking and are documented rather than silently corrected:

- **Judge false positives.** A keyword-only judge reported 100% ASR for the encoding-obfuscation attack.
  Manual inspection showed the model was failing to decode the request and returning filler. The
  NON_ANSWER category fixed this, and the corrected ASR is 19.6%. A later repetition check caught
  over-ablated models looping on a phrase. See paper §4.6.
- **Architecture bug.** Gemma-3's multimodal wrapper nests decoder layers under
  `model.model.language_model.layers`, not `model.model.layers`. Every white-box attack failed until this
  was found by introspecting the loaded model. The lookup is centralized in `src/model.py`.

### Known limitations

- The judge is rule-based. It is not a learned classifier and was not validated against human labels
  on every result.
- The benchmark is small (46 prompts). The self-play, GCG and multimodal experiments used 8-prompt
  subsets, so their uncertainty is wide.
- The QA battery has 10 items. "Capability unchanged" means no obvious damage on a small check.
- Ablation findings are specific to this implementation. The paper notes that a different application
  strategy, such as direct weight editing or hooking a subset of layers, may behave differently.
- Results are for one model. They do not establish how other models behave.

## Setup

Requirements: Apple Silicon Mac (16 GB or more recommended), Python 3.12, and a Hugging Face account.
Gemma is a gated model, so you must accept its license on the
[model page](https://huggingface.co/google/gemma-3-4b-it) under your own account.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
hf auth login                  # or `huggingface-cli login` on older huggingface_hub versions
./scripts/download_model.sh    # retrying download of google/gemma-3-4b-it
```

Model weights are not included in this repository. `results/raw/` is gitignored and is created locally
when experiments run.

## Usage

Each phase is a module under `src/eval/`. Run them from the repo root, in roughly this order:

```bash
python -m src.eval.run_baseline                                          # Phase A
python -m src.eval.run_blackbox                                          # Phase B
python -m src.eval.run_ablation_sweep --alphas 0,0.05,0.1,0.2,1.0        # Phases C / E
python -m src.eval.run_gcg --limit 4 --num-steps 100 --batch-size 64     # Phase F (writes results/raw/gcg.jsonl)
python -m src.eval.run_selfplay --limit 8 --max-rounds 3                 # Phase G
python -m src.eval.run_multimodal --limit 8                              # Phase H
python -m src.eval.run_defense --ablation-alpha 0.05 --refusion-strength 0.05   # Phase I (needs the GCG output)
python scripts/rescore.py && python -m src.eval.report                   # re-score and regenerate results/REPORT.md
streamlit run demo/app.py                                                # interactive demo
```

The demo shows side-by-side responses using a handful of benign, illustrative prompts only.

## Project structure

```
.
├── README.md
├── requirements.txt
├── paper/paper.md                 full technical write-up
├── blog/post.md                   non-technical summary
├── demo/app.py                    Streamlit demo (benign prompts only)
├── docs/
│   ├── MODEL_CARD.md              model card for the released GGUF artifact
│   └── project_plan_explainer.pdf design rationale and phase plan
├── scripts/
│   ├── download_model.sh          retrying model download
│   ├── rescore.py                 re-score saved raw transcripts with the current judge
│   └── run_remaining_phases*.sh   orchestration for the later phases
├── src/
│   ├── model.py                   model loading, generation, decoder-layer lookup
│   ├── benchmark/                 prompt set and three-way judge
│   ├── attacks/
│   │   ├── blackbox/              roleplay, obfuscation, multi-turn, prefix injection
│   │   ├── whitebox/              refusal-direction extraction and ablation
│   │   ├── gcg/                   gradient-based suffix search
│   │   ├── selfplay/              automated attacker loop
│   │   └── multimodal/            image-channel injection
│   ├── defense/                   perplexity filter, refusal re-fusion
│   └── eval/                      one runner per phase, plus capability check and report
└── results/
    ├── summary_*.csv              aggregate numbers (committed)
    ├── REPORT.md                  generated roll-up of every summary CSV
    ├── charts/                    ASR and ablation-sweep charts
    └── refusal_direction.pt       the extracted direction vector used by the ablation and defense scripts
```

`results/raw/` (full transcripts, gitignored) is intentionally absent from the repository.

## Released research artifact

A refusal-ablated GGUF build of the model was published on Hugging Face:
[`tirthfx/gemma-3-4b-it-ablated-GGUF`](https://huggingface.co/tirthfx/gemma-3-4b-it-ablated-GGUF).
Its model card is in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

- It uses a different intervention from the runtime-hook ablation in the paper. Its alpha values are not
  comparable to the sweep above.
- The model card reports results on the baked safetensors checkpoint, not the GGUF file. The GGUF was not
  separately re-evaluated.
- The refusal re-fusion defense was **not** validated against this artifact.
- The script that produced the artifact is intentionally not published in this repository.

The artifact is a safety-relevant research object. Do not deploy it or present it as aligned.

## Responsible use and ethics

- **Target and scope.** All experiments ran on a model on the author's own machine. No production system, service
  with real users, or third-party target was touched.
- **No harmful content in the repository.** The benchmark is written at category level. This README and the
  repository do not contain prompt text, jailbreak templates, or model outputs. Full completions produced during
  testing are written to `results/raw/`, which is gitignored and never committed. Only aggregate statistics,
  judge labels, and charts are public.
- **Attack paired with measurement and defense.** Every attack is reported with its measured effect, a
  capability check, and, where feasible, a defense. Negative results, including GCG, are reported as well.
- **Purpose of publication.** Methods are drawn from published work and cited below. The results are shared so
  others can evaluate and harden similar models.
- **Terms.** Use of Gemma model weights, including any derivative, is governed by the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- **Do not** use this work to produce or distribute harmful content, to attack systems you do not own
  or have authorization to test, or to build products that depend on a de-refused model.

## Licence

No open-source licence has been granted for the code in this repository yet, so the code is
**all rights reserved** by the author. Model weights (including the released GGUF) are subject to the
Gemma Terms of Use described above.

## References

- Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models* (2023). GCG.
- Chao et al., *Jailbreaking Black Box Large Language Models in Twenty Queries* (2023). PAIR.
- Liu et al., *AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models* (2023).
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024).
- Alon & Kamfonas, *Detecting Language Model Attacks with Perplexity* (2023).
