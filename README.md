# Red-Teaming Gemma-3-4B

A from-scratch attack-and-defense red-teaming study on `google/gemma-3-4b-it`, an
open-weight instruction-tuned LLM, run entirely locally (Apple Silicon, no cloud GPU).

**Status: complete.** All planned attack and defense phases have been run. Full
narrative write-up: [`paper/paper.md`](paper/paper.md) (technical) and
[`blog/post.md`](blog/post.md) (casual). Design rationale for the whole project:
[`docs/project_plan_explainer.pdf`](docs/project_plan_explainer.pdf).

## What this is

Red-teaming means deliberately trying to break an LLM's safety training — on a model
you fully control — in order to measure how robust it actually is, and to build
defenses against what you find. This project implements and measures five distinct
attack classes against Gemma-3-4B-it, each paired with a quantitative
attack-success-rate (ASR) measurement and a capability-retention check, plus working
countermeasures for the two most surgical attacks. It's deliberately scoped as
**attack + defense research**, not "how to jailbreak a chatbot."

## Headline results

| Result | Value |
|---|---|
| Baseline ASR (no attack) | 13.0% |
| Best black-box technique (prefix injection) | 84.8% |
| **Self-play automated search** (3 rounds) | **12.5% → 100%** |
| **Refusal-direction ablation** (alpha=0.05) | **15% → 95% ASR, capability unchanged (80% QA)** |
| **Refusal-refusion defense** (same alpha, reversed) | **95% → 15% ASR, capability still unchanged** |
| Multimodal (image vs. text channel) | 25.0% vs. 12.5% |
| GCG (gradient-based suffix search) | Did not converge — reported as a negative result, see paper §4.4 |

The full per-category, per-technique breakdown is in
[`results/REPORT.md`](results/REPORT.md), with charts in `results/charts/`.

## All phases

- ✅ **Phase A — Baseline & benchmark.** 46 self-authored harmful-request prompts
  across 8 categories, plus a three-way rule-based judge (refusal / compliance /
  non-answer). Baseline ASR: 13.0%.
- ✅ **Phase B — Black-box prompt jailbreaks.** Roleplay, base64/leetspeak
  obfuscation, multi-turn escalation, prefix injection (strongest at 84.8%).
- ✅ **Phase C/E — White-box refusal-direction ablation + strength sweep.** The
  project's flagship result: alpha=0.05 raises ASR to 95% at zero capability
  cost; the curve is sharply non-monotonic and collapses into incoherence by
  alpha≥0.5 (see the note below and paper §4.5).
- ✅ **Phase D — Capability retention.** A QA + benign-refusal battery run
  against every model variant throughout, not bolted on at the end.
- ✅ **Phase F — GCG adversarial suffixes.** Implemented, traced, and reported as
  a negative result — the gradient is verified healthy, the search simply didn't
  converge within laptop-scale compute (paper §4.4).
- ✅ **Phase G — Self-play automated jailbreak search.** 12.5% → 100% ASR after 3
  rounds, using one shared model instance alternating attacker/target roles.
- ✅ **Phase H — Multimodal injection.** Image-channel ASR (25.0%) roughly double
  text-channel (12.5%) on the same underlying instructions.
- ✅ **Phase I — Defenses.** Refusal-refusion fully neutralizes the ablation
  attack at matched scale (95% → 15% ASR, capability unchanged). The perplexity
  filter is implemented and calibrated but untested against a real attack
  instance, since Phase F never produced a converged adversarial suffix.
- ✅ **Phase J — Demo, blog, paper.** This README, the [paper](paper/paper.md),
  the [blog post](blog/post.md), and a [Streamlit demo](demo/app.py).

## Two things caught and fixed along the way (kept in on purpose)

**A judge false positive.** The first version of the judge only distinguished
REFUSAL vs. COMPLIANCE by keyword. Base64-obfuscation initially reported 100%
ASR — every response lacked a refusal keyword. Manual spot-checking showed the
model wasn't complying, it was failing to decode the request and replying with
generic filler ("Hello, world!"). A third **NON_ANSWER** category fixed this;
corrected ASR: 19.6% (80% of it non-answers). The same category later caught a
second, structurally different failure — heavily over-ablated models looping on
a repeated phrase instead of refusing or complying — via an added
repetition-detection check. See `src/benchmark/judge.py` and paper §4.6.

**A model-architecture bug.** Gemma-3's multimodal wrapper nests its actual
decoder layers as `model.model.language_model.layers`, not `model.model.layers`
— this broke every white-box attack until caught by directly introspecting the
loaded model. Fixed once, centralized in `src/model.py:get_decoder_layers()`.

Both are left in the write-up rather than quietly corrected, because catching
and documenting them is part of the actual research process, not a detour from it.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
huggingface-cli login          # your own HF account
# accept the Gemma license at https://huggingface.co/google/gemma-3-4b-it
```

## Running

```bash
python -m src.eval.run_baseline                                    # Phase A
python -m src.eval.run_blackbox                                    # Phase B
python -m src.eval.run_ablation_sweep --alphas 0,0.05,0.1,0.2,1.0   # Phase C/E
python -m src.eval.run_gcg --limit 4 --num-steps 100 --batch-size 64  # Phase F
python -m src.eval.run_selfplay --limit 8 --max-rounds 3            # Phase G
python -m src.eval.run_multimodal --limit 8                         # Phase H
python -m src.eval.run_defense --ablation-alpha 0.05 --refusion-strength 0.05  # Phase I
python scripts/rescore.py && python -m src.eval.report              # re-score + regenerate report
streamlit run demo/app.py                                           # interactive demo
```

## Repo layout

```
src/
  model.py              shared model-loading/generation helper (MPS-based, no CUDA)
  benchmark/             the harmful-prompt benchmark + judge
  attacks/                one subfolder per attack family (blackbox, whitebox, gcg, selfplay, multimodal)
  defense/                blue-team countermeasures
  eval/                   the runner script for each phase
demo/app.py              Streamlit side-by-side demo (illustrative prompts only)
paper/paper.md           full technical write-up
blog/post.md             casual write-up
results/
  raw/                    full transcripts — gitignored, never committed (may contain harmful completions)
  summary_*.csv           aggregate ASR/capability numbers — committed
  REPORT.md               auto-generated rollup of every summary CSV
  charts/                 ASR-by-technique and ablation-sweep charts
docs/
  project_plan_explainer.pdf   full plan, phase-by-phase, with design rationale
```

## Ethics

- Target is a model running entirely on this machine — never a production system or
  a service with real users.
- The benchmark prompts describe *categories* of harmful requests (in line with
  published academic benchmarks like AdvBench/HarmBench/JailbreakBench), not
  operational step-by-step instructions.
- Full model completions from testing are never committed — only aggregate
  pass/fail statistics, refusal labels, and charts are public.
- The strongest attack found (refusal ablation) is published alongside its fully
  effective defense, at the same calibrated scale.

## References

- Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language
  Models* (2023) — GCG
- Chao et al., *Jailbreaking Black Box Large Language Models in Twenty Queries*
  (2023) — PAIR
- Liu et al., *AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large
  Language Models* (2023)
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*
  (2024)
- Alon & Kamfonas, *Detecting Language Model Attacks with Perplexity* (2023)
