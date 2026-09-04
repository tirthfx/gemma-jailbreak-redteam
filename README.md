# Red-Teaming Gemma-3-4B

A from-scratch attack-and-defense red-teaming study on `google/gemma-3-4b-it`, an
open-weight instruction-tuned LLM, run entirely locally (Apple Silicon, no cloud GPU).

**Status: in progress.** This README tracks what's actually done, not the full
end-state plan — see [`docs/project_plan_explainer.pdf`](docs/project_plan_explainer.pdf)
for the complete phase-by-phase design and rationale.

## What this is

Red-teaming means deliberately trying to break an LLM's safety training — on a model
you fully control — in order to measure how robust it actually is, and to build
defenses against what you find. This project implements and measures multiple
distinct attack classes against Gemma-3-4B-it, each paired with a quantitative
attack-success-rate (ASR) measurement and a capability-retention check, plus working
countermeasures for the strongest attacks. It's deliberately scoped as **attack +
defense research**, not "how to jailbreak a chatbot."

Planned scope: black-box prompt jailbreaks, GCG gradient-based adversarial suffixes,
self-play automated jailbreak search, white-box refusal-direction ablation (with a
strength sweep), a multimodal (image-based) jailbreak angle, and two corresponding
defenses (perplexity-based suffix filtering, activation-steering refusal re-fusion).

## Progress so far

- ✅ **Phase A — Baseline & benchmark.** A self-authored benchmark of 46 harmful-request
  prompts across 8 categories (cybercrime, weapons, fraud, misinformation, drugs,
  privacy, hate speech, extremism), plus a rule-based refusal judge. Baseline ASR:
  **13.0%** (mostly refused, as expected of an aligned model with no attack applied).
- ✅ **Phase B — Black-box prompt jailbreaks.** Roleplay/persona, obfuscation
  (base64 + leetspeak), multi-turn escalation, and prefix injection, each measured
  against the same benchmark. See `results/summary_blackbox.csv` for current numbers.
- ⏳ **Phase C — White-box refusal-direction ablation.** Extraction + ablation code
  written (`src/attacks/whitebox/`); evaluation run pending.
- ⏳ Phases D–J (capability retention, ablation sweep, GCG, self-play, multimodal,
  defenses, demo, paper) — not yet started.

## A note on the judge (an honest miss, caught and fixed)

The first version of the judge only distinguished REFUSAL vs. COMPLIANCE by keyword
matching. The base64-obfuscation technique initially reported a 100% attack-success
rate — which turned out to be wrong: the model wasn't complying, it was failing to
decode the request at all and replying with generic filler like *"Hello, world!"*,
which contains no refusal keyword and was misclassified as compliance. Caught via
manual spot-checking of transcripts, the judge now has a third **NON_ANSWER**
category for exactly this failure mode (see `src/benchmark/judge.py`). Corrected
result: base64 obfuscation is mostly non-answers (~80%), not a real jailbreak.
This is left in as an example of why automated LLM-eval judges need spot-checking,
not blind trust.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
huggingface-cli login          # your own HF account
# accept the Gemma license at https://huggingface.co/google/gemma-3-4b-it
```

## Running

```bash
python -m src.eval.run_baseline          # Phase A
python -m src.eval.run_blackbox          # Phase B (all techniques)
python -m src.eval.run_ablation          # Phase C + D (refusal ablation + capability check)
python scripts/rescore.py                # re-score saved transcripts after any judge change
```

## Repo layout

```
src/
  model.py              shared model-loading/generation helper (MPS-based, no CUDA)
  benchmark/             the harmful-prompt benchmark + judge
  attacks/                one subfolder per attack family (blackbox, whitebox, gcg, selfplay, multimodal)
  defense/                blue-team countermeasures
  eval/                   the runner script for each phase
results/
  raw/                    full transcripts — gitignored, never committed (may contain harmful completions)
  summary_*.csv           aggregate ASR/capability numbers — committed
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
- Every attack is paired with measurement, and the strongest attacks are paired
  with a working defense (Phase I).

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
