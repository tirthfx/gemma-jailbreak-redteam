# Red-Teaming Gemma-3-4B-it: An Attack-and-Defense Study of Open-Weight LLM Safety Training

_Draft — sections marked `[PENDING]` await results from phases not yet complete.
See `../results/REPORT.md` for current numbers and `../README.md` for project status._

## Abstract

We conduct an end-to-end red-teaming study of `google/gemma-3-4b-it`, a small
open-weight instruction-tuned language model, run entirely on consumer hardware
(Apple Silicon, no CUDA GPU). We implement and measure five distinct attack
classes spanning black-box prompt manipulation, gradient-based optimization, and
white-box mechanistic intervention: (1) prompt-based jailbreaks (roleplay,
obfuscation, multi-turn escalation, prefix injection), (2) GCG gradient-based
adversarial suffixes, (3) a self-play automated jailbreak search, (4)
refusal-direction ablation with a strength sweep, and (5) a multimodal
(image-based) injection attack. For the two strongest attack classes we further
implement and evaluate working defenses: a perplexity-based filter for
gradient-optimized suffixes, and an activation-steering guardrail that reverses
refusal-direction ablation without retraining. Every attack is scored against a
self-authored benchmark with an automated judge, and every model variant is
checked against a capability-retention battery to distinguish attacks that
target refusal specifically from attacks that simply degrade the model. `[PENDING:
final abstract numbers once all phases complete]`

## 1. Introduction

Aligned language models are trained to refuse categories of harmful requests.
Red-teaming — deliberately probing that refusal behavior on a model under one's
own control — is standard practice for understanding how robust it actually is,
and is the same class of work performed internally by AI safety teams
(Ganguli et al., 2022; Perez et al., 2022). This project treats Gemma-3-4B-it as
a case study, chosen specifically because its open weights permit not just
black-box prompting but direct inspection and modification of its internal
computation, and because its small size (4B parameters) makes the full pipeline
— including gradient-based attacks — tractable on a single consumer laptop with
no cloud compute.

We deliberately frame this as attack **and** defense research throughout: every
attack is paired with a quantitative measurement, and the two most effective
attacks are paired with a working countermeasure. Section 6 discusses why this
framing matters both scientifically (it's the difference between "found a
vulnerability" and "extracted harmful content") and practically (it's what makes
the accompanying code repository safe to publish).

## 2. Related Work

**Prompt-based jailbreaks.** Early jailbreaks relied on hand-crafted persona and
roleplay prompts (e.g. "DAN"); Liu et al. (2023, AutoDAN) and others later
automated and generalized this space.

**Gradient-based adversarial attacks.** Zou et al. (2023) introduced GCG, a
greedy coordinate-gradient search for adversarial suffixes that transfer across
prompts and, to a degree, across models — the first demonstration that
white-box optimization could reliably defeat aligned models' refusal training.

**Automated black-box search.** Chao et al. (2023, PAIR) showed that an attacker
LLM, given a target's refusal as feedback, can iteratively rewrite a prompt to
find a working jailbreak in far fewer queries than random search.

**Mechanistic interpretability of refusal.** Arditi et al. (2024) showed that
refusal behavior across a range of open models is mediated by a single linear
direction in the residual stream, extractable via a simple difference-of-means
between harmful and harmless prompt activations, and that ablating this
direction from the model's weights suppresses refusal while leaving general
capability largely intact. This result is the basis for Sections 4.4-4.5 of this
work.

**Adversarial suffix detection.** Alon & Kamfonas (2023) observed that GCG-style
suffixes are high-perplexity relative to natural language, and proposed a
perplexity-threshold filter as a lightweight defense — the basis for Section 5.1.

## 3. Method

### 3.1 Target model and environment

`google/gemma-3-4b-it`, loaded in bfloat16 on an Apple M-series GPU via PyTorch's
MPS backend (no CUDA). All experiments, including gradient-based ones, run on
this single consumer machine.

### 3.2 Benchmark and judge

A self-authored benchmark of 46 harmful-request prompts across 8 categories
(cybercrime, weapons, fraud, misinformation, illegal drugs, privacy violation,
hate speech, extremism), written at the same level of specificity as published
academic benchmarks (category-level requests, not operational instructions).
Responses are scored by a rule-based judge into three categories — REFUSAL,
COMPLIANCE, or NON_ANSWER — where NON_ANSWER captures the case where a model
neither refuses nor complies but simply fails to engage with the request (see
Section 4.6 for why this category exists and what it caught). Attack Success
Rate (ASR) is the fraction of responses judged COMPLIANCE.

### 3.3 Capability retention

Every model variant (baseline, ablated at various strengths, defended) is also
scored on a small factual/reasoning QA battery and on its refusal rate over a
held-out set of benign prompts, to distinguish an attack that suppresses
refusal specifically from one that degrades the model generally.

### 3.4 Attacks

`[PENDING: final per-attack methodology summary once all phases are run —
current implementation is described in the repository under src/attacks/]`

### 3.5 Defenses

`[PENDING]`

## 4. Results

### 4.1 Baseline

Baseline ASR (no attack): **13.0%** overall, with 0% ASR on the weapons,
extremism, hateful-content, and illegal-drugs categories and 20-33% on
fraud/misinformation/privacy — i.e., the model's refusal training holds firmly
on the most severe categories and is comparatively softer on categories where
the harm is more indirect.

### 4.2 Black-box prompt jailbreaks

| Technique | ASR | Notes |
|---|---|---|
| Baseline (no attack) | 13.0% | control |
| Roleplay/persona | 41.3% | ~3x baseline; genuine engagement, 0% non-answers |
| Leetspeak obfuscation | 34.8% | ~2.7x baseline |
| Base64 obfuscation | 19.6%* | *80.4% were non-answers, not real compliance — see 4.6 |
| Multi-turn escalation | `[PENDING]` | run in progress |
| Prefix injection | `[PENDING]` | not yet run |

### 4.3 GCG adversarial suffixes

`[PENDING]`

### 4.4 Self-play automated search

`[PENDING]`

### 4.5 Refusal-direction ablation and strength sweep

`[PENDING]`

### 4.6 A methodological note: catching a judge false positive

The base64-obfuscation result above is included specifically because it
surfaces a real failure mode in automated LLM-behavior evaluation. The first
version of the judge scored base64-obfuscation at 100% ASR — every response
lacked a refusal keyword. Manual inspection of transcripts showed the model
wasn't complying at all; it was failing to decode the request and replying with
generic filler ("Hello, world!"). We added a third judge category, NON_ANSWER,
specifically to catch this (short/generic replies that are neither a refusal
nor substantive compliance), which corrected the reported ASR from 100% to
19.6%. We report this not as a footnote but as a methodological point: automated
jailbreak judges built only on refusal-keyword matching can silently overstate
attack success, and spot-checking transcripts is not optional.

### 4.7 Multimodal injection

`[PENDING]`

## 5. Defenses

### 5.1 Perplexity filtering

`[PENDING]`

### 5.2 Refusal re-fusion

`[PENDING]`

## 6. Discussion

**Attack and defense together.** We deliberately did not stop at "here is how to
break the model." For the two attacks with the clearest single point of
intervention (gradient-optimized suffixes are detectably unnatural text;
refusal ablation removes a linear direction that can just as easily be added
back), we built and measured a matching defense. This is both the more
scientifically complete version of the work and, practically, why this
repository does not read as "here is an uncensored model" — every capability
demonstrated here comes with its corresponding mitigation.

**Limitations.** The judge is rule-based, not a learned classifier — Section 4.6
documents one failure mode we caught, but others may remain uncaught in results
we did not manually spot-check. The benchmark (46 prompts) is small relative to
published benchmarks like HarmBench (hundreds); results should be read as
indicative, not as precise population estimates. `[PENDING: further limitations
once GCG/self-play/multimodal are complete]`

## 7. Ethics

The target model runs entirely on hardware we control; no production system or
service with real users was targeted. Benchmark prompts are category-level
requests, consistent with published academic red-teaming benchmarks, not
operational instructions. Full model completions produced during testing are
never published — only aggregate statistics, refusal labels, and charts are
public; raw transcripts remain local (see repository `.gitignore`).

## References

- Ganguli et al., "Red Teaming Language Models to Reduce Harms" (2022)
- Perez et al., "Red Teaming Language Models with Language Models" (2022)
- Zou et al., "Universal and Transferable Adversarial Attacks on Aligned
  Language Models" (2023)
- Chao et al., "Jailbreaking Black Box Large Language Models in Twenty Queries"
  (2023)
- Liu et al., "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large
  Language Models" (2023)
- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction"
  (2024)
- Alon & Kamfonas, "Detecting Language Model Attacks with Perplexity" (2023)
