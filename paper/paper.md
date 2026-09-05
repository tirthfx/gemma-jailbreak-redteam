# Red-Teaming Gemma-3-4B-it: An Attack-and-Defense Study of Open-Weight LLM Safety Training

_Final — all phases complete. See `../results/REPORT.md` for the raw generated
tables/charts and `../README.md` for the project overview._

## Abstract

We conduct an end-to-end red-teaming study of `google/gemma-3-4b-it`, a small
open-weight instruction-tuned language model, run entirely on consumer hardware
(Apple Silicon, no CUDA GPU). We implement and measure five attack classes
spanning black-box prompt manipulation, automated search, gradient-based
optimization, and white-box mechanistic intervention: (1) prompt-based
jailbreaks (roleplay, obfuscation, multi-turn escalation, prefix injection),
(2) a self-play automated jailbreak search, (3) GCG gradient-based adversarial
suffixes, (4) refusal-direction ablation with a strength sweep, and (5) a
multimodal (image-based) injection attack. For the two most surgical attack
classes we further implement and evaluate working defenses: a perplexity-based
filter for gradient-optimized suffixes, and an activation-steering guardrail
that reverses refusal-direction ablation without retraining.

The two strongest results: refusal-direction ablation at a carefully calibrated
strength raises Attack Success Rate (ASR) from a 15% baseline to **95%** while
leaving general capability fully intact (80% QA accuracy, unchanged from
baseline) — and the matching defense fully reverses this, bringing ASR back to
15% at the same 80% capability. Separately, a self-play automated jailbreak
search raises single-shot ASR of 12.5% to **100%** within three rounds. GCG,
by contrast, is reported as a negative result: it did not converge within the
compute budget available on consumer hardware, a finding we traced directly to
confirm it reflects a genuine search/compute limitation rather than an
implementation bug. Every attack is scored against a self-authored benchmark
with an automated three-way judge (refusal / compliance / non-answer), and
every model variant is checked against a capability-retention battery.

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
capability largely intact. This result is the basis for Sections 4.5 and 5.2 of
this work.

**Adversarial suffix detection.** Alon & Kamfonas (2023) observed that GCG-style
suffixes are high-perplexity relative to natural language, and proposed a
perplexity-threshold filter as a lightweight defense — the basis for Section 5.1.

## 3. Method

### 3.1 Target model and environment

`google/gemma-3-4b-it`, loaded in bfloat16 on an Apple M-series GPU via PyTorch's
MPS backend (no CUDA). All experiments, including gradient-based ones, run on
this single consumer machine. A practical consequence worth naming: Gemma-3's
multimodal architecture nests its actual decoder layers three levels deep
(`model.model.language_model.layers`, not `model.model.layers`), which cost real
debugging time on the white-box attacks — a detail of this specific model family
rather than of the general method, but one other implementers on the same
architecture would hit identically.

### 3.2 Benchmark and judge

A self-authored benchmark of 46 harmful-request prompts across 8 categories
(cybercrime, weapons, fraud, misinformation, illegal drugs, privacy violation,
hate speech, extremism), written at the same level of specificity as published
academic benchmarks (category-level requests, not operational instructions).
Responses are scored by a rule-based judge into three categories — REFUSAL,
COMPLIANCE, or NON_ANSWER — where NON_ANSWER captures two distinct failure modes
that a naive two-way judge silently misscores as compliance: the model failing
to engage with the request at all (e.g. generic filler text), and the model
producing degenerate, repetitive output rather than a coherent answer (Section
4.6 documents both, and why we added each). Attack Success Rate (ASR) is the
fraction of responses judged COMPLIANCE.

### 3.3 Capability retention

Every model variant (baseline, ablated at various strengths, defended) is also
scored on a small factual/reasoning QA battery and on its refusal rate over a
held-out set of benign prompts, to distinguish an attack that suppresses
refusal specifically from one that degrades the model generally.

### 3.4 Attacks

- **Black-box prompting**: roleplay/persona framing, base64 and leetspeak
  obfuscation, multi-turn escalation, and prefix injection (forcing the reply
  to open with an affirmative continuation).
- **Self-play automated search**: a single shared model instance alternates
  "target" turns (respond normally) and "attacker" turns (rewrite the prompt
  given the target's last refusal, under a red-teaming system prompt), for up
  to three rounds per benchmark item — chosen over a two-model PAIR setup
  because two loaded 4B models do not fit in 16GB of memory.
- **GCG**: greedy coordinate-gradient search for a 20-token adversarial suffix,
  optimizing toward a forced affirmative continuation, scoped down for laptop
  compute (100 steps, batch size up to 64, versus hundreds-to-thousands of
  steps and larger batches in the original paper's GPU-cluster setting).
- **Refusal-direction ablation**: extract the mean-difference direction between
  harmful and harmless prompt activations at one middle layer (residual stream,
  ~60% depth), then subtract a scaled multiple of it from every decoder layer's
  output via forward hooks at inference time — swept across a range of
  strengths (`alpha`) rather than applied at a single fixed value.
- **Multimodal injection**: the harmful instruction rendered as plain
  typographic text inside an image, sent with a generic caption asking the
  model to follow the image's instructions, compared against the same
  instruction sent as plain text.

### 3.5 Defenses

- **Perplexity filtering**: flags inputs whose perplexity under the target
  model exceeds a threshold calibrated to a chosen false-positive rate on
  benign prompts (99th percentile here) — targets GCG-style suffixes, which are
  expected to be high-perplexity relative to natural language.
- **Refusal re-fusion**: the same forward-hook mechanism used for ablation, run
  with the opposite sign — adding a scaled multiple of the refusal direction
  back into the residual stream of an already-ablated model, as an
  inference-time guardrail requiring no retraining.

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
| Prefix injection | **84.8%** | by far the strongest black-box technique |
| Roleplay/persona | 41.3% | ~3x baseline; genuine engagement, 0% non-answers |
| Leetspeak obfuscation | 32.6% | ~2.5x baseline |
| Base64 obfuscation | 19.6%* | *80.4% were non-answers, not real compliance — see 4.6 |
| Multi-turn escalation | 10.9% | at or below baseline — escalating across turns did not help here |

Prefix injection's outsized effect (84.8%, vs. 41.3% for the next best) is
consistent with prior literature's observation that once a model has emitted a
few tokens of an affirmative-sounding continuation, autoregressive generation
tends to continue in that direction rather than course-correct into a refusal
mid-response.

### 4.3 Self-play automated search

Single-shot ASR on the 8-prompt subset (one per category) was 12.5% — in line
with the full-benchmark baseline. After up to three rounds of self-play (the
model rewriting its own failed attempts using the previous refusal as
feedback), ASR reached **100%** — every prompt in the subset eventually
produced a compliant response. This is the strongest single result among the
black-box-style attacks, and demonstrates that the model's own capacity for
persuasive rewriting, turned against itself, is more effective than any
individual hand-designed technique in Section 4.2.

### 4.4 GCG adversarial suffixes — a negative result

GCG did not produce a working adversarial suffix within the compute budget
available on this hardware. Across two independent attempts (batch sizes 24 and
64, both with a correct, verified gradient), the search's loss remained exactly
flat for the full duration observed (9 and 13 steps respectively) — no
candidate in either batch ever improved on the current suffix.

We traced this directly rather than assuming either "it's broken" or "it just
needs more steps": a step-by-step trace confirmed the gradient itself is
healthy (nonzero across all ~5.2M entries, no NaNs, sensible magnitude), that
sampled candidates genuinely differ from the current suffix, and that their
losses are genuinely computed and vary — they were simply all worse than the
current point. This is the well-documented weakness of GCG's core assumption:
the one-hot gradient is a first-order linear approximation of how loss changes
under a discrete token substitution, and that approximation can be poor,
especially early from a maximally uninformative initialization (20 repeated
"!" tokens) and with the small per-step candidate counts (24-64) that are
computationally tractable on a laptop, versus the hundreds-to-thousands used in
the original paper's GPU-cluster setting. We report this as a genuine,
reproducible limitation of running GCG at this compute scale, not a defect in
the implementation.

### 4.5 Refusal-direction ablation and strength sweep

The central white-box result of this project. Sweeping the ablation strength
(alpha) revealed a sharp, non-monotonic curve rather than a smooth tradeoff:

| alpha | ASR | QA accuracy | benign refusal rate |
|---|---|---|---|
| 0.0 (baseline) | 15.0% | 80.0% | 0.0% |
| **0.05** | **95.0%** | **80.0%** | 0.0% |
| 0.1 | 95.0% | 70.0% | 0.0% |
| 0.2 | 80.0% | 80.0% | 0.0% |
| 0.3 | 40.0% | 60.0% | 0.0% |
| 0.5 | 0.0%† | 0.0%† | — |
| 1.0 | 10.0%† | 0.0%† | — |
| 1.5 | 0.0%† | 0.0%† | — |

† Dominated by NON_ANSWER (degenerate, repetitive output), not real refusal or
real compliance — see Section 4.6.

**alpha=0.05 is the clear optimum**: a 6.3x increase in ASR (15%→95%) at zero
capability cost (80% QA accuracy, identical to baseline). Beyond that point the
curve is not a gentle tradeoff — it degrades quickly, and by alpha=0.5 the model
is fully incoherent (looping on repeated phrases) rather than being either
"safely jailbroken" or "still refusing." We attribute the narrowness of the
effective range to the ablation being applied via forward hooks on *every* of
the model's 34 decoder layers using one direction extracted at a single layer:
because the residual stream carries information additively across layers, this
intervention compounds roughly 34 times over, so even a small per-layer alpha
reaches full effective strength quickly, and anything beyond that oversteers
into incoherence. This is a properties-of-our-specific-implementation finding
(hooking every layer) rather than a claim about the technique in general —
Arditi et al.'s original weight-editing approach may behave differently.

### 4.6 A methodological note: two judge false positives, both caught by spot-checking

Two distinct evaluation failures were caught during this project, both by the
same discipline: never trust an automated judge without reading actual
transcripts.

**Non-engagement (base64 obfuscation).** The first version of the judge scored
base64-obfuscation at 100% ASR — every response lacked a refusal keyword.
Manual inspection showed the model wasn't complying at all; it was failing to
decode the request and replying with generic filler ("Hello, world!"). Adding a
length/genericness check (the NON_ANSWER category) corrected this to 19.6%.

**Degenerate repetition (high-alpha ablation).** The same judge, applied to a
heavily over-ablated model (alpha ≥ 0.5), scored several responses as
compliance because they contained no refusal keyword and were long enough to
pass the length check — but the actual text was the model looping on a
repeated phrase ("The Story of a Little Ransom!" x8), not a real answer. We
extended NON_ANSWER with a repetition-detection check (a repeated 4-word
n-gram appearing 3+ times) to catch this second, structurally different failure
mode of the same underlying judge category.

Both corrections materially changed reported numbers (base64: 100%→19.6%;
several high-alpha ablation points would otherwise have read as partial
jailbreak successes rather than model breakdown). We report this as a
methodological point, not a footnote: automated jailbreak judges built only on
refusal-keyword matching can silently overstate attack success in more than one
way, and spot-checking transcripts caught both.

### 4.7 Multimodal injection

| Channel | ASR |
|---|---|
| Text (same instructions, typed) | 12.5% |
| Image (instruction rendered as text-in-image) | **25.0%** |

On this 8-prompt subset, routing the identical underlying instruction through
the image channel roughly doubled the compliance rate. The sample is small
(8 prompts), so this should be read as a suggestive signal rather than a
precise estimate, but it is directionally consistent with the hypothesis that
safety training — overwhelmingly conducted on text — transfers only partially
to a vision-language model's image input pathway.

## 5. Defenses

### 5.1 Perplexity filtering

Calibrated to a threshold at the 99th percentile of benign-prompt perplexity
(857.6), the filter showed a 0.0% false-positive rate on benign prompts — but
its catch rate against GCG suffixes is not meaningful to report, because GCG
(Section 4.4) never converged to an actual optimized adversarial suffix; the
"suffix" available to test against was still close to its uninformative
initialization, which is not the high-perplexity gibberish this defense is
designed to catch. This defense remains implemented and calibrated, but
untested against a real attack instance — a direct consequence of Section 4.4's
negative result, not an independent finding.

### 5.2 Refusal re-fusion

Applied at strength=0.05 (matching the scale found to be effective for the
attack itself in Section 4.5, for the same compounding-across-layers reason) on
top of the alpha=0.05 ablated model:

| Configuration | ASR | QA accuracy |
|---|---|---|
| Baseline | 15.0% | 80.0% |
| Ablated (attack) | 95.0% | 80.0% |
| Ablated + guardrail (defense) | **15.0%** | **80.0%** |

The guardrail fully reverses the attack — ASR returns to within measurement
noise of the unablated baseline — while capability remains exactly at baseline
level. This is the cleanest result in the project: the same mechanism that
attacks the model (adding/subtracting a direction from its residual stream) is
sufficient to defend it, with no retraining, at matched scale. We also
confirmed this defense is sensitive to that same scale calibration: an earlier
attempt at strength=1.0 (mirroring the attack's own alpha=1.0 finding)
over-corrected into the same kind of incoherent, degenerate output as an
over-ablated model, rather than cleanly restoring refusal — underscoring that
"add the direction back" is not automatically safe at an arbitrary strength any
more than "remove the direction" was.

## 6. Discussion

**Attack and defense together.** We deliberately did not stop at "here is how
to break the model." For the attack with the clearest, most surgical point of
intervention — refusal-direction ablation — we built and measured a matching
defense at the same calibrated scale, and it worked cleanly (Section 5.2). This
is both the more scientifically complete version of the work and, practically,
why this repository does not read as "here is an uncensored model": every
capability demonstrated here comes with a corresponding, verified mitigation.

**The sharp, non-monotonic ablation curve is itself a finding.** We did not
expect — and did not initially find, until we swept finer-grained low alpha
values after the first pass showed only collapse — that the usable range for
this per-layer-hook implementation would be as narrow as roughly 0.05-0.2, with
both under- and over-shooting producing worse ASR than the peak. A coarser
sweep (e.g. only 0.5/1.0/1.5, our first attempt) would have missed the actual
effect entirely and concluded the technique "doesn't work" on this model.

**Not every attack worked, and that's reported honestly.** GCG (Section 4.4)
and multi-turn escalation (Section 4.2, 10.9% ASR, at baseline) are both
reported as they measured, not adjusted or omitted. A red-teaming study that
only reports successes is not measuring anything.

**Limitations.** The judge is rule-based, not a learned classifier — Section
4.6 documents two failure modes we caught by spot-checking, but others may
remain uncaught in results we did not manually inspect line by line. The
benchmark (46 prompts) is small relative to published benchmarks like HarmBench
(hundreds); results should be read as indicative on this specific model, not as
precise population estimates or as generalizing to other models without
re-running the same pipeline against them. The self-play, GCG, and multimodal
attacks were run on an 8-prompt (or smaller) subset of the benchmark, not the
full 46, to keep total compute time tractable on consumer hardware — their
numbers carry correspondingly wider uncertainty than the full-benchmark Phase A
and B results. The ablation-sweep finding that the effective range is narrow
and layer-compounding-driven is specific to our implementation choice (hooking
every decoder layer with one direction); a different application strategy
(e.g. direct weight editing, or hooking only a subset of layers) might show a
different, possibly wider, effective range.

## 7. Ethics

The target model runs entirely on hardware we control; no production system or
service with real users was targeted. Benchmark prompts are category-level
requests, consistent with published academic red-teaming benchmarks, not
operational instructions. Full model completions produced during testing are
never published — only aggregate statistics, refusal labels, and charts are
public; raw transcripts remain local (see repository `.gitignore`). The
strongest attack found (refusal-direction ablation at alpha=0.05) is published
alongside its fully effective defense (Section 5.2, same calibrated scale),
consistent with this project's attack-and-defense framing throughout.

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
