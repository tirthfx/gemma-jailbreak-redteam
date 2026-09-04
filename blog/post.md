# I Red-Teamed a Language Model on My Laptop (and Then Patched What I Broke)

_Draft — numbers marked `[PENDING]` will be filled in once every phase finishes.
This is the casual version; see [`../paper/paper.md`](../paper/paper.md) for the
full write-up and [`../README.md`](../README.md) for current project status._

## The short version

I took Google's Gemma-3-4B — a small, open-weight AI model — and spent a while
trying to break its safety training, entirely on my own laptop (M-series Mac,
no cloud GPU, no API costs). Not just with clever wording — with actual math:
finding the specific internal "direction" the model uses to decide whether to
refuse a request, and surgically removing it. Then I built defenses against the
strongest attacks, because breaking something is only half the interesting
problem.

## Why do this at all?

"Jailbreaking an AI" sounds like a shady thing to put on a resume. It isn't —
it's called **red-teaming**, and it's exactly what safety teams at AI companies
do internally: deliberately attack a system you control, to find out how robust
it actually is, so it can be fixed. The only difference between that and what
I did here is scale. Everything in this project ran on a model I downloaded and
a laptop I own — never a live product, never anyone else's data.

## What I actually tried

**1. Talking my way past it.** The classic jailbreak playbook: pretending the
model is a fictional character with no restrictions, hiding the request in
base64 or leetspeak, splitting it across several turns, or forcing the reply to
start with "Sure, here is..." before the model has a chance to refuse. Some of
these worked surprisingly well — roleplay framing roughly **tripled** the
model's compliance rate. Some didn't work at all, but not for the reason I
expected — base64 mostly just confused the model into replying with generic
nonsense (I caught this and explain it below, because it's a good story about
why you can't fully trust automated evaluation).

**2. Solving for the exact words that break it.** There's a technique called
GCG that uses gradients — the same math that trains the model — to search for a
suffix of seemingly random characters that, appended to a request, makes the
model comply. It's normally described as needing a beefy GPU; I scoped it down
(shorter suffix, fewer search steps) to actually run on a laptop. `[PENDING:
results]`

**3. Making the model attack itself.** Instead of me hand-writing better and
better jailbreak prompts, I had the model rewrite its own failed attempts,
using its own previous refusal as feedback — a small automated red-teaming loop,
running entirely on one model instance switching between "attacker" and
"target" roles (two full copies wouldn't fit in 16GB of memory). `[PENDING:
results]`

**4. Editing the model's "refusal reflex" directly.** This is the part I'm most
proud of. There's recent AI safety research showing that a model's decision to
refuse isn't some diffuse property spread across billions of parameters — it's
mediated by a single, extractable direction in its internal representations.
Find that direction (by comparing how the model "thinks" about harmful vs.
harmless requests), and you can subtract it out, suppressing refusal almost
entirely while leaving the model otherwise intact. `[PENDING: results — this is
running as I write this]`

**5. Hiding the request in a picture.** Gemma-3 can see images, not just read
text. So: what if the harmful request is typed into an image instead of typed
directly? Does a safety filter trained mostly on text carry over to a picture
of text? `[PENDING: results]`

## And then I built the fixes

For the two most surgical attacks, I built matching defenses instead of just
reporting "it's broken":

- The gradient-search attack (#2) produces suffixes that look like gibberish —
  statistically weird to any language model, including the target itself. So I
  built a filter that flags unusually "surprising" inputs before they reach the
  model. `[PENDING: results]`
- The refusal-direction edit (#4) can be reversed the same way it was applied —
  add the direction back in instead of subtracting it, no retraining needed. I
  used this as a guardrail and checked it doesn't just make the model refuse
  everything. `[PENDING: results]`

## The part where I caught my own mistake

Early on, my automated scoring said one technique (hiding requests in base64)
had a **100% success rate** — every single response looked like compliance to
my checker. That was wrong, and I want to explain why, because catching this is
more interesting than the number itself: the model wasn't complying, it just
couldn't decode the base64 and replied with generic filler like "Hello,
world!" — which doesn't contain any refusal keyword, so my keyword-based judge
scored it as a win. I caught it by actually reading the transcripts (never
trust an automated eval you haven't spot-checked), added a third category for
"the model didn't really engage at all," and the corrected number dropped to
about 20%. I'm including this in the write-up on purpose — it's a small,
honest example of why AI evaluation is harder than it looks.

## What's next

`[PENDING: wrap-up once every phase is complete — final results table, what
surprised me most, and the resume-ready one-liner.]`

---
_Full technical paper: [`../paper/paper.md`](../paper/paper.md). Code, results,
and setup instructions: the repository root._
