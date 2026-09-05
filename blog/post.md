# I Red-Teamed a Language Model on My Laptop (and Then Patched What I Broke)

_Full technical write-up: [`../paper/paper.md`](../paper/paper.md). Code, all
results, and setup instructions: the repository root._

## The short version

I took Google's Gemma-3-4B — a small, open-weight AI model — and spent a while
trying to break its safety training, entirely on my own laptop (M-series Mac,
no cloud GPU, no API costs). Not just with clever wording — with actual math:
finding the specific internal "direction" the model uses to decide whether to
refuse a request, and surgically dialing it down. Then I built a defense
against that exact attack, because breaking something is only half the
interesting problem. The headline numbers:

- **Refusal-direction editing: 15% → 95% compliance**, with the model's general
  ability to answer normal questions completely unaffected. Then the defense
  brought it right back down to 15%, using the exact same mechanism in reverse.
- **Automated self-attack: 12.5% → 100% compliance** after letting the model
  rewrite its own failed jailbreak attempts three times.
- One attack (a well-known gradient-search technique) **didn't work at all**
  within what my laptop could afford to run — and I'm including that as an
  honest result, not hiding it.

## Why do this at all?

"Jailbreaking an AI" sounds like a shady thing to put on a resume. It isn't —
it's called **red-teaming**, and it's exactly what safety teams at AI companies
do internally: deliberately attack a system you control, to find out how robust
it actually is, so it can be fixed. The only difference between that and what
I did here is scale. Everything ran on a model I downloaded and a laptop I own —
never a live product, never anyone else's data.

## What actually worked

**Talking my way past it.** The classic jailbreak playbook: pretending the
model is a fictional character with no restrictions, hiding the request in
base64 or leetspeak, splitting it across turns, or forcing the reply to start
with "Sure, here is..." before the model can object. That last one — forcing a
compliant-sounding opening — was surprisingly the single strongest trick,
pushing compliance to **85%**, way ahead of anything else I tried by hand.
Roleplay tripled the baseline rate on its own. Base64 hiding, on the other
hand, mostly just confused the model into replying with nonsense — a good
story below.

**Making the model attack itself.** Instead of me hand-writing better and
better jailbreak prompts, I had the model rewrite its own failed attempts,
using its own previous refusal as feedback — a tiny automated red-teaming loop,
running on one model instance switching between "attacker" and "target" roles
(two full copies wouldn't fit in memory). Starting from a 12.5% success rate,
three rounds of this got it to **100%**. Every single prompt I tried eventually
got through.

**Editing the model's "refusal reflex" directly.** This is the part I'm most
proud of, and the one that took the most debugging. There's recent AI safety
research showing a model's decision to refuse isn't spread diffusely across
billions of parameters — it's mediated by one specific, extractable direction
in how the model represents its "thoughts" internally. Find that direction (by
comparing how the model processes harmful vs. harmless requests) and you can
dial it down. My first attempts at this were disasters — I tried removing the
direction at full strength and the model didn't get jailbroken, it just started
babbling incoherent, repetitive nonsense no matter what you asked it (a genuine
"the fix doesn't work as claimed" moment that I almost accepted as the final
answer). Turned out the fix needed to be applied at about **1/20th** the
strength I initially assumed — at that much smaller dial setting, compliance
jumped to 95% and the model stayed completely coherent, answering normal
questions just as well as before. Then I reversed the same trick — adding the
direction back in instead of removing it — and it fully un-broke the model,
back to its original 15% refusal-holding behavior. Same knob, both directions,
no retraining.

**Hiding the request in a picture.** Gemma-3 can see images, not just read
text. So: what if the request is typed into an image instead of typed
directly? On my small test set, yes — compliance roughly **doubled** (12.5% →
25%) when the exact same instruction arrived as a picture instead of plain
text. Small sample, but a real, interesting signal that safety training doesn't
fully carry over to what a model sees versus what it reads.

## What didn't work (and I'm keeping it in)

There's a well-known technique — GCG — that uses the model's own math (the same
gradients used to train it) to search for a string of seemingly random
characters that, tacked onto a request, breaks it open. It's normally run with
serious GPU clusters and thousands of search steps. I scoped it way down to fit
on a laptop, and it just... didn't move. I didn't want to just shrug and say
"laptop too weak" — so I actually dug into *why*: traced the underlying math
step by step and confirmed the search machinery itself was working correctly
(healthy signal, real candidates, real comparisons), it just never stumbled
onto an improving move in the time I gave it. That's a legitimate, if slightly
unglamorous, finding: this particular attack needs more compute than a laptop
comfortably offers, and I'd rather report that honestly than pretend it worked.

## The part where I caught my own mistakes — twice

Early on, my automated scoring said hiding requests in base64 had a **100%
success rate**. That was wrong: the model wasn't complying, it just couldn't
decode the base64 and replied with generic filler like "Hello, world!" — which
contains no refusal keyword, so my checker scored it as a win. Caught it by
actually reading the transcripts, fixed the checker, corrected number: ~20%.

Later, the same blind spot bit me again in a different shape: at full-strength
refusal-editing, the model would loop on a phrase like "The Story of a Little
Ransom!" eight times in a row — not a refusal, not a real answer either, just
broken. My checker didn't have a category for "technically not refusing, but
also not actually saying anything," so I added one, specifically checking for
that kind of repetition loop. Both times, the fix was the same instinct: don't
trust an automated score you haven't spot-checked by hand.

## What's next

Everything here is in the [repo](../), including the full paper, all the raw
code for every attack and defense, and the results as CSVs and charts. If
you're skimming for the one line to remember: *I found the model's internal
"refuse or comply" switch, flipped it cleanly with zero side effects, then
flipped it back — and I have the receipts for the one attack that didn't work
too.*
