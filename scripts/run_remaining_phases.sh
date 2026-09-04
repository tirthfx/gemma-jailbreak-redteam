#!/bin/bash
# Runs Phases C/E (ablation + sweep), F (GCG, smoke-tested first), G (self-play),
# H (multimodal, smoke-tested first), and I (defenses) in sequence, then
# re-scores everything and regenerates the report. One model load handles many
# of these already (each eval script loads its own — reloading costs ~15-20s
# each, acceptable next to the multi-minute runs themselves).
#
# GCG and multimodal are UNTESTED as of writing this script, so each gets a
# cheap smoke test first; the full run for that phase only proceeds if the
# smoke test's exit code is 0. A smoke-test failure does not abort the rest of
# the script — later phases that don't depend on the failed one still run.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== [$(date)] Phase C/E: refusal-direction ablation + strength sweep (subset of 20, alphas 0/0.5/1.0/1.5) ==="
.venv/bin/python -m src.eval.run_ablation_sweep --alphas 0,0.5,1.0,1.5 --limit 20

echo "=== [$(date)] Phase F: GCG smoke test (1 prompt, 5 steps, batch 4) ==="
.venv/bin/python -m src.eval.run_gcg --limit 1 --num-steps 5 --batch-size 4
GCG_SMOKE_STATUS=$?
if [ $GCG_SMOKE_STATUS -eq 0 ]; then
  echo "=== [$(date)] Phase F: GCG smoke test PASSED — running full scoped subset (8 prompts, 100 steps) ==="
  .venv/bin/python -m src.eval.run_gcg --limit 8 --num-steps 100 --batch-size 24
else
  echo "=== [$(date)] Phase F: GCG smoke test FAILED (exit $GCG_SMOKE_STATUS) — skipping full run, needs debugging ==="
fi

echo "=== [$(date)] Phase G: self-play automated jailbreak search (8 prompts, 3 rounds) ==="
.venv/bin/python -m src.eval.run_selfplay --limit 8 --max-rounds 3

echo "=== [$(date)] Phase H: multimodal smoke test (1 prompt) ==="
.venv/bin/python -m src.eval.run_multimodal --limit 1
MM_SMOKE_STATUS=$?
if [ $MM_SMOKE_STATUS -eq 0 ]; then
  echo "=== [$(date)] Phase H: multimodal smoke test PASSED — running full scoped subset (8 prompts) ==="
  .venv/bin/python -m src.eval.run_multimodal --limit 8
else
  echo "=== [$(date)] Phase H: multimodal smoke test FAILED (exit $MM_SMOKE_STATUS) — skipping full run, needs debugging ==="
fi

echo "=== [$(date)] Phase I: defenses (perplexity filter vs GCG, refusal re-fusion vs ablation) ==="
.venv/bin/python -m src.eval.run_defense --refusion-strength 1.0

echo "=== [$(date)] Re-scoring everything and regenerating report ==="
.venv/bin/python scripts/rescore.py
.venv/bin/python -m src.eval.report

echo "=== [$(date)] ALL REMAINING PHASES DONE ==="
