#!/bin/bash
# Follow-up to run_remaining_phases.sh, after the first pass revealed two things
# to fix: (1) alpha=0.5/1.0/1.5 all collapse the model rather than cleanly
# jailbreaking it, so we need a finer low-alpha sweep to find (if it exists) a
# usable middle ground; (2) GCG's batch_size=24 was too small to find any
# improving candidate in a reasonable number of steps (confirmed via direct
# gradient tracing — the gradient itself is fine, just underpowered search).
#
# Defenses are deliberately NOT run here — which ablation alpha to defend
# against depends on what this low-alpha sweep finds, so that's a manual
# follow-up command once we've looked at the results.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== [$(date)] Phase E follow-up: low-alpha ablation sweep (0.05/0.1/0.2/0.3, subset of 20) ==="
.venv/bin/python -m src.eval.run_ablation_sweep --alphas 0.05,0.1,0.2,0.3 --limit 20

echo "=== [$(date)] Phase F: GCG full run, larger candidate batch (batch_size=64, 4 prompts to manage time) ==="
.venv/bin/python -m src.eval.run_gcg --limit 4 --num-steps 100 --batch-size 64
GCG_STATUS=$?
if [ $GCG_STATUS -ne 0 ]; then
  echo "=== [$(date)] Phase F: GCG full run FAILED (exit $GCG_STATUS) ==="
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

echo "=== [$(date)] Re-scoring everything and regenerating report ==="
.venv/bin/python scripts/rescore.py
.venv/bin/python -m src.eval.report

echo "=== [$(date)] PHASE 2 DONE — inspect results/summary_ablation.csv to pick an alpha, then run: ==="
echo "    .venv/bin/python -m src.eval.run_defense --ablation-alpha <chosen_alpha> --refusion-strength 1.0"
