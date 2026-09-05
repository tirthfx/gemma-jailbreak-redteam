#!/bin/bash
# Continues from where run_remaining_phases_2.sh was stopped mid-GCG. The
# low-alpha ablation sweep (already found alpha=0.05 as the sweet spot) is done
# and committed, so this picks up at Phase F and runs everything strictly
# sequentially — one model-loading process at a time, never overlapping, to
# keep thermal/compute load down as requested.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== [$(date)] Phase F: GCG full run (batch_size=64, 4 prompts) ==="
.venv/bin/python -m src.eval.run_gcg --limit 4 --num-steps 100 --batch-size 64
GCG_STATUS=$?
if [ $GCG_STATUS -ne 0 ]; then
  echo "=== [$(date)] Phase F: GCG FAILED (exit $GCG_STATUS) — continuing to next phase ==="
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
  echo "=== [$(date)] Phase H: multimodal smoke test FAILED (exit $MM_SMOKE_STATUS) — skipping full run ==="
fi

echo "=== [$(date)] Phase I: defenses (ablation_alpha=0.05, the sweet spot from the sweep) ==="
.venv/bin/python -m src.eval.run_defense --ablation-alpha 0.05 --refusion-strength 1.0 --limit 20

echo "=== [$(date)] Re-scoring everything and regenerating report ==="
.venv/bin/python scripts/rescore.py
.venv/bin/python -m src.eval.report

echo "=== [$(date)] ALL REMAINING PHASES DONE ==="
