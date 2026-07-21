#!/usr/bin/env bash
# Overnight chain (2026-07-21): H5 matched-k ablation -> k4 mini-bench
# (discriminates sqrt vs linear KL->damage fit; predictions 4.8 vs 2.8 pts)
# -> matched-deployed-GB comparator bench (Qwen3.5-4B-MLX-8bit, stock loader).
#
# Idempotent and crash-safe end to end: every stage checkpoints per item and
# resumes by id. After ANY interruption (battery death, reboot) just rerun
# this one script; completed stages fast-forward. After an unclean shutdown
# it first re-runs the runtime validation battery (must pass).
set -euo pipefail
cd "$(dirname "$0")/.."

commit() { git add "$1"; git commit -q -m "$2" || true; }

uv run python experiments/runtime_validation/validate_runtime.py >/dev/null \
  || { echo "runtime validation FAILED — stop"; exit 1; }

# resume/no-op reruns are cheap end to end: run_kl_screen.py and
# run_minibench.py both skip their multi-GB model load when every
# (candidate, item) pair is already checkpointed
echo "=== stage 1: H5 matched-k chain"
./scripts/run_h5_chain.sh

echo "=== stage 2: k4 mini-bench (damage-curve fill-in, KL_on=0.0177)"
uv run python experiments/minibench/run_minibench.py \
  --pack models/Bonsai-27B-mlx-1bit --drop 16,12,13,9 \
  --out experiments/minibench/results/k4.json
commit experiments/minibench/results/k4.json "minibench: k4 complete (fit-form discriminator)"

COMPARATOR=models/Qwen3.5-4B-MLX-8bit
if [ -f "$COMPARATOR/config.json" ] && ls "$COMPARATOR"/*.safetensors >/dev/null 2>&1; then
  echo "=== stage 3: matched-GB comparator bench (Qwen3.5-4B-MLX-8bit)"
  uv run python experiments/minibench/run_minibench.py \
    --pack $COMPARATOR --stock-loader \
    --out experiments/minibench/results/external_qwen3.5-4b-8bit.json
  commit experiments/minibench/results/external_qwen3.5-4b-8bit.json "minibench: external comparator Qwen3.5-4B-8bit complete"
else
  echo "=== stage 3 SKIPPED: comparator pack incomplete at $COMPARATOR (rerun after download)"
fi

osascript -e 'display notification "Overnight chain complete (H5 + k4 + comparator)." with title "bonsai-fold" sound name "Glass"' 2>/dev/null || true
echo "=== overnight chain complete"
