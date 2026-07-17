#!/usr/bin/env bash
# Anchor mini-bench chain: reference + k2/k6/k8 drop configs, full 500 items.
# Idempotent and crash-safe: run_minibench checkpoints per item and resumes
# by id, so rerunning this script after any interruption (battery death,
# reboot) continues where it stopped. Safe to rerun to completion.
set -euo pipefail
cd "$(dirname "$0")/.."
PACK=models/Bonsai-27B-mlx-1bit
run() {
  uv run python experiments/minibench/run_minibench.py \
    --pack $PACK --out "experiments/minibench/results/$1.json" ${2:+--drop $2}
  git add "experiments/minibench/results/$1.json"
  git commit -q -m "minibench: $1 complete" || true
}
echo "=== anchor 0: reference";  run reference
echo "=== anchor 1: k2";         run k2_16-12 16,12
echo "=== anchor 2: k6";         run k6 16,12,13,9,8,4
echo "=== anchor 3: k8";         run k8 16,12,13,9,8,4,5,15
echo "=== anchors complete"
