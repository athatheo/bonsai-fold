#!/usr/bin/env bash
# H5 matched-k ablation chain: does droppability differ by block type
# (full vs linear attention) beyond what block influence already predicts?
#
# Stage A (singles, on+off-policy):
#   new linear by BI rank: 57 17 37 58
#   BI-matched linear controls: 33 (BI .0526 ~ full 15's .0523), 30 (.0547 ~ 55's .0550)
#   top unscreened full-attention: 55 39 23 47
# Stage B (matched-k sets, composed deterministically by scripts/h5_compose_sets.py
#   from stage-A off-policy ranking): k8_linear, k2_full, k4_mixed, k4_full.
#
# Idempotent and crash-safe: run_kl_screen checkpoints per item inside each
# --out and refuses foreign probe fingerprints; rerunning this script resumes
# where it stopped. Est. ~7h GPU total (on-policy ~8 min/candidate pass,
# off-policy ~22 min; reference logits amortized across candidates per stage).
set -euo pipefail
cd "$(dirname "$0")/.."
REF=models/Bonsai-27B-mlx-1bit
ON=experiments/calibration/probes_onpolicy.json
OFF=experiments/calibration/probes_offpolicy.json
R=experiments/kl_screen/results

SINGLES=""
for b in 57 17 37 58 33 30 55 39 23 47; do SINGLES="$SINGLES --drop $b"; done

commit() {
  git add "$@"
  git commit -q -m "H5: $(basename "${1%.json}") complete" || true
}

echo "=== H5 stage A: singles on-policy"
uv run python experiments/kl_screen/run_kl_screen.py \
  --reference $REF --probes $ON --out $R/h5_singles_onpolicy.json $SINGLES
commit $R/h5_singles_onpolicy.json

echo "=== H5 stage A: singles off-policy"
uv run python experiments/kl_screen/run_kl_screen.py \
  --reference $REF --probes $OFF --out $R/h5_singles_offpolicy.json $SINGLES
commit $R/h5_singles_offpolicy.json

echo "=== H5 stage B: compose matched-k sets"
SETS=$(uv run python scripts/h5_compose_sets.py)
echo "sets: $SETS"

echo "=== H5 stage B: sets on-policy"
uv run python experiments/kl_screen/run_kl_screen.py \
  --reference $REF --probes $ON --out $R/h5_sets_onpolicy.json $SETS
commit $R/h5_sets_onpolicy.json $R/h5_selection.json

echo "=== H5 stage B: sets off-policy"
uv run python experiments/kl_screen/run_kl_screen.py \
  --reference $REF --probes $OFF --out $R/h5_sets_offpolicy.json $SETS
commit $R/h5_sets_offpolicy.json

osascript -e 'display notification "H5 matched-k KL chain complete." with title "bonsai-fold" sound name "Glass"' 2>/dev/null || true
echo "=== H5 chain complete"
