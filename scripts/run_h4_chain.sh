#!/usr/bin/env bash
# H4 chain: does format-native merging beat dropping at matched depth cut?
#
# Stage A (zero-copy views, reference logits amortized): singles for the
#   shortlist's unmeasured blocks {18,20,60,61,62} + drop-BOTH sets for each
#   shortlist pair {4,5} {17,18} {18,20} {60,61}. On+off-policy.
# Stage B: build/verify merged packs (4 pairs x sign_election+promotion);
#   idempotent via exact recompute (experiments/merge/build_h4_packs.py).
# Stage C: KL screen each merged pack via --candidate-pack (each loads its
#   own 5GB model alongside the reference), on+off-policy, one output file
#   per pack so screens resume independently.
#
# H4 readout per pair: KL(merge) vs min single vs both-dropped set.
# Crash-safe: every stage resumes; rerun this script after any interruption.
# Est. ~12h GPU total. Runs under launchd com.bonsaifold.chain supervision.
set -euo pipefail
cd "$(dirname "$0")/.."
REF=models/Bonsai-27B-mlx-1bit
ON=experiments/calibration/probes_onpolicy.json
OFF=experiments/calibration/probes_offpolicy.json
R=experiments/kl_screen/results

commit() { git add "$1"; git commit -q -m "$2" || true; }

uv run python experiments/runtime_validation/validate_runtime.py >/dev/null \
  || { echo "runtime validation FAILED — stop"; exit 1; }

VIEWS="--drop 18 --drop 20 --drop 60 --drop 61 --drop 62 \
  --drop 4,5 --drop 17,18 --drop 18,20 --drop 60,61"

echo "=== H4 stage A: shortlist singles+sets on-policy"
uv run python experiments/kl_screen/run_kl_screen.py \
  --reference $REF --probes $ON --out $R/h4_views_onpolicy.json $VIEWS
commit $R/h4_views_onpolicy.json "H4: views on-policy complete"

echo "=== H4 stage A: shortlist singles+sets off-policy"
uv run python experiments/kl_screen/run_kl_screen.py \
  --reference $REF --probes $OFF --out $R/h4_views_offpolicy.json $VIEWS
commit $R/h4_views_offpolicy.json "H4: views off-policy complete"

echo "=== H4 stage B: build merged packs"
uv run python experiments/merge/build_h4_packs.py

echo "=== H4 stage C: screen merged packs"
for pair in 4-5 18-20 17-18 60-61; do
  for op in sign_election promotion; do
    PACK=models/merged_${pair}_${op}
    for regime in on off; do
      PROBES=$ON; [ "$regime" = off ] && PROBES=$OFF
      OUT=$R/h4_merge_${pair}_${op}_${regime}policy.json
      echo "--- screening $PACK ($regime-policy)"
      uv run python experiments/kl_screen/run_kl_screen.py \
        --reference $REF --probes $PROBES --out "$OUT" --candidate-pack "$PACK"
      commit "$OUT" "H4: merge ${pair} ${op} ${regime}-policy complete"
    done
  done
done

osascript -e 'display notification "H4 merge-vs-drop chain complete." with title "bonsai-fold" sound name "Glass"' 2>/dev/null || true
echo "=== H4 chain complete"
