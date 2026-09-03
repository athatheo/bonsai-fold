# bonsai-fold — CANONICAL RESULTS (compaction-proof record)

Update this file whenever a new confirmed result lands. Numbers here are
final and verified; provenance chain: docs/LAB_NOTEBOOK.md (append-only) →
experiments/*/results (raw JSONs) → experiments/paper_tables/tables.md
(generated). If this file and a raw JSON disagree, the JSON wins.

Last updated: 2026-09-03 (head promotion REJECTED at bench, -3.5; 6th screen-vs-bench instance; frontier final: folded-709 / repaired-709 / combined-3.82GB. ALL experimental lines closed — paper remains).

## Headline artifacts

| artifact | size | vs 4.71 GB LM | macro (500-item bench) | status |
|---|---|---|---|---|
| Bonsai-27B-1bit (reference) | 4.71 GB | — | .8413 | baseline (our harness) |
| **Bonsai-27B-1bit-folded-709** | 4.2 GB | −709 MB / −15.4% | **.7963** (GSM8K .910 = ref) | SHIPPING, byte-identical |
| Bonsai-27B-1bit-groupc-scaleq8 | 4.19 GB | −612 MB / −13% | **.8363** (−0.5, within noise) | FLAGGED (value-modifying) |
| **Bonsai-27B-1bit-folded709-scaleq8** | **3.82 GB** | −888 MB / −18.9% | **.7900** (−0.6 vs folded-709) | FLAGGED combined headline |
| **folded-709 + shadow repair** | 4.2 GB (same bytes) | −709 MB / −15.4% | **.8063** (+1.0 vs folded-709) | FLAGGED repair (Group R v2) |

- folded-709 = bias-strip + drop blocks {16,12,13,9} + drop attn sublayers
  {37,38,58}. Full-vocab logits bit-identical to benched views; pack bench
  reproduced 500/500 per-item. All surviving weights byte-identical.
- groupc-scaleq8 = nobias + per-row 8-bit scale plane (no structural cuts).
  KL 2.7e-06 on / 9.2e-06 off; flips vs ref 13/18 bidirectional; report
  SEPARATELY from byte-identical lines always (CLAUDE.md flagged-arm rule).

## Engineering wins (bit-exact, measured)

- **B1 bias plane**: biases == f16(−scales/2) proven exhaustively. Disk
  −420 MB (nobias pack). Kernel stage: MLX fork mode "affine-derived"
  computes bias in-register; 6 matmul routes + dequantize bit-identical;
  **resident RAM −420,225,024 B measured** (3.919 → 3.527 GB on the 27B).
  PrismML's whitepaper §4.3 acknowledges the redundancy as "a current MLX
  limitation" and defers it — B1 is the implementation they deferred.
- Loader: derived kernels are DEFAULT for packs stamped
  bonsai_bias_plane=="derived" (load_bonsai(derived_kernels=False) forces
  materialized path).

## Bench ladder, byte-identical arm (500 items; all in results/)

reference .8413 | k2 .8125 (−118 MB) | k4 .8175 (−236) | k6 .7762 (−360) |
k8 .7500 (−478) | folded-709 .7963 (−709). Damage law: truncation floor +
accelerating knowledge term; same-type stacking pays an interaction tax
(1.09–1.13× on-policy for cross-pool, compounding for same-pool).

## A6 evolutionary search — KL frontier does NOT transfer to bench (measured 2026-08-21)

KL level (100 probes both regimes): search dominates/ties hand-built at
every tier (off-policy margins 5%→14%→40%): T350 364.6 MB @ .0314/.176;
T350-s2 374.8 @ .0322/.187; T450 456.8 @ .0504/.233. BUT the T350 champion
500-item BENCH: **macro .7625 vs byte-matched k6 .7762 (−1.4 pts)** —
GSM8K .880 (=k6) but MATH .610 (k6 .640), IFEval .830 (k6 .860). Verdict:
**the EA Goodharted its 24-probe KL fitness** — screen-frontier dominance
is real at the KL level and does not survive downstream generation. Third
instance of the screens-are-not-benches lesson (block-1 ambush, A4 keep-set,
now A6). Hand configs selected on the bench-anchored ladder transfer
better. Champion genomes in experiments/evosearch/best_t*.json; T350 spec:
drop_block {5,13,16,36} + drop_attn {38,57,58} + drop_mlp {4,12}.

## Flagged arm 2 — Group R + the scale knee (weight modification, 2026-08-26/27)

0a. **REPAIR SCOPE MAPPED (extensions benched 2026-09-01)**: the confirmed
   repair does NOT extend. (i) Combined artifact (folded709-scaleq8) +
   refit repair: paired screen −7.1/−9.0% (identical profile to the
   confirmed repair) but **bench .7825 vs .7900 unrepaired (−0.75)** —
   repair does not compose with scale quantization; FIFTH
   screens-vs-benches instance and the first where a paired same-family
   screen misled. (ii) k6 + 4-site repair: screen flat-on/−20%-off,
   **bench .7750 vs .7762 (flat)** — the recipe does not scale to deeper
   folds (sites saturate the |dW|/|w| cap). Repair's confirmed regime:
   moderate structural damage, one perturbation type, sites below the cap.
0. **REPAIR CONFIRMED (v2, benched 2026-08-29)**: linear-shadow absorption
   with in-format requantization — fit each dropped cluster's static linear
   shadow (ridge, calibration-only, |dW|/|w| capped at 0.25), absorb into 3
   surviving out_proj modules, requantize to 1-bit g128 (0.8-1.7% sign
   flips, <4.5% scale drift, ZERO added bytes). Held-out screen −7.2% on /
   −9.0% off; **bench .8063 vs folded-709 .7963 (+1.0 macro at identical
   bytes; IFEval +3, MATH +1, MMLU +1, GSM8K −1; flips +35/−30)**. Damage
   vs reference shrinks −4.5 → −3.5 pts. First demonstration of
   training-free in-format repair of a 1-bit model. Planes:
   experiments/groupr/shadow_folded709.npz (+ normal equations for refits).
1. **Repair negative (v1)**: closed-form per-channel scale-gain repair of
   folded-709 (calibration-fitted, 3 sites) FAILS the paired held-out
   screen (+1.3% on-policy). Mechanism: RMSNorm renormalization already
   absorbs static scale effects — the loss from dropped modules is
   token-dependent directional content, unreachable without adding bytes.
2. **Scale knee (benched)**: 8-bit −0.5 pts (floor, shipped); 6-bit screen
   9.1e-06 (plausibly free, UNBENCHED); **4-bit BENCH-DEGRADED −2.9 pts**
   despite screening 76x below k2 — establishing that **KL→bench damage
   mappings differ BY OPERATOR FAMILY** (metadata noise harms far more per
   screen-nat than structural deletion; 4th screens-vs-benches instance).

## Confirmed negatives (do not revisit)

1. Width/channel trimming: flat (no low-salience tail at 1-bit).
2. Vocab trimming: rejected BOTH sides (input OOV fat tail; mixed-domain
   emission needs 87–98% of rows; single-domain keep-sets fail task
   transfer end-to-end: +26% gen tokens, 2.3× truncations, caught only by
   pack bench — teacher-forced identity does NOT imply generation fidelity).
3. Byte-matched merging loses to dropping (promotion > sign-election, both
   lose byte-matched; promotion merge remains in-scope as 2-bit blocks).
4. Sign plane: 8.000 bits/byte measured entropy — incompressible.

## Key facts (re-derivable but expensive)

- 1-bit g128: w = s·q + b, q∈{0,1}, s = 2s_g, b = −s_g ⇒ weights ±s_g.
- mlx-lm qwen3_5 types blocks POSITIONALLY — always load folded packs via
  bonsaifold.loader.load_bonsai (layer_types-aware). Never stock loader.
- Disable speculative decoding on folded models (DSpark taps fixed layers).
- Sampling: temp 0.7, top-p 0.95, top-k 20, thinking on. Bench items are
  seeded per-item (deterministic reproduction when logits bit-identical).
- Whitepaper reference copy: scratchpad (re-download:
  github.com/PrismML-Eng/Bonsai-demo). Their 1-bit avg 76.11 (EvalScope,
  H100/vLLM) — different harness; use OUR within-format rows only.
- PrismML's proprietary part: the FP16→binary conversion method (weights
  are Apache; the transform is unreleased "Caltech IP").

## Extension queue (decided 2026-08-19 by Thanasis)

Q1. Combined flagged artifact folded709-scaleq8: DONE 2026-08-20.
    3.82 GB decimal (−888 MB / −18.9% vs original). Screen 3.1e-06 on /
    9.9e-06 off vs folded-709; **bench macro .7900 vs folded-709 .7963**
    (−0.6 pts within noise; GSM8K .870, MATH .690, IFEval .860, MMLU .740;
    flips 11/12 bidirectional, +1.2% gen tokens, 0 truncations). The
    scale-quant increment costs the same ~noise on the folded pack as on
    the unfolded one — compositionality confirmed end-to-end. FLAGGED line.
Q2. A6 T350 champion bench: DONE 2026-08-21 — macro .7625, BELOW
    byte-matched k6 (.7762): the EA Goodharted its KL fitness (see A6
    section above). The searched frontier stays a KL-level result.
Q3. Generality (dense Bonsai-8B): CLOSED 2026-08-22. The map's SHAPE is
    universal — boundary protection (b0 7.7 nats, b35 1.3-3.4, b1 both
    regimes), mid-depth slack, two-regime disagreement, additive
    composition (tax 0.97-1.20x, matching the 27B band) all replicate on a
    DENSE model at 8B. The exploitable slack does NOT: cheapest single is
    6x the 27B's, and benches confirm brutally — 8B ref .7163 (non-thinking
    scoring, --no-think); k2-analog {26,17} .6737 (−4.3 pts / 33 MB);
    k4-analog {26,17,21,31} .5687 (−14.8 pts / 66 MB) vs the 27B's k4 −2.4
    pts / 236 MB. **Foldability is a property of over-provisioned depth,
    not of 1-bit models per se.** Cross-model KL→damage curve hypothesis
    TESTED AND REJECTED (k2-analog predicted −7-9 by the 27B curve,
    measured −4.3): the mapping is monotone within a model only. Bench
    scoring gotcha recorded: pre-27B family has no thinking mode.
Q4. Elastic-depth family: DONE 2026-08-19. Spec generated from measured
    data (experiments/elastic/elastic_spec.json): nested 10-op order, anchor
    prefixes = k2/.8125, k4/.8175, folded-709/.7963 (benched) and k4+s8-tier
    (KL .0339/.171). bonsaifold.elastic.elastic_view(model, budget_mb=…)
    serves any tier zero-copy from the one nobias pack; guards refuse folded
    packs (renumbering trap) and sub-first-tier budgets. Non-anchor prefixes
    are additive-law-bounded, not individually benched (documented).
Q5. Attention-head map (27B): CLOSED 2026-08-24. The head axis is RICH —
    the width-negative kill criterion was NOT met. 64 KV-group singles (16
    full-attn blocks x 4 GQA groups, ~3.8 MB each, byte-identical row/col
    surgery, adapter bit-exact on empty drop): 25x KL spread (.00017 to
    .0042 on-policy), clean depth gradient with EARLY full-attention blocks
    (3/7/11/23) cheapest — GQA over-provisioning where whole-sublayer drops
    were never cheap. Two-regime rank agreement weak (Spearman .354; both
    regimes mandatory again). Sets (max 2 groups/block): H8 30.4 MB @
    .00436 on / .0318 off (tax 1.08/1.06x); H12 45.6 MB @ .00885 / .0450
    (1.18/1.05x) — the additive law extends to head granularity. H8 does
    not overlap folded-709's operators. PROMOTION BENCHED 2026-09-03 and
    REJECTED: 709+H8 screened sub-additive (.0310/.1546, below k6 both
    regimes) but benched .7612 — −3.5 pts vs folded-709 for 30 MB, below
    k6. SIXTH screens-vs-benches instance: KV-group removal is a
    steep-slope operator family (dense attention-pattern perturbation),
    like scale metadata, unlike structural deletion. The head map stays a
    screen-level structural finding; folded-709 remains the byte-identical
    frontier.
Q6. AFTER queue completes: value-modifying weight exploration (scope with
    Thanasis first; extends the Group C flagged precedent — CLAUDE.md
    amendment required before any weight-value edit).

Rule: any new confirmed result updates THIS file + LAB_NOTEBOOK + tables.
