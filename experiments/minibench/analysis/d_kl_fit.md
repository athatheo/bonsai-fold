# d_kl_fit — KL→damage mapping and gate recalibration (2026-07-21)

Anchors (on-policy set KL from topk_onpolicy.json; accuracies recomputed from items, matched stored values exactly; N=500 paired items).

| config | KL_on (nats) | KL_off (nats) | GSM8K | MATH500 | IFEval | MMLU | macro | drop (pts) |
|---|---|---|---|---|---|---|---|---|
| reference | 0.00000 | 0.00000 | 0.910 | 0.720 | 0.930 | 0.805 | 0.8413 | 0.00 |
| k2 | 0.00756 | 0.03304 | 0.920 | 0.660 | 0.880 | 0.790 | 0.8125 | 2.88 |
| k6 | 0.03111 | 0.20461 | 0.880 | 0.640 | 0.860 | 0.725 | 0.7762 | 6.50 |
| k8 | 0.06426 | 0.39148 | 0.870 | 0.720 | 0.790 | 0.620 | 0.7500 | 9.12 |

## Fits: macro drop (points) vs on-policy KL (4 points — form underdetermined)

| form | params | residuals @ (ref,k2,k6,k8) pts | RMSE | R2 |
|---|---|---|---|---|
| drop = b*KL | {'slope_pts_per_nat': 157.207} | [0.0, 1.6873, 1.6091, -0.9774] | 1.2641 | 0.86745 |
| drop = a + b*KL | {'intercept_pts': 1.1925, 'slope_pts_per_nat': 133.395} | [-1.1925, 0.6747, 1.1574, -0.6397] | 0.9521 | 0.9248 |
| drop = c*sqrt(KL) | {'coef_pts_per_sqrtnat': 36.041} | [0.0, -0.2576, 0.143, -0.0112] | 0.1474 | 0.9982 |
| drop = a + c*sqrt(KL) | {'intercept_pts': -0.0895, 'coef_pts_per_sqrtnat': 36.49} | [0.0895, -0.2071, 0.1533, -0.0356] | 0.1376 | 0.99843 |

Recommended: **sqrt_origin** (drop = c·sqrt(KL)). It is the only 1-parameter form whose residuals are <0.3 pts at every anchor (linear-origin misses k2 by +1.7 pts); the anchors are clearly concave (marginal pts/nat fall 380→209→142 across k2/k6/k8). Near KL→0 the sqrt form over-predicts damage — the conservative direction for an accept gate. Linear-origin kept as sensitivity.

Paired item bootstrap (B=10000, seed 20260721): linear-origin slope = 1.5721 pts per 0.01 nats, 95% CI [0.9954, 2.1445]; sqrt coef = 36.041 pts/sqrt-nat, 95% CI [22.781, 49.053]. KL values held fixed (probe noise not propagated).

Macro-drop 95% CIs (paired bootstrap, points): k2: [0.117, 5.673], k6: [3.725, 9.38], k8: [5.391, 12.794]

## Per-task fits (drops in points at k2/k6/k8)

| task | drops (k2,k6,k8) | sqrt coef (pts/sqrt-nat) | lin slope (pts/0.01nats) | sqrt RMSE (pts) | note |
|---|---|---|---|---|---|
| gsm8k | [-1.0, 3.0, 4.0] | 14.148 | 0.665 | 1.1616 | shallow; k2 slightly above ref |
| math500 | [6.0, 8.0, 0.0] | 18.776 | 0.571 | 3.991 | NON-MONOTONE (k8 = ref); fit is descriptive only, high residual |
| ifeval | [5.0, 7.0, 14.0] | 50.698 | 2.241 | 1.1665 | steep, clean monotone staircase |
| mmlu | [1.5, 8.0, 18.5] | 60.539 | 2.811 | 2.7959 | steepest; drives k8 damage |

## Paired McNemar exact tests (vs reference)

| comparison | scope | Δacc (pts) | discordant (ref+/cfg-, ref-/cfg+) | p (exact, 2-sided) |
|---|---|---|---|---|
| k2_vs_reference | all | -2.6 | (34, 21) | 0.105 |
| k2_vs_reference | gsm8k | 1.0 | (2, 3) | 1.0 |
| k2_vs_reference | math500 | -6.0 | (8, 2) | 0.109 |
| k2_vs_reference | ifeval | -5.0 | (8, 3) | 0.227 |
| k2_vs_reference | mmlu | -1.5 | (16, 13) | 0.711 |
| k6_vs_reference | all | -6.8 | (49, 15) | 2.44e-05 |
| k6_vs_reference | gsm8k | -3.0 | (5, 2) | 0.453 |
| k6_vs_reference | math500 | -8.0 | (8, 0) | 0.00781 |
| k6_vs_reference | ifeval | -7.0 | (9, 2) | 0.0654 |
| k6_vs_reference | mmlu | -8.0 | (27, 11) | 0.0139 |
| k8_vs_reference | all | -11.0 | (81, 26) | 9.37e-08 |
| k8_vs_reference | gsm8k | -4.0 | (7, 3) | 0.344 |
| k8_vs_reference | math500 | 0.0 | (6, 6) | 1.0 |
| k8_vs_reference | ifeval | -14.0 | (18, 4) | 0.00434 |
| k8_vs_reference | mmlu | -18.5 | (50, 13) | 3.02e-06 |
| math500_k8_vs_k6 | math500 | 8.0 | (3, 11) | 0.0574 |
| math500_k8_vs_reference | math500 | 0.0 | (6, 6) | 1.0 |

## Interaction tax and off/on ratio (measured)

| set | KL_on | Σ singles_on | tax | KL_off/KL_on |
|---|---|---|---|---|
| k2 | 0.00755 | 0.00756 | 1.000 | 4.374 |
| k4 | 0.01770 | 0.01507 | 1.175 | 4.882 |
| k6 | 0.03111 | 0.02331 | 1.334 | 6.577 |
| k8 | 0.06426 | 0.03422 | 1.878 | 6.092 |

Off/on ratio: mean 5.481, pooled 5.932, range 4.37–6.58 (k-dependent: rises with k).

## Gate recalibration (replaces provisional 0.05-nats on-policy screen)

The old 0.05-nats gate maps to a predicted macro drop of ~8.059 pts (sqrt) / 7.86 pts (linear) — roughly 5x too lax for a 1.5-pt Gate B.

| Gate B budget | (a) on-policy set budget (nats) [95% CI] | (b) off-policy ceiling (nats) | (c) k8 single-block screen (nats) | linear-fit sensitivity (set / single) | k8 feasible? |
|---|---|---|---|---|---|
| 1.5 pts | 0.00173 [0.00093, 0.00434] | 0.00949 | 0.000115 | 0.00954 / 0.000635 | NO |
| 3.0 pts | 0.00693 [0.00374, 0.01734] | 0.03798 | 0.000461 | 0.01908 / 0.001270 | NO |

Single-block screen uses budget / (tax(8)·8) with measured tax(8) = 1.878. Best measured single block is drop[12] at 0.00330 nats — above BOTH per-block thresholds, so **no k~8 set from the current inventory can pass Gate B at 1.5 or even 3.0 points**; beyond-k8 greedy selection under Gate B is ruled out by this calibration. At 3.0 pts, k2 passes directly on measured accuracy (2.88 pts); k4 (KL_on 0.0177) is where the two fit forms disagree most (predicted 4.8 vs 2.8 pts) — benchmarking k4 would discriminate the forms.

## MMLU clustering sensitivity

k8 MMLU drop = 18.5 pts; iid-item bootstrap CI [11.111, 25.743] vs subject-cluster bootstrap CI [9.722, 27.369] (57 subjects). Cluster CI is the honest one for MMLU given subject-clustered item order.

## Caveats

- Only 4 anchors (3 non-zero): functional form is underdetermined; sqrt vs linear differ up to ~2 pts in the unmeasured k3–k5 region and near zero.
- KL x-values are point estimates from 100 probes; bootstrap propagates only bench-item noise, so budget CIs are too narrow.
- Per-task fits use N=100 (200 for MMLU); Wilson CIs are ±7–9 pts — underpowered.
- math500 is non-monotone (k8 = reference): the sqrt fit reports it as high residual; do not use the math500 fit predictively.
- Off/on ratio is k-dependent (4.4→6.6); the mean (~5.5x) ceiling is a guard, not a law.
- Budgets extrapolate BELOW the smallest measured anchor (k2 = 0.0076 nats): no direct accuracy measurement exists in the accept region.
